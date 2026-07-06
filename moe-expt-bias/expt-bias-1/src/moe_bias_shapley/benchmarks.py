"""Benchmark loading: StereoSet, BBQ, WinoGender.

All loaders return a list of PromptPair — stereotype vs anti-stereotype sentence
pairs from the respective benchmark, plus metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class PromptPair:
    """A matched stereotype / anti-stereotype sentence pair."""
    stereo: str           # stereotype sentence (full)
    anti_stereo: str      # anti-stereotype sentence (full)
    bias_type: str        # gender | race | religion | profession
    target: str           # the social-group target word
    source: str           # which benchmark this came from
    item_id: str = ""     # original item id (for traceability)
    extra: dict = field(default_factory=dict)  # benchmark-specific extras


# =========================================================================== #
# StereoSet                                                                    #
# =========================================================================== #

def load_stereoset(
    subset: str = "intrasentence",  # intrasentence | intersentence
    split: str = "validation",
    max_items: Optional[int] = None,
) -> List[PromptPair]:
    """Load StereoSet from HuggingFace datasets.

    Returns one PromptPair per StereoSet item that has both a stereotype and
    anti-stereotype sentence.  Unrelated sentences are discarded.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install `datasets` to load StereoSet: pip install datasets") from exc

    log.info("Loading StereoSet (%s / %s) …", subset, split)
    # NOTE: the legacy no-namespace "stereoset" repo id no longer resolves under
    # current huggingface_hub URI rules (namespace/name required). Use the
    # McGill-NLP parquet mirror instead, which has identical fields/schema.
    ds = load_dataset("McGill-NLP/stereoset", subset, split=split)

    pairs: List[PromptPair] = []
    for item in ds:
        sentences = item["sentences"]
        labels = sentences["gold_label"]
        texts = sentences["sentence"]

        stereo_text = None
        anti_text = None
        for text, label in zip(texts, labels):
            if label == "stereotype":
                stereo_text = text
            elif label == "anti-stereotype":
                anti_text = text

        if stereo_text is None or anti_text is None:
            continue   # skip items missing one of the labels

        pairs.append(PromptPair(
            stereo=stereo_text,
            anti_stereo=anti_text,
            bias_type=item.get("bias_type", "unknown"),
            target=item.get("target", ""),
            source="stereoset",
            item_id=item.get("id", ""),
            extra={"context": item.get("context", "")},
        ))

        if max_items is not None and len(pairs) >= max_items:
            break

    log.info("Loaded %d StereoSet pairs from %d items", len(pairs), len(ds))
    return pairs


# =========================================================================== #
# BBQ                                                                          #
# =========================================================================== #

# BBQ (heegyu/bbq mirror) is split into one config per social-bias category —
# there is no single "all categories" config, so we must iterate over these.
_BBQ_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]


def load_bbq(
    split: str = "test",
    categories: Optional[List[str]] = None,
    max_items: Optional[int] = None,
) -> List[PromptPair]:
    """Load BBQ (Bias Benchmark for QA) from HuggingFace.

    We only use ambiguous-context items where the model must choose between
    a biased and a neutral answer — these translate naturally to PromptPairs.

    `categories` selects which of BBQ's per-category configs to load (default:
    Gender_identity + Race_ethnicity, matching the demographic cohorts named in
    the study design Sec 3.5 — he/she and Black/White/Asian contexts). Pass
    `_BBQ_CATEGORIES` (all 11) for full coverage at higher cost.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install `datasets` to load BBQ: pip install datasets") from exc

    categories = categories or ["Gender_identity", "Race_ethnicity"]
    pairs: List[PromptPair] = []

    for category in categories:
        log.info("Loading BBQ category=%s split=%s …", category, split)
        # heegyu/bbq ships a legacy loading script (bbq.py) that current
        # `datasets` versions refuse to execute ("Dataset scripts are no
        # longer supported"). HF auto-converts script-based datasets to
        # parquet on a `refs/convert/parquet` branch, but that branch collapses
        # all per-category configs into a single "default" builder config —
        # categories are selected via `data_dir` (matching the original
        # per-category directory layout), not via the config-name argument.
        ds = load_dataset("heegyu/bbq", split=split, revision="refs/convert/parquet", data_dir=category)

        for item in ds:
            # Only use ambiguous context (context_condition == "ambig")
            if str(item.get("context_condition", "")).lower() != "ambig":
                continue

            context = item.get("context", "")
            question = item.get("question", "")
            prompt_base = f"{context} {question}"

            label = int(item.get("label", -1))     # index of the correct/unknown answer
            # additional_metadata is a nested struct in the official BBQ schema;
            # stereotyped_groups lives inside it.
            additional_metadata = item.get("additional_metadata") or {}
            stereotyped_groups = additional_metadata.get("stereotyped_groups", []) if isinstance(additional_metadata, dict) else []

            answers = [item.get(f"ans{i}", "") for i in range(3)]
            if label < 0 or label >= len(answers):
                continue

            unknown_ans = answers[label]
            biased_ans = next(
                (a for i, a in enumerate(answers) if i != label and a), None
            )
            if not biased_ans:
                continue

            stereo_prompt = f"{prompt_base} {biased_ans}"
            anti_prompt = f"{prompt_base} {unknown_ans}"

            pairs.append(PromptPair(
                stereo=stereo_prompt,
                anti_stereo=anti_prompt,
                bias_type=item.get("category", category),
                target=", ".join(stereotyped_groups) if isinstance(stereotyped_groups, list) else str(stereotyped_groups),
                source="bbq",
                item_id=str(item.get("example_id", "")),
                extra={"context_condition": "ambig"},
            ))

            if max_items is not None and len(pairs) >= max_items:
                break
        if max_items is not None and len(pairs) >= max_items:
            break

    log.info("Loaded %d BBQ pairs across %d categories", len(pairs), len(categories))
    return pairs


# =========================================================================== #
# Unified loader                                                                #
# =========================================================================== #

def load_benchmarks(
    names: List[str],
    max_items: Optional[int] = None,
) -> List[PromptPair]:
    """Load and concatenate one or more benchmarks by name."""
    pairs: List[PromptPair] = []
    per_bench = max_items  # apply limit per benchmark (generous)

    for name in names:
        if name == "stereoset":
            pairs.extend(load_stereoset(max_items=per_bench))
        elif name == "bbq":
            pairs.extend(load_bbq(max_items=per_bench))
        else:
            log.warning("Unknown benchmark %r — skipping", name)

    if max_items is not None:
        pairs = pairs[:max_items]

    log.info("Total prompt pairs: %d", len(pairs))
    return pairs
