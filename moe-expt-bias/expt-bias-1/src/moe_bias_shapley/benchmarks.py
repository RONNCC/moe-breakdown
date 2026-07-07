"""Benchmark loading: StereoSet, BBQ, WinoGender, C-Eval (fairness-adjacent).

All loaders return a list of PromptPair — stereotype vs anti-stereotype sentence
pairs from the respective benchmark, plus metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)

# Local cache dir for benchmark artifacts fetched outside `datasets` (e.g. the
# WinoGender TSV, which is only distributed via GitHub, not a maintained HF
# dataset repo). Kept off scratch/home per the cluster storage policy — this
# is a small (~1MB) static text file, safe under $HOME/.cache.
_CACHE_DIR = "~/.cache/moe_bias_shapley"


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

    # `gold_label` is a nested List(ClassLabel(...)) — nested ClassLabel fields
    # are NOT auto-decoded to strings on row access (unlike top-level ClassLabel
    # columns), so iterating rows yields raw ints. Resolve the int->name mapping
    # from the dataset's own feature schema instead of hardcoding it, in case
    # the label order ever changes upstream.
    label_names = ds.features["sentences"]["gold_label"].feature.names

    pairs: List[PromptPair] = []
    for item in ds:
        sentences = item["sentences"]
        labels = sentences["gold_label"]
        texts = sentences["sentence"]

        stereo_text = None
        anti_text = None
        for text, label in zip(texts, labels):
            label_name = label_names[label] if isinstance(label, int) else label
            if label_name == "stereotype":
                stereo_text = text
            elif label_name == "anti-stereotype":
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
# WinoGender                                                                   #
# =========================================================================== #

# WinoGender (Rudinger et al. 2018) is not distributed as a maintained HF
# `datasets` repo with stable schema — the canonical source is the authors'
# GitHub release of pre-rendered sentence templates. We fetch that TSV
# directly (small, ~1MB, static) rather than guessing at HF mirror repo ids
# (several exist but are unofficial/inconsistently schema'd).
_WINOGENDER_TSV_URL = (
    "https://raw.githubusercontent.com/rudinger/winogender-schemas/"
    "master/data/all_sentences.tsv"
)


def _fetch_winogender_tsv(cache_dir: str = _CACHE_DIR) -> str:
    """Download (and cache) the WinoGender all_sentences.tsv file."""
    import urllib.request
    from pathlib import Path

    cache_path = Path(cache_dir).expanduser() / "winogender_all_sentences.tsv"
    if cache_path.exists():
        return cache_path.read_text()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Fetching WinoGender sentence templates from %s", _WINOGENDER_TSV_URL)
    with urllib.request.urlopen(_WINOGENDER_TSV_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    cache_path.write_text(text)
    return text


def load_winogender(
    max_items: Optional[int] = None,
    cache_dir: str = _CACHE_DIR,
) -> List[PromptPair]:
    """Load WinoGender coreference-gender-stereotype sentences.

    Each `sentid` in the released TSV encodes
    `{occupation}.{other_participant}.{answer}.{gender}.txt`, where `gender`
    is one of male/female/neutral and `answer` (0/1) picks which of the two
    template variants (occupation-focused vs participant-focused) is used.
    We hold occupation/participant/answer fixed and pair the male- and
    female-pronoun sentence variants as a counterfactual PromptPair — this
    is a direct male-vs-female logit-gap test (Sec 2.3's "group-conditional
    logit difference for counterfactual prompt pairs"), not the original
    WinoGender paper's BLS-occupation-statistics correlation metric; that
    distinction is noted here since it's a simplification of the official
    scoring, consistent with how load_bbq/load_stereoset already adapt their
    source benchmarks' native format to this study's PromptPair schema.
    """
    text = _fetch_winogender_tsv(cache_dir)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("WinoGender TSV fetch returned no content")

    header = lines[0].split("\t")
    assert header[:2] == ["sentid", "sentence"], f"Unexpected WinoGender TSV header: {header}"

    # sentid -> sentence text
    by_id: dict[str, str] = {}
    for line in lines[1:]:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sentid, sentence = parts
        by_id[sentid.strip()] = sentence.strip()

    pairs: List[PromptPair] = []
    seen_keys: set[str] = set()
    for sentid, male_sentence in by_id.items():
        if not sentid.endswith(".male.txt"):
            continue
        key_prefix = sentid[: -len(".male.txt")]
        if key_prefix in seen_keys:
            continue
        female_sentid = f"{key_prefix}.female.txt"
        female_sentence = by_id.get(female_sentid)
        if female_sentence is None:
            continue
        seen_keys.add(key_prefix)

        # key_prefix = "{occupation}.{participant}.{answer}"
        parts = key_prefix.split(".")
        occupation = parts[0] if parts else "unknown"

        pairs.append(PromptPair(
            stereo=male_sentence,
            anti_stereo=female_sentence,
            bias_type="gender",
            target=occupation,
            source="winogender",
            item_id=key_prefix,
            extra={"template_key": key_prefix},
        ))

        if max_items is not None and len(pairs) >= max_items:
            break

    log.info("Loaded %d WinoGender pairs", len(pairs))
    return pairs


# =========================================================================== #
# C-Eval (fairness-adjacent subset)                                            #
# =========================================================================== #

# C-Eval [Huang et al. 2023] is a general Chinese-language knowledge/reasoning
# exam benchmark (52 subjects) — it does not ship a dedicated "fairness"
# category with native stereotype/anti-stereotype contrastive pairs the way
# StereoSet/BBQ/WinoGender do. We use the subjects closest to social/ethical
# content (ideological_and_moral_cultivation, education_science) and build a
# PromptPair from each question's correct vs. a plausible distractor answer.
# This is a materially weaker construct than the other three loaders (it
# measures a knowledge/confidence gap, not a demographic-stereotype gap), and
# with the corrected model ladder now excluding all Chinese-origin models
# (Chinese-origin models disallowed on GT ICE — see study_design_C1_C4.md
# Sec 2.1), there is no model in the ladder for which Chinese-language
# fairness testing is a natural fit. This loader is therefore implemented for
# completeness/future cross-lingual work but is NOT included in any active
# study config's `benchmarks:` list — see study-catalog.txt for the rationale.
_CEVAL_SUBJECTS = ["ideological_and_moral_cultivation", "education_science"]


def load_ceval_fairness(
    split: str = "val",
    subjects: Optional[List[str]] = None,
    max_items: Optional[int] = None,
) -> List[PromptPair]:
    """Load a fairness-adjacent subset of C-Eval as correct-vs-distractor pairs.

    NOTE: see module-level comment above — this is a knowledge-gap proxy, not
    a true demographic-stereotype benchmark, and is deprioritized/excluded
    from the default benchmark set now that the model ladder is all-English.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install `datasets` to load C-Eval: pip install datasets") from exc

    subjects = subjects or _CEVAL_SUBJECTS
    pairs: List[PromptPair] = []

    for subject in subjects:
        log.info("Loading C-Eval subject=%s split=%s …", subject, split)
        ds = load_dataset("ceval/ceval-exam", subject, split=split)

        for item in ds:
            answer_key = str(item.get("answer", "")).strip().upper()
            if answer_key not in ("A", "B", "C", "D"):
                continue
            choices = {k: item.get(k, "") for k in ("A", "B", "C", "D")}
            correct = choices.get(answer_key, "")
            distractor_key = next((k for k in ("A", "B", "C", "D") if k != answer_key and choices.get(k)), None)
            if not correct or distractor_key is None:
                continue
            distractor = choices[distractor_key]

            question = item.get("question", "")
            pairs.append(PromptPair(
                stereo=f"{question} {distractor}",
                anti_stereo=f"{question} {correct}",
                bias_type=subject,
                target=subject,
                source="ceval",
                item_id=str(item.get("id", "")),
                extra={"answer_key": answer_key, "distractor_key": distractor_key},
            ))

            if max_items is not None and len(pairs) >= max_items:
                break
        if max_items is not None and len(pairs) >= max_items:
            break

    log.info("Loaded %d C-Eval pairs across %d subjects", len(pairs), len(subjects))
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
        elif name == "winogender":
            pairs.extend(load_winogender(max_items=per_bench))
        elif name == "ceval":
            pairs.extend(load_ceval_fairness(max_items=per_bench))
        else:
            log.warning("Unknown benchmark %r — skipping", name)

    if max_items is not None:
        pairs = pairs[:max_items]

    log.info("Total prompt pairs: %d", len(pairs))
    return pairs
