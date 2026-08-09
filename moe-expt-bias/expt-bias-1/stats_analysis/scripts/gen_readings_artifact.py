#!/usr/bin/env python3
"""Build artifacts/bias_attribution_readings.json from moe_bias_report_draft.tex.

Every quote is pulled line-by-line from the tex using ONLY line numbers that
were verified live against the file (sed in session; see README note in data).
Each row asserts expected substrings, so a stale line number raises instead of
emitting wrong text.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
TEX = ROOT / "moe_bias_report_draft.tex"
OUT = ROOT / "artifacts" / "bias_attribution_readings.json"

src = TEX.read_text().splitlines()

# Only line numbers verified in-session against the live tex may be referenced.
VERIFIED = {
    29, 35, 37, 38, 39, 40, 42,        # abstract
    73, 75, 85, 90, 98, 97,            # intro RQs + hypotheses (src is 1-indexed; 96/98 are items)
    195, 199, 219,                     # ladder, table caption, Gemma footnote
    224, 229, 233, 245, 246, 252, 253, 248, 249, 250, 257, 261, 264, 273,  # results
    307, 311,                          # synergy findings
    336, 337, 339, 344, 349, 351, 356, # experiments 4/5
    360, 366, 370, 373, 391,           # robustness + conclusion
}


def line(n):
    if n not in VERIFIED:
        raise SystemExit(f"[guard] line {n} is not in the verified set; refusing")
    return src[n - 1].strip()


def extract(row):
    out = []
    for n in row["lines"]:
        t = line(n)
        if t:
            out.append(t)
    text = "\n".join(out)
    for token in row.get("expect", []):
        if token not in text:
            raise SystemExit(
                f"[{row['id']}] token {token!r} missing on lines {row['lines']}\n---\n{text}\n---"
            )
    return text


ROWS = [
    dict(id="sec-frameworks", kind="header",
         title="Framing A vs Framing B"),

    # ---------------- ABSTRACT ----------------
    dict(id="A1", kind="front-matter", section="abstract question",
         claim="RQ1 as posed: does bias attribution concentrate as routing sparsity increases?",
         lines=[29], expect=["more concentrated as routing sparsity increases"],
         fa="Neutral: the question belongs to Framing A; the answer (A3) rejects it.",
         fb="Answerable under B as well; the paper's verdict is 'no'.",
         conflict="None at question level; conflict unfolds elsewhere."),
    dict(id="A2", kind="method", section="abstract method sentence",
         claim="Instrument: gate-level 'routing-contrast proxy formulation' over a sparsity ladder of six open MoEs.",
         lines=[35], expect=["routing-contrast proxy formulation"],
         fa="Framing A stresses 'attribution'; this sentence already labels the instrument 'proxy'.",
         fb="Framing B's reading: the metric is routing contrast, i.e. correlational.",
         conflict="The word 'proxy' seeds the whole A-vs-B dispute."),
    dict(id="A3", kind="result", section="abstract bullet 1 — no sparsity scaling",
         claim="Headline: no monotone scaling of bias concentration; H in [0.88,0.92] vs 0.982 noise floor.",
         lines=[37], expect=["correlate monotonically", "0.88", "0.982"],
         metric="H, H_noise",
         fa="Directly rejects the core monotone premise of Framing A.",
         fb="Exactly the B-expected reading; both share the same numbers.",
         conflict="Unilateral ground: the numbers are not disputed between framings, the interpretation is."),
    dict(id="A4", kind="result", section="abstract bullet 2 — no expert-level localization",
         claim="rho ~ 0; selectivity low/negative; ITL ablation matches; curve = damage curve.",
         lines=[38], expect=["No Expert-Level", "0.24", "damage curve"],
         metric="rho, selectivity",
         fa="Refutes any single-expert or concentrated-bias claim in A.",
         fb="Core B statement, verbatim intact.",
         conflict="Hard conflict with expert-localization claims."),
    dict(id="A5", kind="result", section="abstract bullet 3 — early-layer synergy",
         claim="Early layers >70% dominated by nonlinear pairwise interaction.",
         lines=[39], expect=["Early-Layer Synergy", "70"],
         fa="Interactive attribution contradicts the singleton-expert reading.",
         fb="Supports B's collective-slot story.",
         conflict="Partial."),
    dict(id="A6", kind="result", section="abstract bullet 4 — demographic specificity",
         claim="Diffuse within each cohort but specific across cohorts.",
         lines=[40], expect=["Demographic", "subgroup-specific"],
         fa="'Diffuse in each cohort' again contradicts the concentration narrative.",
         fb="B-friendly conditional specificity.",
         conflict="None severe; tension lurks in 'diffuse within cohort'."),
    dict(id="A7", kind="result", section="abstract concluding causal paragraph",
         claim="'Social bias is systemic; single-expert debiasing is not viable'.",
         lines=[42], expect=["no separable expert-level substrate", "not a viable alignment strategy"],
         fa="Stand-in that Framing A would have to argue around.",
         fb="Directly gives B's own sentence.",
         conflict="Unambiguous sentence-level contradiction with concentratable-bias claims."),

    # ---------------- INTRO ----------------
    dict(id="B1", kind="question", section="RQ1 definitional line",
         claim="RQ1 restated in the introduction: does sparsity -> concentration?",
         lines=[73], expect=["RQ1 (Sparsity", "more concentrated in"],
         fa="Same question (Framing A).",
         fb="Framing B can answer it too."),
    dict(id="B2", kind="result", section="RQ1 verdict",
         claim="Verdict: REJECTED; GPT-OSS G=0.724 vs OLMoE G=0.646 despite 2x sparsity.",
         lines=[75], expect=["Rejected (H1 Rejected)", "Gini coefficient", "0.724", "0.646"],
         metric="H, G",
         fa="Empirical 'Rejected' of the Sparsity-to-Concentration claim.",
         fb="As expected."),
    dict(id="B3", kind="result", section="RQ2 verdict",
         claim="70.2-74.4% synergy; 'standing committees' statement.",
         lines=[85], expect=["Synergy Dominates", "70.2"],
         metric="synergy%",
         fa="Disproves the attributable-to-individual-experts reading.",
         fb="Exactly the B statement; includes a near-quote to B's own framing."),
    dict(id="B4", kind="result", section="RQ3 verdict",
         claim="Subgroup-specific subnetworks; D_JS 0.42 vs null 0.05.",
         lines=[90], expect=["Subgroup-Specific Subnetworks", "0.42"],
         metric="D_JS",
         fa="'Diffuse' + subgroup-specific only partially aligns with A's hub story.",
         fb="B-positive."),
    dict(id="B5", kind="verdict", section="hypotheses H1 & H0",
         claim="The pre-registered hypothesis pair: H1 sparsity-to-concentration (REJECTED), H0 selectivity collapse (STRONGLY SUPPORTED).",
         lines=[97, 98], expect=["Sparsity-to-Concentration", "REJECTED", "STRONGLY SUPPORTED"],
         fa="A maps directly onto H1, which this very block marks REJECTED.",
         fb="H0 marked STRONGLY SUPPORTED; that IS Framing B.",
         conflict="Block is self-verdict: H1=REJECTED, H0=STRONGLY SUPPORTED."),

    # ---------------- METHODS ----------------
    dict(id="C1", kind="method", section="ladder definition",
         claim="Ten permissively licensed models designed to isolate sparsity effects.",
         lines=[195], expect=["ten permissively licensed models"],
         metric="(ladder)"),
    dict(id="C2", kind="method", section="table 1 caption",
         claim="Dataset split note: 5000-pair main runs vs 400-pair pilot (GPT-OSS, Gemma).",
         lines=[199], expect=["The Sparsity-to-Density", "5000 pairs"],
         fa="Feed note only; both framings.",
         fb="Feed note only; both framings."),
    dict(id="C3", kind="method", section="table 1 footnote (Gemma)",
         claim="Gemma excluded: H=0.822, G=0.821 read as routing noise; near-zero bias gap.",
         lines=[219], expect=["Gemma 4 26B's routing metrics", "noise", "0.822"],
         metric="H, G",
         fa="Border claim (v0 dataset + confound); B reads as structural exclusion.",
         fb="Consistent: the metrics are labelled noise, so this is not 'bias' evidence."),

    # ---------------- EXPERIMENT 1 ----------------
    dict(id="D1", kind="experiment", section="Exp1 opener",
         claim="States the Sparsity-to-Concentration Hypothesis being tested.",
         lines=[224], expect=["Sparsity-to-Concentration Hypothesis"],
         fa="Restates A; a hypothesis-test parent.",
         fb="Restates the very hypothesis A has already lost."),
    dict(id="D2", kind="result", section="Fig1 caption",
         claim="Entropy across sparsity; 'does not correlate monotonically'.",
         lines=[229], expect=["does not correlate monotonically"],
         metric="H",
         fa="Caption wording presupposes attribution though H is measured.",
         fb="Same caption supports B's reading if H is accepted as noise-correlated."),
    dict(id="D3", kind="method", section="noise floor",
         claim="H_noise ~ 0.982 via shuffle; definition of perfect randomness.",
         lines=[233], expect=["noise floor", "0.982"],
         metric="H_noise",
         fa="Baseline healthy.",
         fb="Baseline healthy."),
    dict(id="D4", kind="result", section="Exp1 gate-level result paragraphs",
         claim="No monotone sparsity-concentration link (GPT-OSS G=0.724 vs OLMoE G=0.646 despite 2x sparsity); GPT-OSS MXFP4 quantization flagged as confound.",
         lines=[245], expect=["no monotonic", "0.724", "0.646", "4-bit block-scaled", "MXFP4"],
         metric=["H", "G"],
         fa="Against A: the ladder result is described in routing-correlation terms, and the one near-0.88 outlier carries an architectural confound.",
         fb="Exactly B's result text."),

    # ---------------- GEMMA + CAVEAT ----------------
    dict(id="E1", kind="experiment", section="Gemma section",
         claim="Gemma 26B: bias gap ~0; metrics reflect raw routing noise; alignment can collapse bias to the noise floor.",
         lines=[248, 250], expect=["Gemma 4 26B", "0.0", "routing noise rather than actual bias", "noise floor entirely"],
         metric="bias gap, H",
         fa="A-defeating phrasing: 'under our audit its H reflect noise, not attribution'.",
         fb="Frames B: no bias instance; metrics are uninterpreted noise."),
    dict(id="E2", kind="flagged", section="Gate-metric caveat",
         claim="THE flagged statement: metric measures routing structure, not causal bias magnitude.",
         lines=[252], expect=["routing structure", "not causal bias magnitude", "caveat"],
         metric="H",
         flagged=True,
         fa="De-couples whatever A meant the metric to be; concentration becomes unmeasured.",
         fb="The sentence to quote in every downstream claim that cites H."),
    dict(id="E3", kind="result", section="Exp2 fig2 caption",
         claim="Fig2 caption: 'Absolute bias-Shapley' concentration bars.",
         lines=[257], expect=["Absolute bias-Shapley"],
         metric="G, H",
         fa="Under A the tag 'bias' borrows causal load the caveat detaches.",
         fb="Caption keeps the label while the E2 caveat flags it."),

    # ---------------- RESULTS 2-8 ----------------
    dict(id="R1", kind="result", section="DBRX architectural replication",
         claim="DBRX H=0.897 vs Mixtral H=0.917 at N_A/N=0.25; robustly diffuse across architectures.",
         lines=[261], expect=["DBRX", "0.897", "robustly diffuse"],
         metric="H, G",
         fa="Supports only 'metrics similar', not a bias claim.",
         fb="Good B row: diffusion persists across architectures."),
    dict(id="R2", kind="method", section="Exp2 dense baseline controls",
         claim="Dense controls H ~ 0.66-0.76 vs MoE 0.88-0.92: metric discriminates topologies.",
         lines=[273], expect=["0.66", "0.76", "0.88"],
         metric="H",
         fa="Under A the sanity check is about metric resolution, not bias claim.",
         fb="Clarifies the metric has finite resolution; B fine."),
    dict(id="R3", kind="result", section="early-layer synergy",
         claim="Universal early-layer synergy cluster 70.2-74.4%; experts act as committees.",
         lines=[307], expect=["70.2", "not a property of individual experts"],
         metric="synergy%",
         fa="Directly anti-'single expert' reading; A must synthesize.",
         fb="Closely word-based B row."),
    dict(id="R4", kind="result", section="routing-contrast underestimation",
         claim="'Routing-contrast underestimates the true diffuseness of bias.'",
         lines=[311], expect=["underestimates the true diffuseness"],
         fa="Anti-localization sentence.",
         fb="Core B row."),
    dict(id="R5", kind="result", section="Exp4 interpretation bullets",
         claim="Flat-early/steep-late = damage curve of general capability; no bias map; not surgically separable.",
         lines=[336, 337, 339], expect=["Flat-Early", "surgically", "not a bias map"],
         metric="H_LOO",
         fa="These bullets define a localized-bias-free reading of the ablation.",
         fb="Stops the map-making in its tracks; A's hub story has no mechanism left."),
    dict(id="R6", kind="result", section="Exp5 JS divergence",
         claim="D_JS 0.42 vs 0.05 null; 'substantially divergent routing profiles'.",
         lines=[344, 349], expect=["Jensen-Shannon", "0.42", "null floor"],
         metric="D_JS",
         fa="Group-level diffuse-within hinders the designated-bias-hub schema.",
         fb="Default B outcome on group-level evidence."),
    dict(id="R7", kind="result", section="Exp5 takeaway",
         claim="No single 'global' bias pathway; subgroups differ in routing profiles.",
         lines=[351], expect=["not driven by a singular", "subgroup-specific sub-networks"],
         fa="As the fig text gives capability-less nuance, an A-er may re-claim partial concentration per subgroup.",
         fb="B consistent.",
         conflict="None."),
    dict(id="R8", kind="result", section="Exp6/7 robustness: rho ~ 0",
         claim="rho first -0.085 / last 0.058; routing-contrast is correlational, not causal map.",
         lines=[366], expect=["-0.085", "0.058", "not act as a direct"],
         metric="rho",
         fa="The 'correlational framing' is explicitly labelled — the causal map is absent.",
         fb="Exact B claim-statements, verbatim."),
    dict(id="R9", kind="result", section="Exp8 layer-LOO stability",
         claim="H_LOO 0.8830 vs H_routing 0.8779; diffuseness stable across method.",
         lines=[370], expect=["0.8830", "0.8779"],
         metric="H, LOO",
         fa="Stable-metric result irons flat only the method confound, not the causal claim.",
         fb="Stable-metric empirics B appears too."),
    dict(id="R10", kind="result", section="Exp8 persistence",
         claim="Diffuseness 'attenuates but persists' under matched LOO mechanism.",
         lines=[373], expect=["attenuates but persists", "genuine architectural property"],
         fa="The residual gap is reported without significance test; B accepts.",
         fb="B's core sentence."),

    # ---------------- CONCLUSION ----------------
    dict(id="T1", kind="conclusion", section="Conclusion finale",
         claim="No expert-level bias-localization signal; the ablation curve is damage curve, not a bias map.",
         lines=[391], expect=["no expert-level", "not a bias map"],
         fa="Final sentence = A-negation sentence.",
         fb="The conclusion, as written, IS the B conclusion."),
]

ROWS = [r for r in ROWS if r.get("kind") and r.get("kind") != "header"]

data = {
    "readme": {
        "title": "Bias-attribution readings (Framing A vs Framing B)",
        "framing_a": ("Framing A -- 'sparsity concentrates bias'; outlier experts carry it; "
                      "the paper's metrics are readings of real bias."),
        "framing_b": ("Framing B -- 'no separable bias substrate'; the metric is a routing-contrast, "
                      "correlational quantity; the attachment belongs to the capability backbone."),
        "method": ("Verbatim quotes pulled by verified line anchors from the SAME file; "
                   "each row asserts token ground-truth. Line numbers were verified live "
                   "against the file before generation."),
        "governance_rules": [
            "Any claim whose verbatim text already contains 'no', 'rejected', 'not', or "
            "'caveat' is governed by that clause and cannot be quoted out of it.",
            "The E2 line (252) is the *flagged caveat*: metric measures routing structure, "
            "not causal bias magnitude. All rows citing concrete number statements inherit "
            "its scope.",
        ],
        "methods_line_count": len(src),
    },
    "rows": ROWS,
}


def main():
    for r in ROWS:
        r["quote"] = extract(r)
        r.pop("expect", None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    print(f"wrote {len(ROWS)} rows -> {OUT}")


if __name__ == "__main__":
    main()