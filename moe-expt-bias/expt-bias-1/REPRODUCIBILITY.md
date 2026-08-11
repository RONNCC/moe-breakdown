# Reproducibility Guide — MoE Routing-Bias Shapley Study (expt-bias-1)

End-to-end instructions to go from a bare clone of this repo to the paper figures
and appendices in `moe-expt-bias-2/moe_bias_report_acm_v2.tex`. Every command is
copy-pasteable from the repository root
(`/Users/ronnie.ghose/src/priv/gatech/research-projs/moe-breakdown`) unless noted.

---

## 1. Environment

Project convention is `uv`-managed Python 3.11 (see `AGENTS.md`; do **not** use
bare `pip install`). The authoritative environment recipe is
`moe-expt-bias/expt-bias-1/scripts/bootstrap_uv_env.sh` — read it first and mirror
it. In short it creates a venv with `uv venv --python python3.11`, installs the
package editable (`uv pip install -e .`, which pulls `pyyaml`, `numpy`, `scipy`
from `pyproject.toml`), then pins the heavy compute stack:

```bash
# 1. Create the venv at the study root
cd moe-expt-bias/expt-bias-1
uv venv .venv --python python3.11
source .venv/bin/activate

# 2. Editable install of the study package (pyyaml, numpy, scipy from pyproject)
uv pip install --upgrade pip setuptools wheel
uv pip install -e .

# 3. torch pinned to 2.6.0 (torch.accelerator required by the transformers
#    MXFP4 quantizer for openai/gpt-oss-120b); cu124 index for PACE host-driver compat
uv pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124

# 4. Transformers + data/decoding deps (tiktoken needed by dbrx trust_remote_code)
uv pip install 'transformers>=4.44' accelerate datasets sentencepiece protobuf tiktoken

# 5. Optional but recommended for larger ladder models (Phi-3.5-MoE,
#    Mixtral-8x7B, Llama-3.1-8B) on single-GPU nodes
uv pip install bitsandbytes

# 6. triton+kernels pins required by the gpt-oss MXFP4 path; installed last so
#    nothing downgrades them
uv pip install 'triton>=3.4.0' 'kernels>=0.15.2,<0.16.0'

# 7. Analysis-stack extras needed by stats_analysis/scripts (not in the base
#    bootstrap script; pandas/seaborn are imported by paper_figures_seaborn.py)
uv pip install pandas seaborn
```

> Import coverage (verified against `scripts/run_bias_study.py` +
> `stats_analysis/scripts/`): torch, transformers, datasets (run); scipy,
> numpy, pandas, seaborn (analysis). The pinned set above covers all of them.
> A ready venv already exists in-tree at
> `moe-expt-bias/expt-bias-1/.venv/bin/python`.

## 2. Data

### (a) What lives in git: `results/`

Each study run is a directory under `moe-expt-bias/expt-bias-1/results/` (e.g.
`exp1-concentration-olmoe-1b-7b-v1/`) containing:

- `result.json` — aggregated concentration metrics (entropy `H`, gini `G`,
  top-5/top-10% fractions, `n_pairs`, `n_nonzero`) plus run metadata;
- `pair_meta.json` — per-pair `{index, benchmark, group}` rows used for
  stratified (block) bootstrap resampling;
- `player_ids.json` — expert/player id map (`n_players` = `len(player_ids)`);
- aggregates `phi.npy` / `routing_freq.npy` (small);
- `result_shard0of2.json` / `result_shard1of2.json` for sharded runs;
- Exp5 additionally stores per-cohort vectors `phi_group_*.npy` (85 cohorts).

The **per-pair payloads (`per_pair_phi*.npy`) are gitignored** (repo `.gitignore`:
`results/**/per_pair_phi*.npy`, `results/**/phi*.npy`,
`results/**/routing_freq*.npy`) because they total ~565 MB across 15 captures
and GitHub rejects >100 MB files. They are distributed separately — see (b).

### (b) Per-pair phi payloads: Kaggle

The per-pair Shapley payloads are published as the Kaggle dataset
`sghose0/moe-bias-routing-shapley-perpair-phi`:

```bash
kaggle datasets download -d sghose0/moe-bias-routing-shapley-perpair-phi
```

**Placement.** `stats_analysis/scripts/s04_bootstrap_cis.py` locates payloads
verbatim as follows (verified by reading the script):

- Model-dir glob (`main()`): `RESULTS.glob("exp1-concentration-*")` +
  `RESULTS.glob("exp2-dense-*")` + `RESULTS.glob("exp8-lloo-*")`, with
  `RESULTS = Path(__file__).resolve().parents[2] / "results"` — every
  `exp1-concentration-*` (ladder), `exp2-dense-*` (dense baseline), and
  `exp8-lloo-*` (same-mechanism dense-LOO comparison) directory is a
  candidate model.
- Inside each candidate dir it tries, in order: `per_pair_phi.npy`,
  `per_pair_phi-v1.npy`, `per_pair_phi_v1.npy`, then any `per_pair_phi*.npy`
  (glob fallback); for non-`-v1` exp1 dirs it additionally falls back to the
  sibling `-v1` dir's `per_pair_phi.npy` / `per_pair_phi-v1.npy`.
- Model key (`_model_key()`): `exp1-concentration-<name>` -> `<name>`;
  `exp2-<name>-dense-baseline|dense-crosscheck` -> `<name>`; `exp8-lloo-<name>`
  -> **`lloo-<name>`** — namespaced with a `lloo-` prefix so exp8's
  same-mechanism dense-LOO estimator never collides with (silently
  overwrites) the exp1 routing-contrast ladder entry for the same base
  model name in the output `models` dict. A real bug hit this session: before
  the prefix was added, exp8's `phi3.5-moe`/`olmoe-1b-7b` keys clobbered the
  exp1 ladder rows for those models, corrupting their CIs until caught by
  manual inspection.

So unzip the Kaggle download and place each model's payload at
`moe-expt-bias/expt-bias-1/results/exp1-concentration-<model>-v1/per_pair_phi.npy`
(e.g. `.../exp1-concentration-olmoe-1b-7b-v1/per_pair_phi.npy`), alongside the
`pair_meta.json` that ships in git. Dirs without a payload are reported as
`"status": "MISSING"` in `outputs/s04_bootstrap_cis.json` — the script never
crashes. `pair_meta.json` presence enables stratified (within-group) resampling;
without it s04 uses plain iid resampling over pairs.

(Exp5's `s05_exp5_bootstrap_js.py` likewise reads `per_pair_phi.npy` +
`pair_meta.json` from `results/exp5-demographic-specificity-olmoe-1b-7b-v1/`.)

## 3. Local analysis pipeline

All scripts live in `moe-expt-bias/expt-bias-1/stats_analysis/scripts/`, resolve
`RESULTS = <study-root>/results` from their own location (`parents[2]`), take no
CLI arguments, and write into `stats_analysis/outputs/` (and `figures/` where
noted). Run them in this order from the study root with the venv python:

```bash
cd moe-expt-bias/expt-bias-1
PY=.venv/bin/python

$PY stats_analysis/scripts/s01_exp1_stability.py      # -> outputs/s01_exp1_stability.json
$PY stats_analysis/scripts/s02_exp5_js.py             # -> outputs/s02_exp5_js.json + figures/s02_js_distribution.png
$PY stats_analysis/scripts/s03_h1_verdict.py          # -> outputs/s03_h1_verdict.json
$PY stats_analysis/scripts/s04_bootstrap_cis.py       # -> outputs/s04_bootstrap_cis.json
$PY stats_analysis/scripts/paper_figures_seaborn.py   # -> stats_analysis/figures/fig{1..5}_*.png
```

What each does (verified against `main()`):

- **s01** — Exp1 point-estimate stability: v0 (400 pairs) vs v1 (5000) vs
  shard halves from `result_shard*of2.json`; flags broken/zero-phi runs
  (`entropy==0` or `n_nonzero==0`).
- **s02** — Exp5 JS divergence over the 85 stored `phi_group_*.npy` cohort
  vectors: pairwise JS matrix, block-bootstrap CI, expert-identity permutation
  null, cohort-vs-pool. (`s05_exp5_bootstrap_js.py` is the per-pair bootstrap
  variant, optional and only when the Kaggle payload is present.)
- **s03** — H1 monotonicity audit under model-set exclusions (full-6,
  minus-GPT-OSS, paper-valid-4, unique-sparsity-3) with drift-aware verdicts;
  ladder rows read `results/exp1-concentration-*/result.json`, preferring `-v1`
  unless the run is flagged broken.
- **s04** — per-model block-bootstrap 95% percentile CIs for entropy H, gini G,
  top-5/top-10% fractions (n_boot=5000, seed=42, resample pairs with
  replacement, stratified by `pair_meta.json` groups when present); covers
  `exp1-concentration-*` (ladder), `exp2-dense-*` (dense baselines), and
  `exp8-lloo-*` (same-mechanism dense-LOO comparison) — the latter keyed
  `lloo-<model>` so it never collides with an exp1 ladder entry for the same
  model (see glob-pattern note in Sec 2b); cross-checks point estimates
  against `result.json` (`MISMATCH` reported loudly); emits `MISSING` for
  dirs without payloads.
- **paper_figures_seaborn.py** — regenerates all paper figures from the
  hardcoded `MOE` ladder tuple at the top (`(dir, label, N_players, N_active)`,
  currently all `-v1` dirs incl. `exp1-concentration-gpt-oss-120b-v1`), the
  dense baselines, and `stats_analysis/outputs/*.json`. The draft's in-text
  numbers likewise derive from `s03_h1_verdict.json` / `s04_bootstrap_cis.json`
  — rerun s01→s04 before quoting them.

Expected outputs (all present in a fully-synced checkout):

| script | outputs |
|---|---|
| s01 | `outputs/s01_exp1_stability.json` |
| s02 | `outputs/s02_exp5_js.json`, `figures/s02_js_distribution.png` |
| s03 | `outputs/s03_h1_verdict.json` |
| s04 | `outputs/s04_bootstrap_cis.json` |
| s05 (optional) | `outputs/s05_exp5_bootstrap_js.json` |
| paper_figures_seaborn | `figures/fig1_entropy_ladder.png`, `fig2_gini_ladder.png`, `fig3_h1_scatter.png`, `fig4_top_fraction.png`, `fig5_dense_vs_moe.png` |


**Independent appendix-table verification.** `scripts/audit_appendix.py`
(run from `moe-expt-bias/expt-bias-1`, no CLI args) parses every numeric row
of Appendix A's full-measurement table directly out of
`moe-expt-bias-2/moe_bias_report_acm_v2.tex` and cross-checks each value
(H, Gini, top-5/top-10% fractions, CIs, `|dH|`/`|dG|` drift) against its
cited JSON source (`s01_exp1_stability.json`, `s03_h1_verdict.json`,
`s04_bootstrap_cis.json`, per-model `result.json`), failing loudly on any
mismatch:

```bash
cd moe-expt-bias/expt-bias-1
.venv/bin/python scripts/audit_appendix.py
# -> "Parsed N data rows from appendix table"
# -> "[PASS] ALL APPENDIX ROWS MATCH AUTHORITATIVE SOURCES EXACTLY" or a
#    per-row [FAIL] diff naming the exact source field and discrepancy
```

Run this after any edit that touches Appendix A's numbers, and after
re-running s01-s04, to catch transcription drift before it ships.

## 4. Cluster (PACE ICE, Slurm)

Access + queue:

```bash
ssh login-ice.pace.gatech.edu
squeue -u sghose7          # check the queue
squeue -j 5575070          # inspect the currently-running job
```

Submit a study from a login node (the repo must be checked out under
`~/scratch/moe-breakdown`, per the configs' `slurm.workdir`):

```bash
cd moe-expt-bias/expt-bias-1
python3 scripts/submit_slurm_study.py --config configs/<study>.yaml
# sharded + per-pair optional flags:
python3 scripts/submit_slurm_study.py --config configs/<study>.yaml --num-shards 2 --save-per-pair-phi
```

The script builds the sbatch command from each config's `slurm:` block
(`scripts/submit_slurm_study.py`, verified): `--partition ice-gpu`,
`--qos` from config, `--gres=gpu:<type>:<n>` (or `--gpus-per-node=`),
`--export=ALL,STUDY_CONFIG=...,OUT_DIR=...,WORKDIR=...,MODULES=...` (+`SHARD_IDX`/
`NUM_SHARDS` when sharded, `SAVE_PER_PAIR_PHI=1` with `--save-per-pair-phi`),
and the sbatch script `slurm/run_bias_study.sbatch`.

Resource facts to respect (see `CLUSTER-STATUS.md`):

- **QOS `coc-ice` caps GPU-minutes/job at 960** (`MaxTRESMinsPerJob = gres/gpu=960`,
  i.e. 2 GPUs x 8h or 4 x 4h). Longer requests are rejected at submit time.
- Partition is `ice-gpu` (H100/H200/L40S/A100 pools; GPT-OSS-120B bf16 needs
  ~240 GB → 4xH100; large MoE bf16 fits 2xH200).

Pull artifacts back to the laptop with `scp` (remote root
`/storage/ice1/0/2/sghose7/moe-breakdown-bias-runs/expt-bias-1/` or
`/home/hice1/sghose7/scratch/moe-breakdown-bias-runs/expt-bias-1/` per config
`output_root`):

```bash
scp -r login-ice.pace.gatech.edu:/home/hice1/sghose7/scratch/moe-breakdown-bias-runs/expt-bias-1/<study-dir> \
  moe-expt-bias/expt-bias-1/results/
```

### Job status (as of 2026-08-11; PACE ICE in scheduled quarterly
maintenance, 2026-08-11 06:00 -- 2026-08-13 23:59)

- **sbatch 5575070 — COMPLETE**: GPT-OSS-120B exp1 re-run at 5000 pairs
  landed and is integrated (`s04` confirms $H=0.8789$ vs. the certified
  2000-pair $H=0.8765$, $\Delta H=+0.0024$, within CI; Table 1 + Appendix A).
- **5575538 — CANCELLED**: Exp3 collectivity, Mixtral-8x7B refresh
  (mis-submitted: requested `gres/gpu=4` x `06:00:00` = 1440 GPU-min,
  exceeding the `coc-ice` 960 GPU-min/job cap, so it could never run; not
  blocking, the July capture is already verified non-degenerate and used in
  the paper).
- **5575743 -> 5575799** — Exp3 collectivity, DBRX-132B: was **RUNNING**
  (node-pinned, `--time=01:10:00`, after a `slurm.time` fix — the original
  config's `03:00:00` was too short for Exp3's ~9.6 min/pair cost and timed
  out) when the cluster entered the maintenance window; final state unknown
  until the cluster reopens.
- **5575744 -> 5575791** — Exp3 collectivity, GPT-OSS-120B: was **RUNNING**
  (node-pinned, `--time=01:10:00`, same `slurm.time` fix, `08:00:00` request,
  10 pairs) when the cluster entered the maintenance window; final state
  unknown until the cluster reopens.
- **5575745 -> 5575800** — Exp6 ladder extension, GPT-OSS-120B: was
  **RUNNING** (node-pinned, `--time=01:00:00`, same fix) when the cluster
  entered the maintenance window; final state unknown until the cluster
  reopens; this is the only remaining gap in the causal-ablation evidence
  chain.
- **5575752 -> 5575798 — COMPLETED**: Exp8 same-mechanism LOO, OLMoE-1B-7B
  per-pair capture (node-pinned, `--time=01:10:00`; the prior per-pair job
  5575255 only wrote a summary `result.json`, no `per_pair_phi.npy`):
  $H=0.7361$, Gini$=0.6239$, $n_{pairs}=100$; per-pair CI integrated into
  Appendix B.

The `ReqNodeNotAvail, Reserved for maintenance` reason seen earlier this
session was **not** a GPU-minute backlog or scheduler bug: it is a real,
non-admin-visible PACE quarterly-maintenance reservation, since directly
confirmed by the login banner on `login-ice.pace.gatech.edu` (maintenance
2026-08-11 06:00 -- 2026-08-13 23:59; login itself is refused cluster-wide,
not just job scheduling). First action once the cluster reopens: `sacct -j
5575799,5575791,5575800 --format=JobID,State,ExitCode,Elapsed -X -n`. See
`CLUSTER-STATUS.md` for the full root-cause writeup and queue history.

## 5. Paper compilation

Draft: `moe-expt-bias-2/moe_bias_report_acm_v2.tex` (supersedes
`stats_analysis/paper/moe_bias_report_aug8/`). Self-contained:
`thebibliography` is embedded (no bibtex run needed) and `acmart.cls` ships
next to the .tex. Figures are pulled via
`\graphicspath{{../moe-expt-bias/expt-bias-1/stats_analysis/figures/}}` (set at
the top of the file) — figures must exist there before compiling.

```bash
cd moe-expt-bias-2
pdflatex moe_bias_report_acm_v2.tex   # pass 1
pdflatex moe_bias_report_acm_v2.tex   # pass 2 (resolves cross-refs)
open moe_bias_report_acm_v2.pdf
```

## 6. Directory map

```
repo_root/  (moe-breakdown)
├── moe-expt-bias-2/
│   └── moe_bias_report_acm_v2.tex          # current ACM draft (pdflatex x2, embedded thebibliography)
└── moe-expt-bias/
    ├── expt-bias-1/                        # study root (everything below)
    │   ├── results/                        # per-study dirs: result.json + pair_meta.json + player_ids.json
    │   │                                   #   + phi.npy/routing_freq.npy (+phi_group_*.npy for Exp5);
    │   │                                   #   per_pair_phi*.npy gitignored -> Kaggle
    │   ├── stats_analysis/
    │   │   ├── scripts/                    # s01_exp1_stability.py, s02_exp5_js.py, s03_h1_verdict.py,
    │   │   │                               #   s04_bootstrap_cis.py, s05_exp5_bootstrap_js.py,
    │   │   │                               #   paper_figures_seaborn.py, run_bootstrap_all.sh
    │   │   ├── outputs/                    # s01..s05 *.json (paper-number sources)
    │   │   ├── figures/                    # fig{1..5}_*.png + s02_js_distribution.png (paper figures,
    │   │   │                               #   referenced by \graphicspath)
    │   │   └── paper/                      # aug8 legacy draft + refs.bib (superseded by moe-expt-bias-2)
    │   ├── configs/                        # study.*.yaml (e.g. study.gpt-oss-120b.concentration.v2.yaml)
    │   ├── scripts/                        # run_bias_study.py, submit_slurm_study.py, bootstrap_uv_env.sh,
    │   │                                   #   run_experiment6_ablation.py, merge_study_shards.py, ...
    │   ├── src/moe_bias_shapley/           # package (config, modeling, benchmars, shapley, reporting)
    │   ├── CLUSTER-STATUS.md               # GPU inventory, queue, run book, Kaggle release notes
    │   ├── slurm/                          # run_bias_study.sbatch, run_experiment_script.sbatch
    │   └── uv.lock / pyproject.toml
    ├── figures_flint/, latex_build/, figures/, memory.md, study_design_C1_C4.md, ...
    └── ...
└── moe-routing/                           # context only: earlier routing-latency experiments (separate workstream)
```
