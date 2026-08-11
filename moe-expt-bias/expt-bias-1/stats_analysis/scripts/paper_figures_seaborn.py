"""Generate all figures for the ACM revision draft.

Data sources: expt-bias-1/results/*/result.json (per-model concentration
metrics from the routing-Shapley study), stats_analysis/outputs/*.json
(stability / JS / verdict artifacts).

Run:
    uv venv --system-site-packages /tmp/figuv
    uv pip install --python /tmp/figuv/bin/python seaborn pandas
    /tmp/figuv/bin/python stats_analysis/scripts/figures_seaborn.py

Output: stats_analysis/figures/fig_*.png (300 dpi, acmart-friendly).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "stats_analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.05)

# ---------------------------------------------------------------------------
# Load ladder data (all -v1 certified captures for the six MoE models, so the
# figures match Table 1 / s03 ladder; gpt-oss-120b-v1 flipped in Aug-09)
# ---------------------------------------------------------------------------
MOE = [
    # (dir, label, N_players, N_active) -- verified against s03_h1_verdict.json ladder row.
    ("exp1-concentration-olmoe-1b-7b-v1",     "OLMoE-1B-7B",             1024, 16),
    ("exp1-concentration-phi3.5-moe-v1",      "Phi-3.5-MoE",             512,  64),
    ("exp1-concentration-gpt-oss-120b-v1",    "GPT-OSS-120B",            4608, 144),
    ("exp1-concentration-mixtral-8x7b-v1",    "Mixtral-8x7B",            256,  64),
    ("exp1-concentration-dbrx-v1",            "DBRX-132B",               640,  160),
    ("exp1-concentration-gemma4-26b-v1",      "Gemma-4-26B",             3840, 240),
]
DENSE = [
    # v1 per-pair captures (1800-4000 pairs) -- matches tab:dense / Appendix A.
    ("exp2-dense-baseline-olmo-7b-v1",        "OLMo-7B",                 32),
    ("exp2-dense-baseline-phi3.5-mini-v1",    "Phi-3.5-mini",            32),
    ("exp2-dense-crosscheck-llama-2-7b-v1",   "Llama-2-7B",              32),
    ("exp2-dense-crosscheck-llama-3.1-8b-v1", "Llama-3.1-8B",            32),
]


def load(d):
    return json.loads((RESULTS / d / "result.json").read_text())


def load_ladder():
    rows = []
    for d, name, n, k in MOE:
        cm = load(d)["concentration_metrics"]
        rows.append(dict(family="MoE", model=name, k=k, N=n, k_over_N=k / n,
                        entropy=cm["entropy"], gini=cm["gini"],
                        top5=cm["top_5_fraction"], top10=cm["top_10pct_fraction"],
                        n_players=cm["n_players"], n_nonzero=cm["n_nonzero"]))
    for d, name, n in DENSE:
        cm = load(d)["concentration_metrics"]
        rows.append(dict(family="Dense", model=name, k=None, N=n, k_over_N=1.0,
                        entropy=cm["entropy"], gini=cm["gini"],
                        top5=cm["top_5_fraction"], top10=cm["top_10pct_fraction"],
                        n_players=cm["n_players"], n_nonzero=cm["n_nonzero"]))
    return pd.DataFrame(rows)


def load_metrics(d):
    return json.loads((RESULTS / d / "result.json").read_text())["concentration_metrics"]


# ---------------------------------------------------------------------------
# Fig 1: Sparsity ladder entropy
# ---------------------------------------------------------------------------
def fig_entropy_ladder(df):
    moe = df[df.family == "MoE"].sort_values("k_over_N")
    dense = df[df.family == "Dense"]
    plt.figure(figsize=(7.2, 4.4))
    ax = plt.gca()
    order = list(moe.model)
    sns.pointplot(x="model", y="entropy", data=moe, order=order,
                  markers="o", errorbar=None, color="C0", label="MoE")
    ax.hlines(dense.entropy.mean() - dense.entropy.std(), -1, len(order),
              colors="#9aa0a6", linestyles="--", linewidth=1,
              label=f"dense mean $\pm$ sd ({dense.entropy.mean():.3f} ± {dense.entropy.std():.3f})")
    for x, (_, m) in enumerate(moe.iterrows()):
        ax.annotate(f"${m.N}$", (x, m.entropy), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Shapley bias-load entropy across sparsity ladder")
    ax.set_ylabel("Entropy $H$ over expert attribution")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_entropy_ladder.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Fig 2: Gini
# ---------------------------------------------------------------------------
def fig_gini_ladder(df):
    moe = df[df.family == "MoE"]
    plt.figure(figsize=(7.2, 4.4))
    sns.pointplot(x="model", y="gini", data=moe, color="C1", errorbar=None)
    plt.xticks(rotation=25)
    plt.title("Gini over expert attribution")
    plt.ylabel("Gini")
    plt.tight_layout(); plt.savefig(OUT / "fig2_gini_ladder.png", dpi=300); plt.close()


# ---------------------------------------------------------------------------
# Fig 3: concentration vs sparsity (H1 reframed)
# ---------------------------------------------------------------------------
def fig_h1_scatter(df):
    moe = df[df.family == "MoE"]
    plt.figure(figsize=(6.8, 4.6))
    ax = plt.gca()
    sns.scatterplot(data=moe, x="k_over_N", y="entropy", s=120, color="C0", ax=ax)
    ax.axhline(df[df.family == "Dense"].entropy.mean(),
               color="gray", ls="--", lw=1, label="dense mean")
    ax.set_xscale("log")
    ax.set_xticks(sorted(moe.k_over_N.unique()))
    ax.set_xticklabels([f"{f:.3f}" for f in sorted(moe.k_over_N.unique())])
    for _, row in moe.iterrows():
        ax.annotate(row.model, (row.k_over_N, row.entropy), xytext=(4, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("active/ total experts $k/N$ (log)")
    ax.set_ylabel("entropy $H$")
    ax.set_title("(H1) No monotonic concentration vs sparsity")
    ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "fig3_h1_scatter.png", dpi=300); plt.close()


# ---------------------------------------------------------------------------
# Fig 4: top-5 vs top-10 concentration
# ---------------------------------------------------------------------------
def fig_top_fraction(df):
    plt.figure(figsize=(7.2, 4.2))
    idx = np.arange(len(df)); w = 0.38
    plt.bar(idx - w / 2, df.top5, w, label="top-5 fraction", color="C3")
    plt.bar(idx + w / 2, df.top10, w, label="top-10% fraction", color="C4")
    plt.xticks(idx, df.model, rotation=25)
    plt.ylabel("fraction of attribution")
    plt.legend(); plt.title("Top-expert attribution share")
    plt.tight_layout(); plt.savefig(OUT / "fig4_top_fraction.png", dpi=300); plt.close()


# ---------------------------------------------------------------------------
# Fig 5: dense-vs-MoE comparison
# ---------------------------------------------------------------------------
def fig_dense_vs_moe(df):
    plt.figure(figsize=(6.2, 4.2))
    sns.boxplot(data=df, x="family", y="gini", palette=["#cccccc", "#4C72B0"])
    np.random.seed(42)  # seaborn<0.14 strip jitter draws from np.random: seed for byte-reproducible figures
    sns.stripplot(data=df, x="family", y="gini", color=".3", size=6)
    plt.ylim(0.35, 0.95)
    plt.ylabel("Gini")
    plt.title("Dense vs MoE: attribution inequality measure")
    plt.tight_layout(); plt.savefig(OUT / "fig5_dense_vs_moe.png", dpi=300); plt.close()


def main():
    df = load_ladder()
    print(df[["model", "k_over_N", "entropy", "gini"]].to_string(index=False))
    fig_entropy_ladder(df)
    fig_h1_scatter(df)
    fig_top_fraction(df)
    fig_gini_ladder(df)
    fig_dense_vs_moe(df)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()