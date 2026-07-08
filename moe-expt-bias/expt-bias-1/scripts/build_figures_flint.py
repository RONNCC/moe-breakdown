#!/usr/bin/env python3
"""
Microsoft Flint Chart Figure Generation Script for MoE Bias Attribution Study.
Generates all 6 paper figures by compiling Flint specifications into static PNGs
via the flint-chart-mcp server stdio RPC protocol.
"""

import os
import subprocess
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure target directories exist
os.makedirs("moe-expt-bias/figures_flint", exist_ok=True)

def render_flint_chart(data, semantic_types, chart_spec, output_path, backend="vegalite", background="#ffffff", scale=2.0):
    """
    Subprocess stdio JSON-RPC wrapper for npx flint-chart-mcp.
    Compiles and renders a Flint specification directly to a local PNG.
    """
    print(f"Compiling and rendering {os.path.basename(output_path)} via Flint MCP...")
    
    # Standard MCP JSON-RPC protocol requests
    reqs = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "render_chart",
                "arguments": {
                    "data": data,
                    "semantic_types": semantic_types,
                    "chart_spec": chart_spec,
                    "backend": backend,
                    "format": "png",
                    "scale": scale,
                    "background": background
                }
            }
        }
    ]

    # Spawn MCP server stdio process
    proc = subprocess.Popen(
        ["npx", "flint-chart-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send pipeline write
    stdin_data = "\n".join(json.dumps(r) for r in reqs) + "\n"
    proc.stdin.write(stdin_data)
    proc.stdin.close()

    # Retrieve outputs
    stdout_data = proc.stdout.read()
    proc.wait()

    lines = [line for line in stdout_data.split("\n") if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Flint MCP failed to respond. Raw stdout:\n{stdout_data}")

    parsed = json.loads(lines[1])
    if "error" in parsed:
        raise ValueError(f"Flint MCP Execution Error: {parsed['error']}")

    content_list = parsed["result"]["content"]
    img_base64 = None
    for item in content_list:
        if item.get("type") == "image" and item.get("mimeType") == "image/png":
            img_base64 = item.get("data")
            break

    if not img_base64:
        raise ValueError(f"No PNG output returned by Flint MCP. Content received:\n{content_list}")

    # Decode and save PNG
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(img_base64))
    print(f"Saved: {output_path}")


# ==============================================================================
# DATASET INITIALIZATION
# ==============================================================================
raw_data = [
    {"model": "OLMoE-1B-7B", "type": "MoE", "sparsity": 0.016, "n_players": 1024, "H": 0.8786, "Gini": 0.6457, "top5": 0.0928, "top10": 0.5088, "gap": 0.2444},
    {"model": "GPT-OSS-120B", "type": "MoE", "sparsity": 0.031, "n_players": 4608, "H": 0.880, "Gini": 0.724, "top5": 0.020, "top10": 0.549, "gap": 0.165},
    {"model": "Gemma 4 26B", "type": "Gemma 4 (null bias)", "sparsity": 0.063, "n_players": 3840, "H": 0.822, "Gini": 0.821, "top5": 0.038, "top10": 0.657, "gap": 0.0},
    {"model": "Phi-3.5-MoE", "type": "MoE", "sparsity": 0.125, "n_players": 512, "H": 0.8779, "Gini": 0.6435, "top5": 0.0974, "top10": 0.4704, "gap": 0.1614},
    {"model": "Mixtral-8x7B", "type": "MoE", "sparsity": 0.250, "n_players": 256, "H": 0.9167, "Gini": 0.5189, "top5": 0.1074, "top10": 0.3559, "gap": 0.2000},
    {"model": "DBRX-instruct", "type": "MoE", "sparsity": 0.250, "n_players": 640, "H": 0.8970, "Gini": 0.5995, "top5": 0.0896, "top10": 0.4329, "gap": 0.1686},
    {"model": "OLMo-7B", "type": "Dense", "sparsity": 1.0, "n_players": 32, "H": 0.7041, "Gini": 0.7160, "top5": 0.7477, "top10": 0.5607, "gap": 0.1563},
    {"model": "Phi-3.5-mini", "type": "Dense", "sparsity": 1.0, "n_players": 32, "H": 0.7583, "Gini": 0.6466, "top5": 0.6349, "top10": 0.5018, "gap": 0.2150},
    {"model": "Llama-3.1-8B", "type": "Dense", "sparsity": 1.0, "n_players": 32, "H": 0.6563, "Gini": 0.7099, "top5": 0.7232, "top10": 0.6722, "gap": 0.1877},
    {"model": "Llama-2-7B", "type": "Dense", "sparsity": 1.0, "n_players": 32, "H": 0.6950, "Gini": 0.6574, "top5": 0.6972, "top10": 0.6495, "gap": 0.1724}
]
df = pd.DataFrame(raw_data)


# ==============================================================================
# FIGURE 1 — Sparsity Ladder
# ==============================================================================
f1_data = {"values": raw_data}
f1_sem = {
    "model": "Category",
    "sparsity": "Quantity",
    "H": "Quantity",
    "n_players": "Quantity",
    "type": "Category"
}
f1_spec = {
    "chartType": "Scatter Plot",
    "encodings": {
        "x": {"field": "sparsity", "scale": {"type": "log"}},
        "y": {"field": "H"},
        "color": {"field": "type"},
        "size": {"field": "n_players"}
    },
    "baseSize": {"width": 550, "height": 400}
}
render_flint_chart(f1_data, f1_sem, f1_spec, "moe-expt-bias/figures_flint/fig1_sparsity_ladder.png")


# ==============================================================================
# FIGURE 2 — Full Concentration Bars (Grouped Bar Chart)
# ==============================================================================
f2_rows = []
for row in raw_data:
    f2_rows.append({"model": row["model"], "metric": "Shannon Entropy (H)", "value": row["H"], "type": row["type"]})
    f2_rows.append({"model": row["model"], "metric": "Gini Coefficient (G)", "value": row["Gini"], "type": row["type"]})

f2_data = {"values": f2_rows}
f2_sem = {
    "model": "Category",
    "metric": "Category",
    "value": "Quantity"
}
f2_spec = {
    "chartType": "Grouped Bar Chart",
    "encodings": {
        "x": {"field": "model"},
        "y": {"field": "value"},
        "group": {"field": "metric"}
    },
    "baseSize": {"width": 600, "height": 400}
}
render_flint_chart(f2_data, f2_sem, f2_spec, "moe-expt-bias/figures_flint/fig2_concentration_bars.png")


# ==============================================================================
# FIGURE 3 — Metric Discrimination Sanity Control
# ==============================================================================
fig3_rows = [
    {"pair": "OLMoE-1B-7B vs. OLMo-7B", "model_type": "MoE", "H": 0.8786},
    {"pair": "OLMoE-1B-7B vs. OLMo-7B", "model_type": "Dense Control", "H": 0.7041},
    {"pair": "Phi-3.5-MoE vs. Phi-3.5-mini", "model_type": "MoE", "H": 0.8779},
    {"pair": "Phi-3.5-MoE vs. Phi-3.5-mini", "model_type": "Dense Control", "H": 0.7583}
]
f3_data = {"values": fig3_rows}
f3_sem = {
    "pair": "Category",
    "model_type": "Category",
    "H": "Quantity"
}
f3_spec = {
    "chartType": "Grouped Bar Chart",
    "encodings": {
        "x": {"field": "pair"},
        "y": {"field": "H"},
        "group": {"field": "model_type"}
    },
    "baseSize": {"width": 550, "height": 400}
}
render_flint_chart(f3_data, f3_sem, f3_spec, "moe-expt-bias/figures_flint/fig3_localizability_h2.png")


# ==============================================================================
# FIGURE 4 — Synergy Fractions Across Layers
# ==============================================================================
fig4_rows = [
    {"model": "OLMoE-1B-7B", "layer": "First MoE Layer (Early)", "synergy_fraction": 0.702},
    {"model": "OLMoE-1B-7B", "layer": "Last MoE Layer (Late)", "synergy_fraction": 0.278},
    {"model": "Phi-3.5-MoE", "layer": "First MoE Layer (Early)", "synergy_fraction": 0.744},
    {"model": "Phi-3.5-MoE", "layer": "Last MoE Layer (Late)", "synergy_fraction": 0.219},
    {"model": "Mixtral-8x7B", "layer": "First MoE Layer (Early)", "synergy_fraction": 0.716},
    {"model": "Mixtral-8x7B", "layer": "Last MoE Layer (Late)", "synergy_fraction": 0.503}
]
f4_data = {"values": fig4_rows}
f4_sem = {
    "model": "Category",
    "layer": "Category",
    "synergy_fraction": "Quantity"
}
f4_spec = {
    "chartType": "Grouped Bar Chart",
    "encodings": {
        "x": {"field": "model"},
        "y": {"field": "synergy_fraction"},
        "group": {"field": "layer"}
    },
    "baseSize": {"width": 550, "height": 400}
}
render_flint_chart(f4_data, f4_sem, f4_spec, "moe-expt-bias/figures_flint/fig4_synergy_fractions.png")


# ==============================================================================
# FIGURE 5 — Ablation Curve and Capability Loss
# ==============================================================================
fig5_rows = [
    {"fraction_ablated": 0.0, "metric": "Disparity Reduction (%)", "value": 0.0},
    {"fraction_ablated": 0.0, "metric": "Perplexity Increase (%)", "value": 0.0},
    {"fraction_ablated": 0.001, "metric": "Disparity Reduction (%)", "value": -0.2},
    {"fraction_ablated": 0.001, "metric": "Perplexity Increase (%)", "value": 0.02},
    {"fraction_ablated": 0.01, "metric": "Disparity Reduction (%)", "value": 6.2},
    {"fraction_ablated": 0.01, "metric": "Perplexity Increase (%)", "value": 0.8},
    {"fraction_ablated": 0.099, "metric": "Disparity Reduction (%)", "value": 92.2},
    {"fraction_ablated": 0.099, "metric": "Perplexity Increase (%)", "value": 85.0},
    {"fraction_ablated": 0.200, "metric": "Disparity Reduction (%)", "value": 94.3},
    {"fraction_ablated": 0.200, "metric": "Perplexity Increase (%)", "value": 320.0}
]
f5_data = {"values": fig5_rows}
f5_sem = {
    "fraction_ablated": "Quantity",
    "metric": "Category",
    "value": "Quantity"
}
f5_spec = {
    "chartType": "Line Chart",
    "encodings": {
        "x": {"field": "fraction_ablated"},
        "y": {"field": "value"},
        "color": {"field": "metric"}
    },
    "baseSize": {"width": 550, "height": 400}
}
render_flint_chart(f5_data, f5_sem, f5_spec, "moe-expt-bias/figures_flint/fig5_ablation_curve.png")


# ==============================================================================
# FIGURE 6 — Demographic Specificity Heatmap (Experiment 5)
# ==============================================================================
groups = [
    "M", "F", "male", "sister", "mother",
    "African", "Arab", "Hispanic", "Japanese", "Bengali", "Muslim",
    "civil_servant", "software_developer", "nurse", "mover", "butcher", "plumber", "bartender"
]

display_names = [
    "Male Pronouns", "Female Pronouns", "Male Target", "Sister Target", "Mother Target",
    "African Cohort", "Arab Cohort", "Hispanic Cohort", "Japanese Cohort", "Bengali Cohort", "Muslim Cohort",
    "Civil Servant", "Software Dev", "Nurse", "Mover", "Butcher", "Plumber", "Bartender"
]

results_path = Path("moe-expt-bias/expt-bias-1/results/exp5-demographic-specificity-olmoe-1b-7b-v1")
vectors = []
valid_names = []

for g, name in zip(groups, display_names):
    file_path = results_path / f"phi_group_{g}.npy"
    if file_path.exists():
        v = np.load(file_path)
        v_abs = np.abs(v)
        v_abs /= (v_abs.sum() + 1e-12)
        vectors.append(v_abs)
        valid_names.append(name)

from scipy.spatial.distance import jensenshannon
num_groups = len(vectors)
fig6_rows = []
for i in range(num_groups):
    for j in range(num_groups):
        js_dist = jensenshannon(vectors[i], vectors[j])
        js_div = float(js_dist ** 2)
        fig6_rows.append({
            "cohort_a": valid_names[i],
            "cohort_b": valid_names[j],
            "js_divergence": round(js_div, 3)
        })

f6_data = {"values": fig6_rows}
f6_sem = {
    "cohort_a": "Category",
    "cohort_b": "Category",
    "js_divergence": "Quantity"
}
f6_spec = {
    "chartType": "Heatmap",
    "encodings": {
        "x": {"field": "cohort_a"},
        "y": {"field": "cohort_b"},
        "color": {"field": "js_divergence"}
    },
    "baseSize": {"width": 600, "height": 500}
}
render_flint_chart(f6_data, f6_sem, f6_spec, "moe-expt-bias/figures_flint/fig6_demographic_divergence.png")

print("\nAll 6 figures compiled and rendered successfully via Microsoft Flint to moe-expt-bias/figures_flint/!")
