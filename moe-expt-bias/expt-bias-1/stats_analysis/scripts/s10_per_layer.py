#!/usr/bin/env python3
"""s10: Per-layer concentration analysis (E12).

Reshapes flattened phi (n_pairs, n_players) into (n_pairs, n_layers, experts_per_layer)
using player_ids.json to infer layer count, then computes per-layer H.

Addresses Gap R: token/layer aggregation hides depth effects. Exp3 shows synergy
varies 0.7->0.2 from first to last layer; per-layer H likely varies similarly.

Outputs: outputs/s10_per_layer.json + figure s10_per_layer.png (optional)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import re

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def normalized_entropy(p):
    n=len(p)
    if n<=1: return 0.0
    total=np.abs(p).sum()
    if total<=0: return 0.0
    phat=np.abs(p)/total
    nz=phat[phat>0]
    h=-np.sum(nz*np.log(nz))
    return float(h/np.log(n)) if n>1 else 0.0

def find_phi(exp_dir: Path):
    for pat in ("per_pair_phi.npy","per_pair_phi-v1.npy","per_pair_phi_v1.npy"):
        f=exp_dir/pat
        if f.exists():
            return f
    gl=list(exp_dir.glob("per_pair_phi*.npy"))
    if gl:
        return gl[0]
    if not exp_dir.name.endswith("-v1"):
        v1=Path(str(exp_dir)+"-v1")
        if v1.exists():
            for pat in ("per_pair_phi.npy","per_pair_phi-v1.npy"):
                f=v1/pat
                if f.exists():
                    return f
    return None

def infer_layers(exp_dir: Path):
    pid_file=exp_dir/"player_ids.json"
    if not pid_file.exists() and not exp_dir.name.endswith("-v1"):
        v1=Path(str(exp_dir)+"-v1")/"player_ids.json"
        if v1.exists():
            pid_file=v1
    if not pid_file.exists():
        return None
    try:
        ids=json.loads(pid_file.read_text())
    except:
        return None
    # ids like "layer0-expert0"
    layers={}
    pattern=re.compile(r"layer(\d+)-expert(\d+)")
    for idx, pid in enumerate(ids):
        m=pattern.match(pid)
        if m:
            li=int(m.group(1))
            ei=int(m.group(2))
            layers.setdefault(li, []).append((ei, idx))
    if not layers:
        # dense case: layer0-ffn
        pattern2=re.compile(r"layer(\d+)-ffn")
        for idx, pid in enumerate(ids):
            m=pattern2.match(pid)
            if m:
                li=int(m.group(1))
                layers.setdefault(li, []).append((0, idx))
        if layers:
            return {"n_layers": len(layers), "experts_per_layer": 1, "layer_map": layers, "type":"dense"}
        return None
    # sort
    n_layers=len(layers)
    # assume uniform experts per layer
    epl=len(layers[0]) if 0 in layers else len(next(iter(layers.values())))
    return {"n_layers": n_layers, "experts_per_layer": epl, "layer_map": layers, "type":"moe"}

def process(exp_dir: Path):
    phi_file=find_phi(exp_dir)
    if phi_file is None:
        return {"status":"MISSING"}
    phi=np.load(phi_file)
    if phi.ndim!=2:
        return {"status":"MISSING","reason":"not 2D"}
    n_pairs,n_players=phi.shape
    info=infer_layers(exp_dir)
    if info is None:
        return {"status":"no_layer_info","n_players":n_players}
    n_layers=info["n_layers"]
    epl=info["experts_per_layer"]
    if n_players != n_layers*epl and info["type"]=="moe":
        # try to infer via division
        # some models have varying experts per layer? but ours uniform
        return {"status":"layer_mismatch","n_players":n_players,"n_layers":n_layers,"epl":epl}
    # compute mean phi over pairs first
    mean_phi=phi.mean(axis=0)  # (n_players,)
    # reshape per layer
    per_layer_H=[]
    per_layer_G=[]
    per_layer_abs_mass=[]
    layer_map=info["layer_map"]
    for li in sorted(layer_map.keys()):
        indices=[idx for _,idx in sorted(layer_map[li])]
        layer_phi=mean_phi[indices]
        abs_sum=float(np.abs(layer_phi).sum())
        H=normalized_entropy(layer_phi)
        # gini
        mass=np.sort(np.abs(layer_phi))
        if mass.sum()==0:
            G=0.0
        else:
            cum=np.cumsum(mass)
            G=float((len(mass)+1-2*np.sum(cum)/cum[-1])/len(mass))
        per_layer_H.append(H)
        per_layer_G.append(G)
        per_layer_abs_mass.append(abs_sum)

    # also compute per-pair per-layer H variance (optional)
    # For each pair, compute layer H
    per_pair_layer_H=[]
    for j in range(min(n_pairs, 100)):  # limit to 100 pairs for speed
        pj=phi[j]
        layer_H=[]
        for li in sorted(layer_map.keys()):
            indices=[idx for _,idx in sorted(layer_map[li])]
            layer_H.append(normalized_entropy(pj[indices]))
        per_pair_layer_H.append(layer_H)

    return {
        "status":"ok",
        "n_pairs":n_pairs,
        "n_players":n_players,
        "n_layers":n_layers,
        "experts_per_layer":epl,
        "type":info["type"],
        "mean_per_layer_H": per_layer_H,
        "mean_per_layer_G": per_layer_G,
        "mean_per_layer_abs_mass": per_layer_abs_mass,
        "mean_H_std_across_layers": float(np.std(per_layer_H)) if per_layer_H else 0.0,
        "first_layer_H": per_layer_H[0] if per_layer_H else None,
        "last_layer_H": per_layer_H[-1] if per_layer_H else None,
        "per_pair_layer_H_sample": per_pair_layer_H[:5],  # first 5 pairs
    }

def main():
    exp_dirs=sorted([d for d in RESULTS.glob("exp1-concentration-*") if not d.name.endswith("-smoke")])
    out={}
    for ed in exp_dirs:
        print(f"Processing {ed.name}")
        out[ed.name]=process(ed)
    # summary for paper ladder v1
    ladder=["exp1-concentration-olmoe-1b-7b-v1","exp1-concentration-phi3.5-moe-v1","exp1-concentration-mixtral-8x7b-v1","exp1-concentration-dbrx-v1","exp1-concentration-gpt-oss-120b-v1","exp1-concentration-gemma4-26b-v1"]
    summary={}
    for name in ladder:
        r=out.get(name)
        if r and r.get("status")=="ok":
            summary[name]={
                "n_layers": r["n_layers"],
                "epl": r["experts_per_layer"],
                "first_H": r["first_layer_H"],
                "last_H": r["last_layer_H"],
                "std_H": r["mean_H_std_across_layers"],
                "mean_H": float(np.mean(r["mean_per_layer_H"])),
            }
    out["_summary"]=summary
    out_path=OUT/"s10_per_layer.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(summary, indent=2))

if __name__=="__main__":
    main()
