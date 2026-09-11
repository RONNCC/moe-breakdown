#!/usr/bin/env python3
"""s09: Per-benchmark concentration split (E11).

Analysis-only, no GPU. Uses existing per_pair_phi.npy + pair_meta.json
to compute H/G/t5/t10 separately for each benchmark (stereoset, bbq, winogender).

This addresses Gap D/F/X: pooled result mixes incomparable constructs.
If per-benchmark H differs substantially, pooling is misleading.

Outputs: outputs/s09_per_benchmark.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from collections import Counter

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
    return float(h/np.log(n))

def gini(p):
    mass=np.sort(np.abs(p))
    n=len(mass)
    if n==0 or mass.sum()==0: return 0.0
    cum=np.cumsum(mass)
    return float((n+1-2*np.sum(cum)/cum[-1])/n)

def top_fraction(p, top_n):
    total=np.abs(p).sum()
    if total<=0: return 0.0
    top=np.sort(np.abs(p))[::-1][:top_n]
    return float(top.sum()/total)

def concentration_metrics(p):
    n=len(p)
    top10pct_n=max(1,int(round(0.10*n)))
    return {
        "entropy": normalized_entropy(p),
        "gini": gini(p),
        "t5": top_fraction(p,5),
        "t10": top_fraction(p,top10pct_n),
    }

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

def load_meta(exp_dir: Path):
    mf=exp_dir/"pair_meta.json"
    if not mf.exists():
        return None
    try:
        d=json.loads(mf.read_text())
    except:
        return None
    if isinstance(d, dict):
        # dict of dicts
        vals=list(d.values()) if d and isinstance(next(iter(d.values())), dict) else list(d)
        # but original is list
        if isinstance(d, list):
            pairs=d
        else:
            pairs=vals
            # Actually if dict values are dicts, we already have list
            # Try to handle list case properly
            if isinstance(d, list):
                pairs=d
            else:
                # if dict keys are indices, values are dicts
                if len(vals)>0 and isinstance(vals[0], dict):
                    pairs=vals
                else:
                    pairs=d if isinstance(d, list) else []
        # fallback: if d is list, use it
        if isinstance(d, list):
            pairs=d
    else:
        pairs=d if isinstance(d, list) else []
    # Re-read correctly
    if isinstance(d, list):
        pairs=d
    else:
        # attempt to get list from file directly
        try:
            pairs=json.loads(mf.read_text())
            if not isinstance(pairs, list):
                pairs=[]
        except:
            pairs=[]
    benchmarks=[]
    for e in pairs:
        if isinstance(e, dict):
            b=e.get("benchmark") or "unknown"
            benchmarks.append(b)
        else:
            benchmarks.append(str(e))
    return benchmarks, pairs

def process(exp_dir: Path):
    phi_file=find_phi(exp_dir)
    if phi_file is None:
        return {"status":"MISSING"}
    phi=np.load(phi_file)
    if phi.ndim!=2:
        return {"status":"MISSING","reason":"not 2D"}
    n_pairs,n_players=phi.shape
    meta=load_meta(exp_dir)
    if meta is None:
        return {"status":"no_meta","n_pairs":n_pairs}
    benchmarks,_=meta
    if len(benchmarks)!=n_pairs:
        # try fallback to sibling v1 meta
        if not exp_dir.name.endswith("-v1"):
            v1=Path(str(exp_dir)+"-v1")/"pair_meta.json"
            if v1.exists():
                try:
                    d=json.loads(v1.read_text())
                    benchmarks=[e.get("benchmark","unknown") if isinstance(e,dict) else str(e) for e in d]
                except:
                    pass
    if len(benchmarks)!=n_pairs:
        return {"status":"meta_len_mismatch","n_pairs":n_pairs,"meta_len":len(benchmarks)}

    counter=Counter(benchmarks)
    per_bench={}
    for bench in sorted(set(benchmarks)):
        idx=[i for i,b in enumerate(benchmarks) if b==bench]
        if len(idx)<5:
            continue
        p=phi[idx].mean(axis=0)
        # normalize for metric
        # keep signed mean then metric does abs inside
        per_bench[bench]={
            "n_pairs": len(idx),
            "metrics": concentration_metrics(p/np.abs(p).sum() if np.abs(p).sum()>0 else p),
            "raw_mean_abs_sum": float(np.abs(p).sum()),
        }
    # overall
    p_all=phi.mean(axis=0)
    overall=concentration_metrics(p_all/np.abs(p_all).sum() if np.abs(p_all).sum()>0 else p_all)
    # heterogeneity: Cochran Q-like: variance of H across benches
    Hs=[v["metrics"]["entropy"] for v in per_bench.values()]
    heterogeneity = float(np.std(Hs)) if len(Hs)>1 else 0.0
    return {
        "status":"ok",
        "n_pairs":n_pairs,
        "n_players":n_players,
        "benchmark_counts": dict(counter),
        "per_benchmark": per_bench,
        "overall": overall,
        "heterogeneity_std_H": heterogeneity,
    }

def main():
    exp_dirs=sorted([d for d in RESULTS.glob("exp1-concentration-*") if not d.name.endswith("-smoke")]) + \
             sorted(RESULTS.glob("exp2-dense-*")) + \
             sorted(RESULTS.glob("exp8-lloo-*"))
    out={}
    for ed in exp_dirs:
        key=ed.name
        print(f"Processing {key} ...")
        out[key]=process(ed)
    # also include summary table for paper ladder v1 only
    ladder_v1=["exp1-concentration-olmoe-1b-7b-v1","exp1-concentration-phi3.5-moe-v1","exp1-concentration-mixtral-8x7b-v1","exp1-concentration-dbrx-v1","exp1-concentration-gpt-oss-120b-v1","exp1-concentration-gemma4-26b-v1"]
    summary={}
    for name in ladder_v1:
        ed=RESULTS/name
        if ed.exists():
            res=out.get(name, {})
            if res.get("status")=="ok":
                summary[name]={
                    "counts": res["benchmark_counts"],
                    "H_stereoset": res["per_benchmark"].get("stereoset",{}).get("metrics",{}).get("entropy"),
                    "H_bbq": res["per_benchmark"].get("bbq",{}).get("metrics",{}).get("entropy"),
                    "overall_H": res["overall"]["entropy"],
                    "heterogeneity": res["heterogeneity_std_H"],
                }
    out["_summary_ladder_v1"]=summary
    out_path=OUT/"s09_per_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(summary, indent=2))

if __name__=="__main__":
    main()
