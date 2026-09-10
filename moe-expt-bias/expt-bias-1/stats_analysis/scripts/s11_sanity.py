#!/usr/bin/env python3
"""s11: Sanity controls for routing-contrast (E13 / Gap Z).

Tests whether H=0.88-0.92 is a property of model or estimator bias.

Controls:
1. Random router shuffle: shuffle expert IDs within each layer per pair,
   recompute mean phi -> should give H~1 if estimator sensitive to routing.
2. Label shuffle: swap stereo/anti per pair -> mean gap ~0, H should stay same if symmetric.
3. Weight randomization thought experiment: if all phi equal, H=1.

We can't rerun model inference here (no GPU), but we can simulate controls
on existing per_pair_phi.npy by shuffling.

For true random router control, we need to shuffle indices within layer.

Outputs: outputs/s11_sanity.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import re
from collections import defaultdict

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

def infer_layer_groups(exp_dir: Path):
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
    pat=re.compile(r"layer(\d+)-expert(\d+)")
    groups=defaultdict(list)
    for idx, pid in enumerate(ids):
        m=pat.match(pid)
        if m:
            li=int(m.group(1))
            groups[li].append(idx)
    if not groups:
        return None
    return groups

def process(exp_dir: Path, rng, n_shuffle=20):
    phi_file=find_phi(exp_dir)
    if phi_file is None:
        return {"status":"MISSING"}
    phi=np.load(phi_file)
    if phi.ndim!=2:
        return {"status":"MISSING"}
    n_pairs,n_players=phi.shape
    mean_phi=phi.mean(axis=0)
    H_orig=normalized_entropy(mean_phi)

    # 1. Label shuffle: flip sign of half pairs (swap stereo/anti -> gap sign flip)
    # routing_contrast phi is proportional to gap, so flipping sign per pair
    # If we randomly flip sign per pair, mean should go to ~0 if gaps are consistent,
    # but H of absolute mean may stay similar or become noisier.
    # Simulate by multiplying each pair's phi by random +/-1
    H_label_shuffled=[]
    for _ in range(n_shuffle):
        signs=rng.choice([-1,1], size=n_pairs)
        phi_shuffled=phi * signs[:,None]
        H_label_shuffled.append(normalized_entropy(phi_shuffled.mean(axis=0)))
    H_label_mean=float(np.mean(H_label_shuffled))
    H_label_std=float(np.std(H_label_shuffled))

    # 2. Random router shuffle: within each layer, permute expert indices per pair
    layer_groups=infer_layer_groups(exp_dir)
    H_router_shuffled=[]
    if layer_groups is not None:
        for _ in range(n_shuffle):
            phi_perm=np.empty_like(phi)
            for j in range(n_pairs):
                for li, idxs in layer_groups.items():
                    perm=rng.permutation(idxs)
                    # Actually we want to shuffle values among experts within layer
                    # So take phi[j, idxs] and permute
                    phi_perm[j, idxs]=phi[j, perm]
                # For indices not in any layer group (shouldn't happen), copy
                # (handled by perm above covering all)
            H_router_shuffled.append(normalized_entropy(phi_perm.mean(axis=0)))
        H_router_mean=float(np.mean(H_router_shuffled)) if H_router_shuffled else None
        H_router_std=float(np.std(H_router_shuffled)) if H_router_shuffled else None
    else:
        H_router_mean=None
        H_router_std=None

    # 3. Uniform baseline: if all phi equal, H=1
    uniform_phi=np.ones(n_players)
    H_uniform=normalized_entropy(uniform_phi)

    # 4. Single-spike baseline: one expert holds all mass -> H=0
    spike=np.zeros(n_players)
    spike[0]=1.0
    H_spike=normalized_entropy(spike)

    return {
        "status":"ok",
        "n_pairs":n_pairs,
        "n_players":n_players,
        "H_original": H_orig,
        "H_uniform_baseline": H_uniform,
        "H_spike_baseline": H_spike,
        "H_label_shuffle_mean": H_label_mean,
        "H_label_shuffle_std": H_label_std,
        "H_router_shuffle_mean": H_router_mean,
        "H_router_shuffle_std": H_router_std,
        "interpretation": {
            "label_shuffle": "If H stays ~same after random sign flip, estimator is symmetric but not sensitive to gap direction; if H drops to ~0 or becomes unstable, gap cancellation matters",
            "router_shuffle": "If H_router_shuffle ~ H_original, then observed H is not due to routing structure but estimator bias (shuffling within layer shouldn't change much if diffuse); if H_router_shuffle ~1, then original routing has structure",
        }
    }

def main():
    rng=np.random.default_rng(42)
    exp_dirs=sorted([d for d in RESULTS.glob("exp1-concentration-*") if not d.name.endswith("-smoke")])
    out={}
    for ed in exp_dirs:
        if "v1" not in ed.name and (RESULTS / f"{ed.name}-v1").exists():
            continue  # prefer v1
        print(f"Processing {ed.name}")
        out[ed.name]=process(ed, rng)
    out_path=OUT/"s11_sanity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")
    # summary
    for k,v in out.items():
        if v.get("status")=="ok":
            print(f"{k}: H_orig={v['H_original']:.4f} H_label_shuf={v['H_label_shuffle_mean']:.4f}±{v['H_label_shuffle_std']:.4f} H_router_shuf={v['H_router_shuffle_mean']}")

if __name__=="__main__":
    main()
