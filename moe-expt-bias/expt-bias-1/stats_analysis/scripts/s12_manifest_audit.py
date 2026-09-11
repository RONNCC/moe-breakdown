#!/usr/bin/env python3
"""s12: Manifest audit – benchmark composition, seed usage, common intersection (E9/E14).

This script does NOT require GPU. It audits:
- pair_meta.json benchmark composition per result dir (already known: 2106 stereoset + 2894 bbq for most v1)
- Whether seed is used (it isn't in benchmarks.py load_benchmarks)
- Common intersection size across models (StereoSet-only is the only common set)
- Generates a proposed frozen manifest spec for future reruns
- Checks for provenance fields in result.json (commit, config hash, GPU, etc.)

Outputs: outputs/s12_manifest_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def load_meta(exp_dir: Path):
    mf=exp_dir/"pair_meta.json"
    if not mf.exists():
        return None
    try:
        d=json.loads(mf.read_text())
    except:
        return None
    if not isinstance(d, list):
        return None
    return d

def audit():
    exp_dirs=sorted([d for d in RESULTS.glob("exp1-concentration-*") if not d.name.endswith("-smoke")]) + \
             sorted(RESULTS.glob("exp2-dense-*")) + \
             sorted(RESULTS.glob("exp8-lloo-*"))
    per_dir={}
    all_item_ids=defaultdict(set)  # item_id -> set of dirs containing it
    for ed in exp_dirs:
        meta=load_meta(ed)
        if meta is None:
            per_dir[ed.name]={"status":"no_meta"}
            continue
        benchmarks=[e.get("benchmark","unknown") if isinstance(e,dict) else str(e) for e in meta]
        bias_types=[e.get("bias_type","unknown") if isinstance(e,dict) else "unknown" for e in meta]
        # item_id may be example_id or id
        item_ids=[e.get("item_id","") if isinstance(e,dict) else "" for e in meta]
        bc=Counter(benchmarks)
        btc=Counter(bias_types)
        # check for group field
        groups=[e.get("group") if isinstance(e,dict) else None for e in meta]
        n_none=sum(1 for g in groups if g is None)
        # provenance in result.json
        res_file=ed/"result.json"
        prov={}
        if res_file.exists():
            try:
                r=json.loads(res_file.read_text())
                meta_field=r.get("metadata",{})
                prov={
                    "has_metadata": bool(meta_field),
                    "metadata_keys": list(meta_field.keys()) if isinstance(meta_field, dict) else [],
                    "n_pairs": r.get("n_pairs"),
                    "has_commit": "commit" in str(r) or "git" in str(meta_field).lower(),
                    "has_config_hash": "config" in str(meta_field).lower(),
                    "has_seed": "seed" in str(meta_field).lower(),
                }
            except:
                prov={"error":"failed to parse result.json"}
        per_dir[ed.name]={
            "status":"ok",
            "n_pairs":len(meta),
            "benchmark_counts": dict(bc),
            "bias_type_counts": dict(btc),
            "n_group_none": n_none,
            "group_null_fraction": n_none/len(meta) if meta else 0,
            "provenance": prov,
            "unique_item_ids": len(set(item_ids)),
            "sample_item_ids": item_ids[:3],
        }
        for iid in item_ids:
            if iid:
                all_item_ids[iid].add(ed.name)

    # common intersection: item_ids present in ALL dirs that have meta
    # For ladder v1 only, common intersection is likely StereoSet subset
    ladder_v1=["exp1-concentration-olmoe-1b-7b-v1","exp1-concentration-phi3.5-moe-v1","exp1-concentration-mixtral-8x7b-v1","exp1-concentration-dbrx-v1","exp1-concentration-gpt-oss-120b-v1","exp1-concentration-gemma4-26b-v1"]
    ladder_dirs=[RESULTS/d for d in ladder_v1 if (RESULTS/d).exists()]
    # compute intersection of item_ids across ladder
    item_sets=[]
    for ed in ladder_dirs:
        meta=load_meta(ed)
        if meta:
            ids=set(e.get("item_id","") for e in meta if isinstance(e,dict))
            item_sets.append(ids)
    common_intersection=set.intersection(*item_sets) if item_sets else set()

    # per-benchmark intersection
    # StereoSet ids are from McGill-NLP/stereoset validation split
    # For now, compute how many stereoset ids are common
    stereoset_common=None
    if item_sets:
        # filter to stereoset only
        stereo_sets=[]
        for ed in ladder_dirs:
            meta=load_meta(ed)
            if meta:
                s_ids=set(e.get("item_id","") for e in meta if isinstance(e,dict) and e.get("benchmark")=="stereoset")
                stereo_sets.append(s_ids)
        if stereo_sets:
            stereoset_common=set.intersection(*stereo_sets)

    # propose frozen manifest spec
    proposed_manifest={
        "version":"v1-frozen-2026-09-10",
        "total_pairs":5000,
        "benchmarks":{
            "stereoset": {"n":2106, "source":"McGill-NLP/stereoset validation", "stratify_by":"bias_type"},
            "bbq": {"n":2894, "source":"heegyu/bbq parquet", "categories":["Age","Disability_status","Gender_identity","Nationality","Physical_appearance","Race_ethnicity","Race_x_SES","Race_x_gender","Religion","SES","Sexual_orientation"], "condition":"ambig", "stratify_by":"category"},
            "winogender": {"n":0, "note":"currently 0 in all v1, need to add with occupation stats for direction"},
        },
        "sampling_rule":"seeded shuffle with seed 42, then take first N per benchmark, record manifest hash",
        "required_fields_per_item":["item_id","benchmark","bias_type","source","template_key","group","stereo","anti_stereo"],
        "provenance_fields":["commit_hash","config_hash","model_revision","dataset_revision","seed","gpu_type","precision","manifest_hash","timestamp"],
        "notes":[
            "Current runs use deterministic concatenation + slice, seed recorded but not used – fix in benchmarks.py",
            "BBQ loader must use target_loc + question_polarity to identify stereotyped answer, not arbitrary non-unknown",
            "WinoGender must derive stereotype direction from BLS occupation stats, not male=stereo",
            "pair_meta.json group field is None for all items – fix to use benchmark:bias_type",
            "Common intersection across ladder v1 is 0 if including gpt-oss-120b-v1 (2000 stereoset only) – recompute on stereoset-only for interim comparison",
        ]
    }

    out={
        "per_dir": per_dir,
        "common_intersection": {
            "ladder_v1_all": {"n": len(common_intersection), "sample": list(common_intersection)[:5]},
            "ladder_v1_stereoset_only": {"n": len(stereoset_common) if stereoset_common else 0, "sample": list(stereoset_common)[:5] if stereoset_common else []},
        },
        "proposed_manifest": proposed_manifest,
        "audit_findings": {
            "seed_not_used": "load_benchmarks in benchmarks.py has max_items slice but no seeded shuffle; seed field in config is recorded but not used for sampling",
            "bbq_bug": "biased_ans = next(a for i!=label) picks arbitrary non-unknown, not stereotyped – ignores target_loc and polarity",
            "winogender_bug": "stereo=male always, anti=female always – ignores occupation stats",
            "group_null": "pair_meta.json group=None for all items – breaks stratified bootstrap",
            "benchmark_mixture_varies": "MoE v1 mostly 2106+2894, GPT-OSS v1 2000 stereoset only, dense v1 1800 stereoset only except phi-mini 2106+1894, LOO 100/50 stereoset only",
        }
    }
    return out

def main():
    out=audit()
    out_path=OUT/"s12_manifest_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(out["audit_findings"], indent=2))
    print("\nCommon intersection ladder v1:", out["common_intersection"])

if __name__=="__main__":
    main()
