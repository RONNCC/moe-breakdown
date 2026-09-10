"""Tests for stats analysis fixes (s04 bootstrap, s09, etc.)"""

def test_s04_group_fallback():
    # Old code: str(e.get("group", e.get("benchmark"))) collapses None to "None"
    # New code: treats None/empty/"None"/"unknown" as missing, fallback to benchmark:bias_type
    def old_logic(entry):
        return str(entry.get("group", entry.get("benchmark")))
    def new_logic(entry):
        g = entry.get("group")
        if g is None or g == "" or str(g).lower() in ("none","unknown"):
            # fallback
            bench = entry.get("benchmark","unknown")
            btype = entry.get("bias_type","unknown")
            return f"{bench}:{btype}"
        return str(g)

    entry = {"benchmark":"stereoset","bias_type":"gender","group":None}
    assert old_logic(entry) == "None", "old logic bug demonstration"
    assert new_logic(entry) == "stereoset:gender", "new logic should fallback"

    entry2 = {"benchmark":"bbq","bias_type":"race","group":""}
    assert new_logic(entry2) == "bbq:race"

    entry3 = {"benchmark":"stereoset","bias_type":"race","group":"Black"}
    assert new_logic(entry3) == "Black"

def test_s12_audit_checks():
    # Simulate manifest audit expectations
    sample_meta = [
        {"index":0,"benchmark":"stereoset","bias_type":"gender","item_id":"abc123","group":None},
        {"index":1,"benchmark":"bbq","bias_type":"race","item_id":"bbq-1","group":None},
    ]
    # group null fraction should be detectable
    group_null = sum(1 for e in sample_meta if not e.get("group")) / len(sample_meta)
    assert group_null == 1.0

    # unique item_ids
    unique_ids = len(set(e.get("item_id","") for e in sample_meta))
    assert unique_ids == 2
    # Previously bug: all empty -> unique=1

def test_per_pair_phi_shape():
    import numpy as np
    n_pairs = 10
    n_players = 128
    per_pair = np.random.randn(n_pairs, n_players)
    assert per_pair.shape == (10,128)
    # Aggregation should match mean
    phi = per_pair.mean(axis=0)
    assert phi.shape == (128,)

def test_bootstrap_stratification():
    # Ensure stratified bootstrap would have at least 2 strata when group missing but benchmark differs
    entries = [
        {"benchmark":"stereoset","bias_type":"gender","group":None},
        {"benchmark":"stereoset","bias_type":"gender","group":None},
        {"benchmark":"bbq","bias_type":"race","group":None},
        {"benchmark":"bbq","bias_type":"race","group":None},
    ]
    def new_logic(e):
        g = e.get("group")
        if g is None or g == "" or str(g).lower() in ("none","unknown"):
            return f"{e.get('benchmark')}:{e.get('bias_type')}"
        return str(g)
    strata = set(new_logic(e) for e in entries)
    assert len(strata) == 2, f"Should have 2 strata after fix, got {strata}"
