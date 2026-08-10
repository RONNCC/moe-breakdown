#!/usr/bin/env bash
# run_bootstrap_all.sh — idempotent launcher for s04_bootstrap_cis.py.
#
# s04 takes no CLI args: it globs results/exp1-concentration-*/ itself and
# computes per-model bootstrap CIs (or emits MISSING entries) in one shot.
# This wrapper mirrors that contract at the shell level:
#   1. For each model dir, report READY (per_pair_phi*.npy present) or
#      SKIP <model> (per-pair payload missing).
#   2. If >=1 payload is present, run s04 once (it covers all models).
#   3. If none are present, do not invoke s04; exit 0.
# No files are written by the wrapper; re-running is safe.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # expt-bias-1/
RESULTS="$ROOT/results"
S04="$ROOT/stats_analysis/scripts/s04_bootstrap_cis.py"

PY="python3"
for cand in "$ROOT/stats_analysis/.venv/bin/python" "$ROOT/.venv/bin/python"; do
  [[ -x "$cand" ]] && PY="$cand" && break
done

have=0
for d in "$RESULTS"/exp1-concentration-*; do
  [[ -d "$d" ]] || continue
  model="${d##*/}"
  model="${model#exp1-concentration-}"
  if [[ -n "$(ls "$d"/per_pair_phi*.npy 2>/dev/null | head -n 1)" ]]; then
    echo "READY $model (per-pair payload present)"
    have=$((have + 1))
  else
    echo "SKIP $model (per-pair payload missing)"
  fi
done

if [[ "$have" -eq 0 ]]; then
  echo "== no per-pair payloads on disk; not invoking s04 (all MISSING) =="
  exit 0
fi

echo "== per-pair payloads found for ${have} model(s); running s04 =="
exec "$PY" "$S04"