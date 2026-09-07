#!/usr/bin/env bash
# Batch: every policy x a workflow (default staircase_chain). Writes results/.
set -e
WF="${1:-staircase_chain}"; N="${2:-6}"
for pol in s_cache cocache dsp pcache dl_lru dl_mac offline_opt; do
  echo "=== $WF / $pol ==="
  python run_dag_workflow.py --workflow "$WF" --policy "$pol" --instances "$N"
done
