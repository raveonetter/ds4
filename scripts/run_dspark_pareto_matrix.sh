#!/usr/bin/env bash
set -euo pipefail

# Run the existing DSpark quality matrix unchanged, then evaluate the exact same
# prompt/arm outputs on the deployment objective:
#   1) generation_tps
#   2) first generated token divergence vs sequential

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-matrix}
export RUN_ID OUT_ROOT

./scripts/run_dspark_quality_matrix.sh

python3 ./scripts/dspark_token_pareto.py \
  --run-dir "$OUT_ROOT/$RUN_ID" \
  --ds4-bin "${DS4_BIN:-./ds4}" \
  --model "${MODEL:-./ds4flash.gguf}"

echo
echo "PARETO_RESULTS_DIR=$OUT_ROOT/$RUN_ID"
