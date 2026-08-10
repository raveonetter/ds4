#!/usr/bin/env bash
set -euo pipefail

# Run the existing DSpark matrix unchanged, add cumulative implemented-family
# variants, then make the deployment objective the primary report:
#   1) generation_tps
#   2) first generated token divergence vs sequential
#
# The old byte/checkpoint-oriented tables remain in the run directory and the
# complete legacy runner stdout is kept as legacy-matrix.log for diagnostics.
# FAMILY_SWEEP_MODE may be cumulative (default), individual, all, or off.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-matrix}
FAMILY_SWEEP_MODE=${FAMILY_SWEEP_MODE:-cumulative}
export RUN_ID OUT_ROOT FAMILY_SWEEP_MODE

RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

bash ./scripts/run_dspark_quality_matrix.sh >"$RUN_DIR/legacy-matrix.log"

RUN_DIR="$RUN_DIR" FAMILY_SWEEP_MODE="$FAMILY_SWEEP_MODE" \
  bash ./scripts/run_dspark_family_variants.sh >>"$RUN_DIR/legacy-matrix.log"

python3 ./scripts/dspark_token_pareto.py \
  --run-dir "$RUN_DIR" \
  --ds4-bin "${DS4_BIN:-./ds4}" \
  --model "${MODEL:-./ds4flash.gguf}"

echo
echo "PARETO_RESULTS_DIR=$RUN_DIR"
echo "LEGACY_DIAGNOSTICS=$RUN_DIR/legacy-matrix.log"
