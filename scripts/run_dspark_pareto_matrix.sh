#!/usr/bin/env bash
set -euo pipefail

# Run the existing DSpark matrix unchanged, append generic-based family
# ablations, then report the deployment objective:
#   1) generation_tps
#   2) first generated token divergence vs sequential
#
# Default family search:
#   singleton F3/F4/F5/F6/F7 + targeted F4+F5
#
# F1 and F1+F3 already come from run_dspark_quality_matrix.sh.
# Old byte/checkpoint-oriented outputs remain under the run directory and the
# complete legacy runner stdout is retained in legacy-matrix.log.
#
# FAMILY_SWEEP_MODE:
#   ablation   default; singleton F3-F7 plus F4+F5
#   singleton  only F3-F7
#   targeted   only F4+F5
#   cumulative historical cumulative chain
#   all        all of the above
#   off        no additional family runs

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-matrix}
FAMILY_SWEEP_MODE=${FAMILY_SWEEP_MODE:-ablation}
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
echo "ROBUST_PARETO=$RUN_DIR/robust-pareto.tsv"
echo "LEGACY_DIAGNOSTICS=$RUN_DIR/legacy-matrix.log"
