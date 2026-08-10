#!/usr/bin/env bash
set -euo pipefail

# Staged generic DSpark confidence search.
#
# Stage 1 screens a small confidence grid once at a shorter horizon. Stage 2
# repeats only the screen Pareto confidence(s) and their immediate neighbours
# at the full horizon. This keeps the production objective unchanged while
# avoiding an exhaustive 5-repeat sweep of every confidence.
#
# Required:
#   SOURCE_RUN_DIR=quality-results/dspark-matrix/<run-id>
#
# Defaults:
#   SCREEN_CONFIDENCES="0.55 0.60 0.65 0.70 0.75"
#   SCREEN_TOKENS=512
#   REFINE_REPEATS=3
#   TOKENS=<source run TOKENS, usually 768>
#
# The exhaustive run_dspark_confidence_search.sh remains available for final
# confirmation after a narrow confidence region has been selected.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR to an existing quality-matrix run}
SCREEN_CONFIDENCES=${SCREEN_CONFIDENCES:-"0.55 0.60 0.65 0.70 0.75"}
SCREEN_TOKENS=${SCREEN_TOKENS:-512}
REFINE_REPEATS=${REFINE_REPEATS:-3}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-confidence-fast}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

if [[ -z "${TOKENS:-}" && -f "$SOURCE_RUN_DIR/config.txt" ]]; then
  TOKENS=$(awk -F= '$1=="TOKENS" {print $2; exit}' "$SOURCE_RUN_DIR/config.txt")
fi
TOKENS=${TOKENS:-768}

[[ "$REFINE_REPEATS" =~ ^[1-9][0-9]*$ ]] || {
  echo "REFINE_REPEATS must be a positive integer" >&2
  exit 2
}

SCREEN_ROOT="$OUT_ROOT/$RUN_ID"
SCREEN_DIR="$SCREEN_ROOT/screen"
REFINE_DIR="$SCREEN_ROOT/refine"
mkdir -p "$SCREEN_ROOT"

# Stage 1: cheap trajectory/speed screen.
SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
CONFIDENCES="$SCREEN_CONFIDENCES" \
REPEATS=1 \
TOKENS="$SCREEN_TOKENS" \
OUT_ROOT="$SCREEN_ROOT" \
RUN_ID=screen \
DS4_BIN="${DS4_BIN:-./ds4}" \
GENERIC_BIN="${GENERIC_BIN:-${DS4_BIN:-./ds4}}" \
MODEL="${MODEL:-./ds4flash.gguf}" \
DSPARK_MODEL="${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}" \
  bash ./scripts/run_dspark_confidence_search.sh

# Select all non-sequential robust Pareto confidences, plus their immediate
# neighbours in the screen grid. If the screen has no fast Pareto point, fall
# back to the confidence with the highest median throughput.
SELECTED_CONFIDENCES=$(python3 - "$SCREEN_DIR/robust-pareto.tsv" "$SCREEN_CONFIDENCES" <<'PY'
import csv
import sys

path, grid_text = sys.argv[1], sys.argv[2]
grid = grid_text.split()
rows = []
with open(path, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

selected = []
for r in rows:
    if r.get("arm", "").startswith("fast_generic_") and r.get("robust_pareto") == "YES":
        c = r.get("confidence", "")
        if c in grid:
            selected.append(c)

if not selected:
    candidates = []
    for r in rows:
        if not r.get("arm", "").startswith("fast_generic_"):
            continue
        try:
            tps = float(r.get("median_generation_tps", "NA"))
        except ValueError:
            continue
        candidates.append((tps, r.get("confidence", "")))
    if candidates:
        selected.append(max(candidates)[1])

expanded = set()
for c in selected:
    try:
        i = grid.index(c)
    except ValueError:
        continue
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(grid):
            expanded.add(grid[j])

# Preserve grid order for readable output and deterministic execution.
print(" ".join(c for c in grid if c in expanded))
PY
)

if [[ -z "$SELECTED_CONFIDENCES" ]]; then
  echo "screen produced no confidence candidates" >&2
  exit 2
fi

echo
echo "SCREEN_SELECTED_CONFIDENCES=$SELECTED_CONFIDENCES"
echo "REFINE_REPEATS=$REFINE_REPEATS TOKENS=$TOKENS"

# Stage 2: full-horizon repeated timing only near the screen frontier.
SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
CONFIDENCES="$SELECTED_CONFIDENCES" \
REPEATS="$REFINE_REPEATS" \
TOKENS="$TOKENS" \
OUT_ROOT="$SCREEN_ROOT" \
RUN_ID=refine \
DS4_BIN="${DS4_BIN:-./ds4}" \
GENERIC_BIN="${GENERIC_BIN:-${DS4_BIN:-./ds4}}" \
MODEL="${MODEL:-./ds4flash.gguf}" \
DSPARK_MODEL="${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}" \
  bash ./scripts/run_dspark_confidence_search.sh

echo
echo "FAST_CONFIDENCE_SEARCH_DIR=$SCREEN_ROOT"
echo "SCREEN_ROBUST_PARETO=$SCREEN_DIR/robust-pareto.tsv"
echo "REFINE_ROBUST_PARETO=$REFINE_DIR/robust-pareto.tsv"
