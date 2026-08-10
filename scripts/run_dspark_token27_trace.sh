#!/usr/bin/env bash
set -euo pipefail

# Localize the first warehouse token divergence to the DSpark commit iteration
# that emitted it. Production source is never modified: instrumentation is
# injected only into a temporary detached git worktree, then deleted.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-./quality-results/dspark-matrix/20260810-181808}
DS4_BIN=${DS4_BIN:-./ds4}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
CONFIDENCE=${CONFIDENCE:-0.80}
TOKENS=${TOKENS:-64}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-token27}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR="$OUT_ROOT/$RUN_ID"

abs_path() {
  python3 - "$1" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

SOURCE_RUN_DIR=$(abs_path "$SOURCE_RUN_DIR")
DS4_BIN=$(abs_path "$DS4_BIN")
MODEL=$(abs_path "$MODEL")
DSPARK_MODEL=$(abs_path "$DSPARK_MODEL")
OUT_DIR=$(abs_path "$OUT_DIR")
PROMPT_FILE="$SOURCE_RUN_DIR/prompts/warehouse.txt"
ORACLE_FILE="$SOURCE_RUN_DIR/warehouse/sequential.txt"
TRACE_TOOL="$ROOT_DIR/scripts/dspark_token27_trace.py"

[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || { echo "not a git worktree: $ROOT_DIR" >&2; exit 2; }
[[ -x "$DS4_BIN" ]] || { echo "missing executable $DS4_BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model $MODEL" >&2; exit 2; }
[[ -f "$DSPARK_MODEL" ]] || { echo "missing DSpark support model $DSPARK_MODEL" >&2; exit 2; }
[[ -f "$PROMPT_FILE" ]] || { echo "missing warehouse prompt $PROMPT_FILE" >&2; exit 2; }
[[ -f "$ORACLE_FILE" ]] || { echo "missing sequential oracle $ORACLE_FILE" >&2; exit 2; }
[[ -f "$TRACE_TOOL" ]] || { echo "missing trace tool $TRACE_TOOL" >&2; exit 2; }
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ds4-token27.XXXXXX")
TRACE_WT="$TMP_ROOT/worktree"
WORKTREE_ADDED=0
cleanup() {
  if [[ "$WORKTREE_ADDED" == 1 ]]; then
    git -C "$ROOT_DIR" worktree remove --force "$TRACE_WT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

echo "TOKEN27_SETUP source_run=$SOURCE_RUN_DIR confidence=$CONFIDENCE tokens=$TOKENS"
git -C "$ROOT_DIR" worktree add --detach "$TRACE_WT" HEAD >/dev/null
WORKTREE_ADDED=1
python3 "$TRACE_TOOL" instrument --source "$TRACE_WT/ds4.c"

# Build only in the temporary worktree. The user's production source/object/
# binary in ROOT_DIR are never touched.
MAKE_JOBS=${MAKE_JOBS:-4}
make -C "$TRACE_WT" -j "$MAKE_JOBS" ds4 >/dev/null
TRACE_BIN="$TRACE_WT/ds4"
[[ -x "$TRACE_BIN" ]] || { echo "trace build did not produce $TRACE_BIN" >&2; exit 2; }

PROMPT=$(cat "$PROMPT_FILE")
CANDIDATE_OUT="$OUT_DIR/warehouse-fast-generic-c0p80.txt"
CANDIDATE_LOG="$OUT_DIR/warehouse-fast-generic-c0p80.log"

(
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
  env DS4_DSPARK_STATS=1 \
      DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1 \
      DS4_DSPARK_TRACE_COMMITS=1 \
      "$TRACE_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink \
      --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$CONFIDENCE" -p "$PROMPT"
) >"$CANDIDATE_OUT" 2>"$CANDIDATE_LOG"

python3 "$TRACE_TOOL" analyze \
  --ds4-bin "$DS4_BIN" \
  --model "$MODEL" \
  --oracle "$ORACLE_FILE" \
  --candidate "$CANDIDATE_OUT" \
  --log "$CANDIDATE_LOG"

echo "TOKEN27_RESULTS_DIR=$OUT_DIR"
