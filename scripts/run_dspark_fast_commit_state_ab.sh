#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-./quality-results/dspark-matrix/20260810-181808}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
CONFIDENCE=${CONFIDENCE:-0.80}
TOKENS=${TOKENS:-64}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-fast-commit-state-ab}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR="$OUT_ROOT/$RUN_ID"

abs_path() {
  python3 - "$1" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

SOURCE_RUN_DIR=$(abs_path "$SOURCE_RUN_DIR")
MODEL=$(abs_path "$MODEL")
DSPARK_MODEL=$(abs_path "$DSPARK_MODEL")
OUT_DIR=$(abs_path "$OUT_DIR")
PROMPT_FILE="$SOURCE_RUN_DIR/prompts/warehouse.txt"
STATE_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_state_ab.py"

[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || { echo "not a git worktree: $ROOT_DIR" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model $MODEL" >&2; exit 2; }
[[ -f "$DSPARK_MODEL" ]] || { echo "missing DSpark support model $DSPARK_MODEL" >&2; exit 2; }
[[ -f "$PROMPT_FILE" ]] || { echo "missing warehouse prompt $PROMPT_FILE" >&2; exit 2; }
[[ -f "$STATE_TOOL" ]] || { echo "missing state tool $STATE_TOOL" >&2; exit 2; }
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ds4-fast-state.XXXXXX")
TRACE_WT="$TMP_ROOT/worktree"
WORKTREE_ADDED=0
cleanup() {
  if [[ "$WORKTREE_ADDED" == 1 ]]; then
    git -C "$ROOT_DIR" worktree remove --force "$TRACE_WT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

echo "FAST_COMMIT_STATE_SETUP source_run=$SOURCE_RUN_DIR confidence=$CONFIDENCE tokens=$TOKENS"
git -C "$ROOT_DIR" worktree add --detach "$TRACE_WT" HEAD >/dev/null
WORKTREE_ADDED=1
python3 "$STATE_TOOL" instrument --source "$TRACE_WT/ds4.c"

grep -q 'DS4_DSPARK_STATE_RETURN' "$TRACE_WT/ds4.c" || { echo "state trace source sentinel missing after instrumentation" >&2; exit 2; }
grep -q 'DS4_DSPARK_TRACE_STATE_AB' "$TRACE_WT/ds4.c" || { echo "state trace env gate missing after instrumentation" >&2; exit 2; }
echo "FAST_COMMIT_STATE_SOURCE_CHECK status=PASS"

MAKE_JOBS=${MAKE_JOBS:-4}
make -C "$TRACE_WT" -j "$MAKE_JOBS" ds4 >/dev/null
TRACE_BIN="$TRACE_WT/ds4"
[[ -x "$TRACE_BIN" ]] || { echo "state trace build did not produce $TRACE_BIN" >&2; exit 2; }
echo "FAST_COMMIT_STATE_BUILD_CHECK status=PASS"

PROMPT=$(cat "$PROMPT_FILE")
COMMON_ARGS=(
  -m "$MODEL"
  --temp 0
  --tokens "$TOKENS"
  --nothink
  --mtp "$DSPARK_MODEL"
  --dspark
  --dspark-confidence "$CONFIDENCE"
  -p "$PROMPT"
)

run_arm() {
  local label=$1
  local fast=$2
  local out="$OUT_DIR/${label}.txt"
  local log="$OUT_DIR/${label}.log"
  echo "FAST_COMMIT_STATE_RUN arm=$label fast_full_commit=$fast"
  (
    unset DS4_FAMILY_REPAIRS
    unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
    unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
    env DS4_DSPARK_STATS=1 \
        DS4_DSPARK_TRACE_STATE_AB=1 \
        DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT="$fast" \
        "$TRACE_BIN" "${COMMON_ARGS[@]}"
  ) >"$out" 2>"$log"
}

run_arm E3_FAST 1
run_arm E3_REPLAY 0

python3 "$STATE_TOOL" analyze \
  --fast-log "$OUT_DIR/E3_FAST.log" \
  --replay-log "$OUT_DIR/E3_REPLAY.log" | tee "$OUT_DIR/state-ab-summary.log"

echo "FAST_COMMIT_STATE_RESULTS_DIR=$OUT_DIR"
