#!/usr/bin/env bash
set -euo pipefail

# E0-E2 causal localization for the warehouse first generated-token divergence.
# Production ds4.c is never modified. Commit tracing is injected only into a
# temporary detached worktree built from the current HEAD.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-./quality-results/dspark-matrix/20260810-181808}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
CONFIDENCE=${CONFIDENCE:-0.80}
TOKENS=${TOKENS:-64}
EXPECTED_FIRST_DIFF_ORDINAL=${EXPECTED_FIRST_DIFF_ORDINAL:-27}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-token27}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR="$OUT_ROOT/$RUN_ID"
TRACE_TOOL="$ROOT_DIR/scripts/dspark_token27_trace.py"

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

[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || { echo "not a git worktree: $ROOT_DIR" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model $MODEL" >&2; exit 2; }
[[ -f "$DSPARK_MODEL" ]] || { echo "missing DSpark support model $DSPARK_MODEL" >&2; exit 2; }
[[ -f "$PROMPT_FILE" ]] || { echo "missing warehouse prompt $PROMPT_FILE" >&2; exit 2; }
[[ -f "$TRACE_TOOL" ]] || { echo "missing trace tool $TRACE_TOOL" >&2; exit 2; }
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 2; }
[[ "$EXPECTED_FIRST_DIFF_ORDINAL" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_FIRST_DIFF_ORDINAL must be positive" >&2; exit 2; }

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

echo "TOKEN27_SETUP source_run=$SOURCE_RUN_DIR confidence=$CONFIDENCE tokens=$TOKENS expected_ordinal1=$EXPECTED_FIRST_DIFF_ORDINAL"
git -C "$ROOT_DIR" worktree add --detach "$TRACE_WT" HEAD >/dev/null
WORKTREE_ADDED=1
python3 "$TRACE_TOOL" instrument --source "$TRACE_WT/ds4.c"

MAKE_JOBS=${MAKE_JOBS:-4}
make -C "$TRACE_WT" -j "$MAKE_JOBS" ds4 >/dev/null
TRACE_BIN="$TRACE_WT/ds4"
[[ -x "$TRACE_BIN" ]] || { echo "trace build did not produce $TRACE_BIN" >&2; exit 2; }

if command -v strings >/dev/null 2>&1; then
  strings "$TRACE_BIN" 2>/dev/null | grep -F 'DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT' >/dev/null || {
    echo "trace binary does not contain full-accept fast-commit hook" >&2
    exit 2
  }
fi

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

clear_experiment_env() {
  unset DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT DS4_DSPARK_TRACE_COMMITS DS4_DSPARK_STATS
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
}

ORACLE_OUT="$OUT_DIR/sequential_oracle.txt"
ORACLE_LOG="$OUT_DIR/sequential_oracle.log"
echo "TOKEN27_RUN arm=SEQUENTIAL_ORACLE"
(
  clear_experiment_env
  "$TRACE_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink -p "$PROMPT"
) >"$ORACLE_OUT" 2>"$ORACLE_LOG"
echo "TOKEN27_ORACLE status=PASS source=fresh_same_binary horizon_tokens=$TOKENS"

run_arm() {
  local arm=$1 fast_commit=$2 trace=$3
  local out="$OUT_DIR/${arm}.txt"
  local log="$OUT_DIR/${arm}.log"
  echo "TOKEN27_RUN arm=$arm fast_full_commit=$fast_commit trace=$trace"
  (
    clear_experiment_env
    env DS4_DSPARK_STATS=1 \
        DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT="$fast_commit" \
        DS4_DSPARK_TRACE_COMMITS="$trace" \
        "$TRACE_BIN" "${COMMON_ARGS[@]}"
  ) >"$out" 2>"$log"
}

# E0: current full-accept no-replay behavior, tracing dormant.
run_arm E0_fast_trace_off 1 0

# E1: identical fast path, commit provenance enabled. E0 vs E1 is the
# instrumentation self-control and must retain the exact generated trajectory.
run_arm E1_fast_trace_on 1 1

# E2: only FULL_ACCEPT_FAST is disabled. The existing DSpark full-accept path
# restores the pre-verify frontier and replays accepted drafts through ordinary
# decode. Partial/correction handling and every other experiment knob are held.
run_arm E2_full_accept_replay 0 1

python3 "$TRACE_TOOL" analyze-ab \
  --ds4-bin "$TRACE_BIN" \
  --model "$MODEL" \
  --oracle "$ORACLE_OUT" \
  --e0-out "$OUT_DIR/E0_fast_trace_off.txt" \
  --e0-log "$OUT_DIR/E0_fast_trace_off.log" \
  --e1-out "$OUT_DIR/E1_fast_trace_on.txt" \
  --e1-log "$OUT_DIR/E1_fast_trace_on.log" \
  --e2-out "$OUT_DIR/E2_full_accept_replay.txt" \
  --e2-log "$OUT_DIR/E2_full_accept_replay.log" \
  --expected-ordinal1 "$EXPECTED_FIRST_DIFF_ORDINAL" | tee "$OUT_DIR/token27-summary.log"

echo "TOKEN27_RESULTS_DIR=$OUT_DIR"
