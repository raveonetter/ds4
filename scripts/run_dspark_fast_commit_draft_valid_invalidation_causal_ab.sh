#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-./quality-results/dspark-matrix/20260810-181808}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
CONFIDENCE=${CONFIDENCE:-0.80}
TOKENS=${TOKENS:-64}
EXPECTED_FIRST_DIFF_ORDINAL=${EXPECTED_FIRST_DIFF_ORDINAL:-27}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-fast-commit-draft-valid-invalidation-causal-ab}
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
E10_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_draft_verify_input_ab.py"
E11_COMPAT_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_scheduler_draft_valid_ab_compat.py"
E12_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_draft_valid_invalidation_causal_ab.py"

[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || { echo "not a git worktree: $ROOT_DIR" >&2; exit 2; }
for f in "$MODEL" "$DSPARK_MODEL" "$PROMPT_FILE" "$E10_TOOL" "$E11_COMPAT_TOOL" "$E12_TOOL"; do
  [[ -f "$f" ]] || { echo "missing required file $f" >&2; exit 2; }
done
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 2; }
[[ "$EXPECTED_FIRST_DIFF_ORDINAL" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_FIRST_DIFF_ORDINAL must be positive" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ds4-fast-draft-valid-invalidation.XXXXXX")
TRACE_WT="$TMP_ROOT/worktree"
WORKTREE_ADDED=0
cleanup() {
  if [[ "$WORKTREE_ADDED" == 1 ]]; then
    git -C "$ROOT_DIR" worktree remove --force "$TRACE_WT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

echo "FAST_COMMIT_DRAFT_VALID_INVALIDATION_SETUP source_run=$SOURCE_RUN_DIR confidence=$CONFIDENCE tokens=$TOKENS expected_ordinal1=$EXPECTED_FIRST_DIFF_ORDINAL"
git -C "$ROOT_DIR" worktree add --detach "$TRACE_WT" HEAD >/dev/null
WORKTREE_ADDED=1

python3 - "$E12_TOOL" <<'PY'
import ast
import pathlib
import sys
ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
PY
python3 "$E12_TOOL" instrument --source "$TRACE_WT/ds4.c"

grep -q 'DS4_DSPARK_E11_SCHED' "$TRACE_WT/ds4.c" || { echo "E11 scheduler instrumentation missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E11_SPEC_ENTER' "$TRACE_WT/ds4.c" || { echo "E11 spec-entry instrumentation missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E12_ENTRY' "$TRACE_WT/ds4.c" || { echo "E12 proposer-entry instrumentation missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E12_READY' "$TRACE_WT/ds4.c" || { echo "E12 proposer-ready instrumentation missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E12_CACHE' "$TRACE_WT/ds4.c" || { echo "E12 proposer-cache instrumentation missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E12_CHAIN' "$TRACE_WT/ds4.c" || { echo "E12 proposer-chain instrumentation missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E12_RESULT' "$TRACE_WT/ds4.c" || { echo "E12 proposer-result instrumentation missing" >&2; exit 2; }
echo "FAST_COMMIT_DRAFT_VALID_INVALIDATION_SOURCE_CHECK status=PASS"

MAKE_JOBS=${MAKE_JOBS:-4}
make -C "$TRACE_WT" -j "$MAKE_JOBS" ds4 >/dev/null
TRACE_BIN="$TRACE_WT/ds4"
[[ -x "$TRACE_BIN" ]] || { echo "E12 build did not produce $TRACE_BIN" >&2; exit 2; }
echo "FAST_COMMIT_DRAFT_VALID_INVALIDATION_BUILD_CHECK status=PASS"

PROMPT=$(cat "$PROMPT_FILE")

echo "FAST_COMMIT_DRAFT_VALID_INVALIDATION_RUN arm=SEQUENTIAL_ORACLE"
(
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
  unset DS4_DSPARK_E11_TRACE DS4_DSPARK_E12_TRACE DS4_DSPARK_PROBE
  unset DS4_METAL_Q8_DECODE_MPP
  "$TRACE_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink -p "$PROMPT"
) >"$OUT_DIR/SEQUENTIAL_ORACLE.txt" 2>"$OUT_DIR/SEQUENTIAL_ORACLE.log"

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
  local trace=$3
  local out="$OUT_DIR/${label}.txt"
  local log="$OUT_DIR/${label}.log"
  echo "FAST_COMMIT_DRAFT_VALID_INVALIDATION_RUN arm=$label fast_full_commit=$fast trace=$trace"
  (
    unset DS4_FAMILY_REPAIRS
    unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
    unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
    unset DS4_DSPARK_E11_TRACE DS4_DSPARK_E12_TRACE DS4_DSPARK_PROBE
    unset DS4_METAL_Q8_DECODE_MPP
    if [[ "$trace" == 1 ]]; then
      export DS4_DSPARK_E11_TRACE=1
      export DS4_DSPARK_E12_TRACE=1
    fi
    env DS4_DSPARK_STATS=1 \
        DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT="$fast" \
        "$TRACE_BIN" "${COMMON_ARGS[@]}"
  ) >"$out" 2>"$log"
}

run_arm E12_REPLAY_TRACE 0 1
run_arm E12_FAST_BASE 1 0
run_arm E12_FAST_TRACE 1 1

python3 "$E12_TOOL" analyze \
  --ds4 "$TRACE_BIN" \
  --model "$MODEL" \
  --oracle "$OUT_DIR/SEQUENTIAL_ORACLE.txt" \
  --replay-out "$OUT_DIR/E12_REPLAY_TRACE.txt" \
  --replay-log "$OUT_DIR/E12_REPLAY_TRACE.log" \
  --fast-base-out "$OUT_DIR/E12_FAST_BASE.txt" \
  --fast-base-log "$OUT_DIR/E12_FAST_BASE.log" \
  --fast-trace-out "$OUT_DIR/E12_FAST_TRACE.txt" \
  --fast-trace-log "$OUT_DIR/E12_FAST_TRACE.log" \
  --expected-ordinal1 "$EXPECTED_FIRST_DIFF_ORDINAL" \
  | tee "$OUT_DIR/draft-valid-invalidation-causal-ab-summary.log"

echo "FAST_COMMIT_DRAFT_VALID_INVALIDATION_RESULTS_DIR=$OUT_DIR"
