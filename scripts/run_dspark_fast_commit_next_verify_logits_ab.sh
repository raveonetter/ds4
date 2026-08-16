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
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-fast-commit-next-verify-logits-ab}
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
TRACE_TOOL="$ROOT_DIR/scripts/dspark_token27_trace.py"
E9_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_next_verify_logits_ab.py"
ADJ_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_next_verify_logits_adjudicate.py"

[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || { echo "not a git worktree: $ROOT_DIR" >&2; exit 2; }
for f in "$MODEL" "$DSPARK_MODEL" "$PROMPT_FILE" "$TRACE_TOOL" "$E9_TOOL" "$ADJ_TOOL"; do
  [[ -f "$f" ]] || { echo "missing required file $f" >&2; exit 2; }
done
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 2; }
[[ "$EXPECTED_FIRST_DIFF_ORDINAL" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_FIRST_DIFF_ORDINAL must be positive" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ds4-fast-next-verify-logits.XXXXXX")
TRACE_WT="$TMP_ROOT/worktree"
WORKTREE_ADDED=0
cleanup() {
  if [[ "$WORKTREE_ADDED" == 1 ]]; then
    git -C "$ROOT_DIR" worktree remove --force "$TRACE_WT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

echo "FAST_COMMIT_VERIFY_LOGITS_SETUP source_run=$SOURCE_RUN_DIR confidence=$CONFIDENCE tokens=$TOKENS expected_ordinal1=$EXPECTED_FIRST_DIFF_ORDINAL"
git -C "$ROOT_DIR" worktree add --detach "$TRACE_WT" HEAD >/dev/null
WORKTREE_ADDED=1

python3 "$TRACE_TOOL" instrument --source "$TRACE_WT/ds4.c"
python3 "$E9_TOOL" instrument --source "$TRACE_WT/ds4.c"

# E9.1 contract fix. metal_graph_verify_suffix_tops_impl initializes only
# row_tops[0..n_tokens-2]; the final spec-logits row is continuation state and
# has no row_tops entry. Never read that uninitialized automatic slot.
python3 - "$TRACE_WT/ds4.c" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = "row_tops ? row_tops[row] : -1"
new = "(row_tops && row + 1u < n_tokens) ? row_tops[row] : -1"
count = s.count(old)
if count != 2:
    raise SystemExit(f"E9.1 row-top contract patch expected 2 sites, found {count}")
p.write_text(s.replace(old, new), encoding="utf-8")
print("FAST_COMMIT_VERIFY_LOGITS_ROW_TOP_CONTRACT_PATCH sites=2 result=PASS")
PY

grep -q 'DS4_DSPARK_TRACE_RETURN' "$TRACE_WT/ds4.c" || { echo "commit trace instrumentation missing" >&2; exit 2; }
grep -q 'ds4_e9_verify_suffix_tops' "$TRACE_WT/ds4.c" || { echo "E9 verifier wrapper missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E9_LOGITS' "$TRACE_WT/ds4.c" || { echo "E9 logits sentinel missing" >&2; exit 2; }
grep -q 'row + 1u < n_tokens' "$TRACE_WT/ds4.c" || { echo "E9.1 row-top contract patch missing" >&2; exit 2; }
echo "FAST_COMMIT_VERIFY_LOGITS_SOURCE_CHECK status=PASS"

MAKE_JOBS=${MAKE_JOBS:-4}
make -C "$TRACE_WT" -j "$MAKE_JOBS" ds4 >/dev/null
TRACE_BIN="$TRACE_WT/ds4"
[[ -x "$TRACE_BIN" ]] || { echo "E9 build did not produce $TRACE_BIN" >&2; exit 2; }
echo "FAST_COMMIT_VERIFY_LOGITS_BUILD_CHECK status=PASS"

PROMPT=$(cat "$PROMPT_FILE")

echo "FAST_COMMIT_VERIFY_LOGITS_RUN arm=SEQUENTIAL_ORACLE"
(
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
  unset DS4_DSPARK_E9_TRACE_LOGITS DS4_DSPARK_TRACE_COMMITS
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
  echo "FAST_COMMIT_VERIFY_LOGITS_RUN arm=$label fast_full_commit=$fast trace=$trace"
  (
    unset DS4_FAMILY_REPAIRS
    unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
    unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
    unset DS4_DSPARK_E9_TRACE_LOGITS DS4_DSPARK_TRACE_COMMITS
    unset DS4_METAL_Q8_DECODE_MPP
    if [[ "$trace" == 1 ]]; then
      export DS4_DSPARK_E9_TRACE_LOGITS=1
      export DS4_DSPARK_TRACE_COMMITS=1
    fi
    env DS4_DSPARK_STATS=1 \
        DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT="$fast" \
        "$TRACE_BIN" "${COMMON_ARGS[@]}"
  ) >"$out" 2>"$log"
}

run_arm E9_REPLAY_TRACE 0 1
run_arm E9_FAST_BASE 1 0
run_arm E9_FAST_TRACE 1 1

python3 "$E9_TOOL" analyze \
  --ds4 "$TRACE_BIN" \
  --model "$MODEL" \
  --oracle "$OUT_DIR/SEQUENTIAL_ORACLE.txt" \
  --replay-out "$OUT_DIR/E9_REPLAY_TRACE.txt" \
  --replay-log "$OUT_DIR/E9_REPLAY_TRACE.log" \
  --fast-base-out "$OUT_DIR/E9_FAST_BASE.txt" \
  --fast-base-log "$OUT_DIR/E9_FAST_BASE.log" \
  --fast-trace-out "$OUT_DIR/E9_FAST_TRACE.txt" \
  --fast-trace-log "$OUT_DIR/E9_FAST_TRACE.log" \
  --expected-ordinal1 "$EXPECTED_FIRST_DIFF_ORDINAL" \
  | tee "$OUT_DIR/next-verify-logits-ab-summary.log"

# E9.1 source-contract adjudication is authoritative for row_tops validity and
# maps the rejection row as accepted_drafts-1.
python3 "$ADJ_TOOL" "$OUT_DIR" | tee "$OUT_DIR/next-verify-logits-e91-summary.log"

echo "FAST_COMMIT_VERIFY_LOGITS_RESULTS_DIR=$OUT_DIR"