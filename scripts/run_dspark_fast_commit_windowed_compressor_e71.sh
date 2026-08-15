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
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-fast-commit-windowed-compressor-e71}
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
E71_TOOL="$ROOT_DIR/scripts/dspark_fast_commit_windowed_compressor_e71.py"

[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || { echo "not a git worktree: $ROOT_DIR" >&2; exit 2; }
for f in "$MODEL" "$DSPARK_MODEL" "$PROMPT_FILE" "$TRACE_TOOL" "$E71_TOOL"; do
  [[ -f "$f" ]] || { echo "missing required file $f" >&2; exit 2; }
done
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 2; }
[[ "$EXPECTED_FIRST_DIFF_ORDINAL" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_FIRST_DIFF_ORDINAL must be positive" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ds4-fast-window-comp-e71.XXXXXX")
TRACE_WT="$TMP_ROOT/worktree"
WORKTREE_ADDED=0
cleanup() {
  if [[ "$WORKTREE_ADDED" == 1 ]]; then
    git -C "$ROOT_DIR" worktree remove --force "$TRACE_WT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_SETUP source_run=$SOURCE_RUN_DIR confidence=$CONFIDENCE tokens=$TOKENS expected_ordinal1=$EXPECTED_FIRST_DIFF_ORDINAL"
git -C "$ROOT_DIR" worktree add --detach "$TRACE_WT" HEAD >/dev/null
WORKTREE_ADDED=1
python3 "$TRACE_TOOL" instrument --source "$TRACE_WT/ds4.c"
python3 "$E71_TOOL" instrument --source "$TRACE_WT/ds4.c"

grep -q 'DS4_DSPARK_TRACE_RETURN' "$TRACE_WT/ds4.c" || { echo "commit trace instrumentation missing" >&2; exit 2; }
grep -q 'ds4_e71_window_match' "$TRACE_WT/ds4.c" || { echo "E7.1 split matcher missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E71_REFRESH_BATCH' "$TRACE_WT/ds4.c" || { echo "E7.1 refresh scout missing" >&2; exit 2; }
grep -q 'DS4_DSPARK_E71_REFRESH_TARGET_POS0' "$TRACE_WT/ds4.c" || { echo "E7.1 refresh target gate missing" >&2; exit 2; }
echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_SOURCE_CHECK status=PASS"

MAKE_JOBS=${MAKE_JOBS:-4}
make -C "$TRACE_WT" -j "$MAKE_JOBS" ds4 >/dev/null
TRACE_BIN="$TRACE_WT/ds4"
[[ -x "$TRACE_BIN" ]] || { echo "E7.1 build did not produce $TRACE_BIN" >&2; exit 2; }
echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_BUILD_CHECK status=PASS"

PROMPT=$(cat "$PROMPT_FILE")

echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_RUN arm=SEQUENTIAL_ORACLE"
(
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
  unset DS4_DSPARK_E7_COMPRESSOR_REPAIR DS4_DSPARK_E7_TARGET_POS0 DS4_DSPARK_E7_TARGET_N_TOKENS
  unset DS4_DSPARK_E71_REFRESH_TARGET_POS0 DS4_DSPARK_E71_REFRESH_TARGET_N_TOKENS
  unset DS4_DSPARK_E7_TRACE DS4_DSPARK_TRACE_COMMITS DS4_METAL_Q8_DECODE_MPP
  "$TRACE_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink -p "$PROMPT"
) >"$OUT_DIR/SEQUENTIAL_ORACLE.txt" 2>"$OUT_DIR/SEQUENTIAL_ORACLE.log"

COMMON_ARGS=(
  -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink
  --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$CONFIDENCE" -p "$PROMPT"
)

run_arm() {
  local label=$1 fast=$2 trace_commits=$3 e7_trace=$4 mode=${5:-}
  local p_pos=${6:-} p_n=${7:-} r_pos=${8:-} r_n=${9:-}
  local out="$OUT_DIR/${label}.txt" log="$OUT_DIR/${label}.log"
  echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_RUN arm=$label fast_full_commit=$fast mode=${mode:-NONE} projection_target=${p_pos:-NONE},${p_n:-NONE} refresh_target=${r_pos:-NONE},${r_n:-NONE}"
  (
    unset DS4_FAMILY_REPAIRS
    unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
    unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
    unset DS4_DSPARK_E7_COMPRESSOR_REPAIR DS4_DSPARK_E7_TARGET_POS0 DS4_DSPARK_E7_TARGET_N_TOKENS
    unset DS4_DSPARK_E71_REFRESH_TARGET_POS0 DS4_DSPARK_E71_REFRESH_TARGET_N_TOKENS
    unset DS4_DSPARK_E7_TRACE DS4_DSPARK_TRACE_COMMITS DS4_METAL_Q8_DECODE_MPP
    [[ -n "$mode" ]] && export DS4_DSPARK_E7_COMPRESSOR_REPAIR="$mode"
    if [[ -n "$p_pos" ]]; then
      export DS4_DSPARK_E7_TARGET_POS0="$p_pos"
      export DS4_DSPARK_E7_TARGET_N_TOKENS="$p_n"
    fi
    if [[ -n "$r_pos" ]]; then
      export DS4_DSPARK_E71_REFRESH_TARGET_POS0="$r_pos"
      export DS4_DSPARK_E71_REFRESH_TARGET_N_TOKENS="$r_n"
    fi
    [[ "$trace_commits" == 1 ]] && export DS4_DSPARK_TRACE_COMMITS=1
    [[ "$e7_trace" == 1 ]] && export DS4_DSPARK_E7_TRACE=1
    env DS4_DSPARK_STATS=1 DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT="$fast" "$TRACE_BIN" "${COMMON_ARGS[@]}"
  ) >"$out" 2>"$log"
}

run_arm E71_REPLAY 0 0 0
run_arm E71_FAST_BASE 1 0 0
run_arm E71_FAST_SCOUT 1 1 1

WINDOW_JSON="$OUT_DIR/e71-window.json"
python3 "$E71_TOOL" select-window --log "$OUT_DIR/E71_FAST_SCOUT.log" --output-json "$WINDOW_JSON" \
  | tee "$OUT_DIR/e71-window-selection.log"

read -r P_POS P_N R_STATUS R_POS R_N < <(
  python3 - "$WINDOW_JSON" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], "r", encoding="utf-8"))
rp = "" if x["refresh_pos0"] is None else str(x["refresh_pos0"])
rn = "" if x["refresh_n_tokens"] is None else str(x["refresh_n_tokens"])
print(x["projection_pos0"], x["projection_n_tokens"], x["refresh_status"], rp or "NONE", rn or "NONE")
PY
)
[[ "$P_POS" =~ ^[0-9]+$ && "$P_N" =~ ^[1-9][0-9]*$ ]] || { echo "invalid projection target" >&2; exit 2; }
if [[ "$R_STATUS" == PASS ]]; then
  [[ "$R_POS" =~ ^[0-9]+$ && "$R_N" =~ ^[1-9][0-9]*$ ]] || { echo "invalid refresh target" >&2; exit 2; }
else
  R_POS=""; R_N=""
fi
echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_TARGET projection_pos0=$P_POS projection_n_tokens=$P_N refresh_status=$R_STATUS refresh_pos0=${R_POS:-NONE} refresh_n_tokens=${R_N:-NONE} status=PASS"

run_arm E71_FAST_PROJECTION 1 0 1 PROJECTION "$P_POS" "$P_N" "$R_POS" "$R_N"
run_arm E71_FAST_REFRESH 1 0 1 REFRESH "$P_POS" "$P_N" "$R_POS" "$R_N"
run_arm E71_FAST_BOTH 1 0 1 BOTH "$P_POS" "$P_N" "$R_POS" "$R_N"

python3 "$E71_TOOL" analyze \
  --ds4 "$TRACE_BIN" --model "$MODEL" --oracle "$OUT_DIR/SEQUENTIAL_ORACLE.txt" --window-json "$WINDOW_JSON" \
  --replay-out "$OUT_DIR/E71_REPLAY.txt" --replay-log "$OUT_DIR/E71_REPLAY.log" \
  --fast-out "$OUT_DIR/E71_FAST_BASE.txt" --fast-log "$OUT_DIR/E71_FAST_BASE.log" \
  --scout-out "$OUT_DIR/E71_FAST_SCOUT.txt" --scout-log "$OUT_DIR/E71_FAST_SCOUT.log" \
  --projection-out "$OUT_DIR/E71_FAST_PROJECTION.txt" --projection-log "$OUT_DIR/E71_FAST_PROJECTION.log" \
  --refresh-out "$OUT_DIR/E71_FAST_REFRESH.txt" --refresh-log "$OUT_DIR/E71_FAST_REFRESH.log" \
  --both-out "$OUT_DIR/E71_FAST_BOTH.txt" --both-log "$OUT_DIR/E71_FAST_BOTH.log" \
  --expected-ordinal1 "$EXPECTED_FIRST_DIFF_ORDINAL" | tee "$OUT_DIR/windowed-compressor-e71-summary.log"

echo "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_RESULTS_DIR=$OUT_DIR"
