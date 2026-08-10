#!/usr/bin/env bash
set -euo pipefail

# Final DSpark quality/performance/speculation matrix.
#
# Baseline controls (once per prompt):
#   sequential       ordinary target-only decode; byte-identity oracle
#   dspark_strict    DSpark CLI forced target-only canonical decode
#
# For every confidence threshold:
#   replay           generic DSpark verifier + normal accepted-token replay
#   fast_generic     full-accept fast commit, all production repairs disabled
#   fast_repaired    same fast commit + REPAIRED_ENV production repairs
#
# fast_* is NOT fully replay-free: full accepts bypass replay; partial accepts
# still take the normal rollback/replay path. DS4_DSPARK_STATS=1 is enabled for
# every DSpark arm and parsed opportunistically; the raw stats line is retained.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

DS4_BIN=${DS4_BIN:-./ds4}
GENERIC_BIN=${GENERIC_BIN:-$DS4_BIN}
REPAIRED_BIN=${REPAIRED_BIN:-$DS4_BIN}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
TOKENS=${TOKENS:-768}
CONFIDENCES=${CONFIDENCES:-"0 0.3 0.5 0.7"}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-matrix}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR="$OUT_ROOT/$RUN_ID"

FAST_COMMIT_ENV=${FAST_COMMIT_ENV:-DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1}
FAST_COMMIT_SENTINEL=${FAST_COMMIT_SENTINEL:-DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT}
REPAIRED_ENV=${REPAIRED_ENV:-DS4_FAMILY1_REPAIR=ALL}
ALLOW_UNVERIFIED_FAST_COMMIT=${ALLOW_UNVERIFIED_FAST_COMMIT:-0}

COMMON_ARGS=(
  -m "$MODEL"
  --temp 0
  --tokens "$TOKENS"
  --nothink
)

mkdir -p "$OUT_DIR/prompts" "$OUT_DIR/warehouse" "$OUT_DIR/intervals"

cat > "$OUT_DIR/prompts/warehouse.txt" <<'PROMPT'
A warehouse has 1200 components.

At the start:
- 35% are type A.
- 25% are type B.
- The rest are type C.

Then the following operations happen in order:

1. 20% of the type A components are discarded.
2. 48 type B components are converted into type A components.
3. 15% of the remaining type C components are converted into type B components.
4. 10% of the current type A components are discarded.
5. 30 type C components are added.
6. Exactly one third of the current type B components are shipped.

Compute the final number of type A, type B, and type C components, and the final total.

Show every intermediate quantity and check that each conservation/change step is internally consistent. Do not approximate or round intermediate values.
PROMPT

cat > "$OUT_DIR/prompts/intervals.txt" <<'PROMPT'
Write a complete Python 3 function:

    merge_intervals(intervals)

Input is a list of [start, end] integer pairs.

Requirements:

1. Intervals are closed: [1, 3] includes both 1 and 3.
2. Normalize reversed intervals, so [5, 2] becomes [2, 5].
3. Merge intervals that overlap.
4. Also merge intervals that are directly adjacent in integer space:
   [1, 3] and [4, 7] must become [1, 7].
5. Preserve negative coordinates correctly.
6. Do not mutate the input.
7. Return intervals sorted by start, then end.
8. Empty input must return [].
9. Use no third-party libraries.

After implementing it, manually evaluate the exact returned value for:

[
    [5, 2],
    [-4, -2],
    [8, 10],
    [3, 4],
    [12, 12],
    [11, 11],
    [-1, 1],
    [20, 18],
    [17, 17]
]

Then provide at least 8 assert-based tests covering:
- empty input
- one interval
- reversed interval
- overlap
- adjacency
- negative values
- duplicate intervals
- multiple disconnected groups

Return only:
1. the implementation,
2. the manually evaluated result,
3. the assert statements.
PROMPT

require_file() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 2
  fi
}

require_file "$DS4_BIN"
require_file "$GENERIC_BIN"
require_file "$REPAIRED_BIN"
require_file "$MODEL"
require_file "$DSPARK_MODEL"

fast_commit_available() {
  local bin=$1
  if [[ "$ALLOW_UNVERIFIED_FAST_COMMIT" == "1" ]]; then
    return 0
  fi
  command -v strings >/dev/null 2>&1 || return 1
  # Do not use grep -q here under pipefail: an early grep exit SIGPIPEs strings
  # and makes a successful match look like a failed pipeline.
  strings "$bin" 2>/dev/null | grep -F "$FAST_COMMIT_SENTINEL" >/dev/null
}

clear_dspark_experiment_env() {
  unset DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT DS4_DSPARK_STATS
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
}

run_with_env_spec() {
  local env_spec=$1
  shift
  if [[ -n "$env_spec" ]]; then
    # env_spec is a whitespace-separated list of NAME=value assignments; the
    # runner's values intentionally contain no spaces.
    # shellcheck disable=SC2086
    env $env_spec "$@"
  else
    "$@"
  fi
}

extract_tps() {
  local log=$1
  local value
  value=$(grep -Eo '[0-9]+([.][0-9]+)?[[:space:]]*t/s' "$log" 2>/dev/null |
          tail -n 1 | sed -E 's/[[:space:]]*t\/s//' || true)
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf 'NA'
  fi
}

extract_stats_line() {
  local log=$1
  grep -F 'ds4: DSpark stats' "$log" 2>/dev/null | tail -n 1 || true
}

extract_stat_any() {
  local log=$1
  shift
  local line key value
  line=$(extract_stats_line "$log")
  [[ -n "$line" ]] || { printf 'NA'; return; }
  for key in "$@"; do
    value=$(printf '%s\n' "$line" | grep -Eo "${key}=[0-9]+([.][0-9]+)?" | tail -n 1 | cut -d= -f2 || true)
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return
    fi
  done
  printf 'NA'
}

calc_ratio() {
  local num=$1 den=$2
  if [[ "$num" == "NA" || "$den" == "NA" || "$den" == "0" ]]; then
    printf 'NA'
    return
  fi
  awk -v n="$num" -v d="$den" 'BEGIN { printf "%.4f", n/d }'
}

calc_speedup() {
  local tps=$1 base=$2
  if [[ "$tps" == "NA" || "$base" == "NA" || "$base" == "0" ]]; then
    printf 'NA'
    return
  fi
  awk -v x="$tps" -v b="$base" 'BEGIN { printf "%.4f", x/b }'
}

confidence_tag() {
  local conf=$1
  conf=${conf//./p}
  printf 'c%s' "$conf"
}

run_plain() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 prompt
  prompt=$(cat "$prompt_file")
  (
    clear_dspark_experiment_env
    "$bin" "${COMMON_ARGS[@]}" -p "$prompt"
  ) >"$stdout_file" 2>"$stderr_file"
}

run_dspark() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 conf=$5 extra_env=$6 strict=$7
  local prompt
  prompt=$(cat "$prompt_file")
  (
    clear_dspark_experiment_env
    if [[ "$strict" == "1" ]]; then
      run_with_env_spec "DS4_DSPARK_STATS=1 $extra_env" \
        "$bin" "${COMMON_ARGS[@]}" \
        --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$conf" --dspark-strict \
        -p "$prompt"
    else
      run_with_env_spec "DS4_DSPARK_STATS=1 $extra_env" \
        "$bin" "${COMMON_ARGS[@]}" \
        --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$conf" \
        -p "$prompt"
    fi
  ) >"$stdout_file" 2>"$stderr_file"
}

run_arm() {
  local prompt_name=$1 prompt_file=$2 arm=$3 conf=$4
  local dir="$OUT_DIR/$prompt_name"
  local out="$dir/$arm.txt"
  local log="$dir/$arm.log"
  local status_file="$dir/$arm.status"

  echo "[$prompt_name] $arm"
  case "$arm" in
    sequential)
      if run_plain "$DS4_BIN" "$prompt_file" "$out" "$log"; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    dspark_strict)
      if run_dspark "$DS4_BIN" "$prompt_file" "$out" "$log" "$conf" "" 1; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    replay_*)
      if run_dspark "$DS4_BIN" "$prompt_file" "$out" "$log" "$conf" "" 0; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    fast_generic_*)
      if ! fast_commit_available "$GENERIC_BIN"; then
        echo "SKIP: fast-commit hook '$FAST_COMMIT_SENTINEL' not found in $GENERIC_BIN" | tee "$log"
        : > "$out"; echo SKIP > "$status_file"; return 0
      fi
      if run_dspark "$GENERIC_BIN" "$prompt_file" "$out" "$log" "$conf" "$FAST_COMMIT_ENV" 0; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    fast_repaired_*)
      if ! fast_commit_available "$REPAIRED_BIN"; then
        echo "SKIP: fast-commit hook '$FAST_COMMIT_SENTINEL' not found in $REPAIRED_BIN" | tee "$log"
        : > "$out"; echo SKIP > "$status_file"; return 0
      fi
      if run_dspark "$REPAIRED_BIN" "$prompt_file" "$out" "$log" "$conf" "$FAST_COMMIT_ENV $REPAIRED_ENV" 0; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 2
      ;;
  esac
}

record_result() {
  local prompt_name=$1 arm=$2 conf=$3
  local dir="$OUT_DIR/$prompt_name"
  local out="$dir/$arm.txt"
  local log="$dir/$arm.log"
  local status ref identity first_byte tps base_tps speedup
  local drafted accepted accept_rate full partial zero replayed stats_line

  status=$(cat "$dir/$arm.status")
  ref="$dir/sequential.txt"
  identity=NA
  first_byte=NA

  if [[ "$arm" == "sequential" ]]; then
    identity=ORACLE
    first_byte=NONE
  elif [[ "$status" == "PASS" ]]; then
    if cmp -s "$ref" "$out"; then
      identity=EXACT
      first_byte=NONE
      : > "$dir/sequential_vs_${arm}.diff"
    else
      identity=DIFF
      first_byte=$(cmp -l "$ref" "$out" 2>/dev/null | head -n 1 | awk '{print $1}' || true)
      [[ -n "$first_byte" ]] || first_byte=EOF
      diff -u "$ref" "$out" > "$dir/sequential_vs_${arm}.diff" || true
    fi
  else
    identity=$status
  fi

  if [[ "$status" == "PASS" ]]; then
    tps=$(extract_tps "$log")
  else
    tps=NA
  fi
  base_tps=$(extract_tps "$dir/sequential.log")
  speedup=$(calc_speedup "$tps" "$base_tps")

  drafted=$(extract_stat_any "$log" drafted_tokens draft_tokens drafted)
  accepted=$(extract_stat_any "$log" accepted_draft_tokens accepted_tokens accepted)
  accept_rate=$(calc_ratio "$accepted" "$drafted")
  full=$(extract_stat_any "$log" full_accepts full_accept)
  partial=$(extract_stat_any "$log" partial_accepts partial_accept)
  zero=$(extract_stat_any "$log" zero_accepts zero_accept rejects)
  replayed=$(extract_stat_any "$log" replayed_tokens replay_tokens replayed)
  stats_line=$(extract_stats_line "$log")
  [[ -n "$stats_line" ]] || stats_line=NA
  stats_line=${stats_line//$'\t'/ }

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$prompt_name" "$arm" "$conf" "$identity" "$first_byte" >> "$OUT_DIR/quality.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$prompt_name" "$arm" "$conf" "$status" "$tps" "$speedup" >> "$OUT_DIR/performance.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$prompt_name" "$arm" "$conf" "$drafted" "$accepted" "$accept_rate" \
    "$full" "$partial" "$zero" "$replayed" >> "$OUT_DIR/speculation.tsv"
  printf '%s\t%s\t%s\t%s\n' \
    "$prompt_name" "$arm" "$conf" "$stats_line" >> "$OUT_DIR/stats_raw.tsv"
}

cat > "$OUT_DIR/config.txt" <<EOF_CONFIG
DS4_BIN=$DS4_BIN
GENERIC_BIN=$GENERIC_BIN
REPAIRED_BIN=$REPAIRED_BIN
MODEL=$MODEL
DSPARK_MODEL=$DSPARK_MODEL
TOKENS=$TOKENS
CONFIDENCES=$CONFIDENCES
FAST_COMMIT_ENV=$FAST_COMMIT_ENV
FAST_COMMIT_SENTINEL=$FAST_COMMIT_SENTINEL
REPAIRED_ENV=$REPAIRED_ENV
FAST_COMMIT_SCOPE=FULL_ACCEPT_ONLY_PARTIAL_ACCEPTS_REPLAY
BYTE_ORACLE=sequential
EOF_CONFIG

printf 'prompt\tarm\tconfidence\tidentity_vs_sequential\tfirst_diff_byte\n' > "$OUT_DIR/quality.tsv"
printf 'prompt\tarm\tconfidence\tstatus\tgeneration_tps\tspeedup_vs_sequential\n' > "$OUT_DIR/performance.tsv"
printf 'prompt\tarm\tconfidence\tdrafted\taccepted\tacceptance_rate\tfull_accepts\tpartial_accepts\tzero_accepts\treplayed_tokens\n' > "$OUT_DIR/speculation.tsv"
printf 'prompt\tarm\tconfidence\traw_dspark_stats\n' > "$OUT_DIR/stats_raw.tsv"

# Parse confidence list once. shellcheck disable=SC2206
CONF_LIST=($CONFIDENCES)
if [[ ${#CONF_LIST[@]} -eq 0 ]]; then
  echo "CONFIDENCES is empty" >&2
  exit 2
fi

for prompt_name in warehouse intervals; do
  prompt_file="$OUT_DIR/prompts/$prompt_name.txt"

  run_arm "$prompt_name" "$prompt_file" sequential NA
  record_result "$prompt_name" sequential NA

  # Strict is a target-only sanity control; one threshold is enough because the
  # speculative scheduler is not used for output generation in this arm.
  strict_conf=${CONF_LIST[${#CONF_LIST[@]}-1]}
  run_arm "$prompt_name" "$prompt_file" dspark_strict "$strict_conf"
  record_result "$prompt_name" dspark_strict "$strict_conf"

  for conf in "${CONF_LIST[@]}"; do
    tag=$(confidence_tag "$conf")
    for mode in replay fast_generic fast_repaired; do
      arm="${mode}_${tag}"
      run_arm "$prompt_name" "$prompt_file" "$arm" "$conf"
      record_result "$prompt_name" "$arm" "$conf"
    done
  done
done

echo
echo "QUALITY_MATRIX"
column -t -s $'\t' "$OUT_DIR/quality.tsv" 2>/dev/null || cat "$OUT_DIR/quality.tsv"

echo
echo "PERFORMANCE_MATRIX"
column -t -s $'\t' "$OUT_DIR/performance.tsv" 2>/dev/null || cat "$OUT_DIR/performance.tsv"

echo
echo "SPECULATION_MATRIX"
column -t -s $'\t' "$OUT_DIR/speculation.tsv" 2>/dev/null || cat "$OUT_DIR/speculation.tsv"

echo
echo "results: $OUT_DIR"
echo "Sequential is the byte oracle. speedup_vs_sequential > 1.0 is the end-to-end speedup gate."
echo "fast_* bypasses replay only on full accepts; partial accepts still replay."
echo "Raw DSpark stats are preserved in stats_raw.tsv even when a field name is not recognized by the parser."
