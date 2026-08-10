#!/usr/bin/env bash
set -euo pipefail

# Fixed DSpark correctness/performance matrix for two deterministic prompts.
# Keep correctness oracles, replay baseline, and approximate full-accept
# fast-commit controls separate:
#
#   sequential               ordinary target-only decode
#   dspark_strict            DSpark CLI forced target-only canonical decode
#   dspark_replay            generic DSpark verifier + normal accepted-token replay
#   fast_commit_generic      PR #746 full-accept fast commit, repairs disabled
#   fast_commit_repaired     same fast commit + explicitly enabled production repairs
#
# IMPORTANT: fast_commit_* is NOT a fully no-replay mode. Only full accepts
# bypass rollback/replay; partial accepts retain the normal replay path. These
# arms are free-running drift/performance controls for production family repair.
#
# This script never treats the canonical oracle as a repair implementation.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

DS4_BIN=${DS4_BIN:-./ds4}
GENERIC_BIN=${GENERIC_BIN:-$DS4_BIN}
REPAIRED_BIN=${REPAIRED_BIN:-$DS4_BIN}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
TOKENS=${TOKENS:-768}
DSPARK_CONFIDENCE=${DSPARK_CONFIDENCE:-0}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-matrix}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR="$OUT_ROOT/$RUN_ID"

# Upstream PR #746 opt-in full-accept fast commit. Override these only when a
# branch deliberately uses another equivalent benchmark hook.
FAST_COMMIT_ENV=${FAST_COMMIT_ENV:-DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1}
FAST_COMMIT_SENTINEL=${FAST_COMMIT_SENTINEL:-DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT}

# Only real production repairs belong here. Do not put exact-row oracle
# fallbacks in the default. Family 1 is the first graduated production repair.
REPAIRED_ENV=${REPAIRED_ENV:-DS4_FAMILY1_REPAIR=ALL}

# Set only when a replacement fast-commit hook cannot be found by strings(1).
# Otherwise fast-commit arms SKIP rather than silently degenerating to replay.
ALLOW_UNVERIFIED_FAST_COMMIT=${ALLOW_UNVERIFIED_FAST_COMMIT:-0}

COMMON_ARGS=(
  -m "$MODEL"
  --temp 0
  --tokens "$TOKENS"
  --nothink
)
DSPARK_ARGS=(
  --mtp "$DSPARK_MODEL"
  --dspark
  --dspark-confidence "$DSPARK_CONFIDENCE"
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

  # Do not use grep -q here. With set -o pipefail, grep -q exits as soon as it
  # finds the sentinel, strings(1) can then receive SIGPIPE, and the pipeline
  # is reported as failed even though the string was present. Read the full
  # strings output instead so the detector is reliable under pipefail.
  if command -v strings >/dev/null 2>&1 &&
     strings "$bin" 2>/dev/null | grep -F "$FAST_COMMIT_SENTINEL" >/dev/null; then
    return 0
  fi
  return 1
}

extract_tps() {
  local log=$1
  local value

  value=$(grep -Eo '[0-9]+([.][0-9]+)?[[:space:]]*t/s' "$log" 2>/dev/null |
          tail -n 1 | sed -E 's/[[:space:]]*t\/s//' || true)
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
    return
  fi

  value=$(grep -Ei 'generation|decode' "$log" 2>/dev/null |
          grep -Eo '[0-9]+([.][0-9]+)?' | tail -n 1 || true)
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf 'NA'
  fi
}

run_plain() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 prompt
  prompt=$(cat "$prompt_file")
  (
    unset DS4_FAMILY_REPAIRS DS4_FAMILY1_REPAIR DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT
    "$bin" "${COMMON_ARGS[@]}" -p "$prompt"
  ) >"$stdout_file" 2>"$stderr_file"
}

run_dspark_strict() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 prompt
  prompt=$(cat "$prompt_file")
  (
    unset DS4_FAMILY_REPAIRS DS4_FAMILY1_REPAIR DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT
    "$bin" "${COMMON_ARGS[@]}" "${DSPARK_ARGS[@]}" --dspark-strict -p "$prompt"
  ) >"$stdout_file" 2>"$stderr_file"
}

run_dspark_replay() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 prompt
  prompt=$(cat "$prompt_file")
  (
    unset DS4_FAMILY_REPAIRS DS4_FAMILY1_REPAIR DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT
    "$bin" "${COMMON_ARGS[@]}" "${DSPARK_ARGS[@]}" -p "$prompt"
  ) >"$stdout_file" 2>"$stderr_file"
}

run_with_env_spec() {
  local env_spec=$1
  shift
  # Intentional splitting: env_spec is NAME=value assignments without spaces
  # inside individual values.
  # shellcheck disable=SC2086
  env $env_spec "$@"
}

run_fast_commit_generic() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 prompt
  prompt=$(cat "$prompt_file")
  (
    unset DS4_FAMILY_REPAIRS DS4_FAMILY1_REPAIR DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT
    run_with_env_spec "$FAST_COMMIT_ENV" \
      "$bin" "${COMMON_ARGS[@]}" "${DSPARK_ARGS[@]}" -p "$prompt"
  ) >"$stdout_file" 2>"$stderr_file"
}

run_fast_commit_repaired() {
  local bin=$1 prompt_file=$2 stdout_file=$3 stderr_file=$4 prompt combined_env
  prompt=$(cat "$prompt_file")
  combined_env="$FAST_COMMIT_ENV $REPAIRED_ENV"
  (
    unset DS4_FAMILY_REPAIRS DS4_FAMILY1_REPAIR DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT
    run_with_env_spec "$combined_env" \
      "$bin" "${COMMON_ARGS[@]}" "${DSPARK_ARGS[@]}" -p "$prompt"
  ) >"$stdout_file" 2>"$stderr_file"
}

run_arm() {
  local prompt_name=$1 prompt_file=$2 arm=$3
  local dir="$OUT_DIR/$prompt_name"
  local out="$dir/$arm.txt" log="$dir/$arm.log" status_file="$dir/$arm.status"

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
      if run_dspark_strict "$DS4_BIN" "$prompt_file" "$out" "$log"; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    dspark_replay)
      if run_dspark_replay "$DS4_BIN" "$prompt_file" "$out" "$log"; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    fast_commit_generic)
      if ! fast_commit_available "$GENERIC_BIN"; then
        echo "SKIP: fast-commit hook '$FAST_COMMIT_SENTINEL' not found in $GENERIC_BIN" | tee "$log"
        : > "$out"; echo SKIP > "$status_file"; return 0
      fi
      if run_fast_commit_generic "$GENERIC_BIN" "$prompt_file" "$out" "$log"; then
        echo PASS > "$status_file"
      else
        echo FAIL > "$status_file"; return 1
      fi
      ;;
    fast_commit_repaired)
      if ! fast_commit_available "$REPAIRED_BIN"; then
        echo "SKIP: fast-commit hook '$FAST_COMMIT_SENTINEL' not found in $REPAIRED_BIN" | tee "$log"
        : > "$out"; echo SKIP > "$status_file"; return 0
      fi
      if run_fast_commit_repaired "$REPAIRED_BIN" "$prompt_file" "$out" "$log"; then
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

compare_to_sequential() {
  local prompt_name=$1
  local arm=$2
  local dir="$OUT_DIR/$prompt_name"
  local ref="$dir/sequential.txt"
  local out="$dir/$arm.txt"
  local status first_byte
  status=$(cat "$dir/$arm.status")

  if [[ "$status" != "PASS" ]]; then
    printf '%s\t%s\t%s\t%s\n' "$prompt_name" "$arm" "$status" "NA" >> "$OUT_DIR/comparisons.tsv"
    return
  fi

  if cmp -s "$ref" "$out"; then
    printf '%s\t%s\tEXACT\tNONE\n' "$prompt_name" "$arm" >> "$OUT_DIR/comparisons.tsv"
    : > "$dir/sequential_vs_${arm}.diff"
  else
    first_byte=$(cmp -l "$ref" "$out" 2>/dev/null | head -n 1 | awk '{print $1}' || true)
    [[ -n "$first_byte" ]] || first_byte="EOF"
    printf '%s\t%s\tDIFF\t%s\n' "$prompt_name" "$arm" "$first_byte" >> "$OUT_DIR/comparisons.tsv"
    diff -u "$ref" "$out" > "$dir/sequential_vs_${arm}.diff" || true
  fi
}

record_perf() {
  local prompt_name=$1
  local arm=$2
  local dir="$OUT_DIR/$prompt_name"
  local status tps
  status=$(cat "$dir/$arm.status")
  if [[ "$status" == "PASS" ]]; then
    tps=$(extract_tps "$dir/$arm.log")
  else
    tps=NA
  fi
  printf '%s\t%s\t%s\t%s\n' "$prompt_name" "$arm" "$status" "$tps" >> "$OUT_DIR/performance.tsv"
}

cat > "$OUT_DIR/config.txt" <<EOF_CONFIG
DS4_BIN=$DS4_BIN
GENERIC_BIN=$GENERIC_BIN
REPAIRED_BIN=$REPAIRED_BIN
MODEL=$MODEL
DSPARK_MODEL=$DSPARK_MODEL
TOKENS=$TOKENS
DSPARK_CONFIDENCE=$DSPARK_CONFIDENCE
FAST_COMMIT_ENV=$FAST_COMMIT_ENV
FAST_COMMIT_SENTINEL=$FAST_COMMIT_SENTINEL
REPAIRED_ENV=$REPAIRED_ENV
FAST_COMMIT_SCOPE=FULL_ACCEPT_ONLY_PARTIAL_ACCEPTS_REPLAY
EOF_CONFIG

printf 'prompt\tarm\tidentity_vs_sequential\tfirst_diff_byte\n' > "$OUT_DIR/comparisons.tsv"
printf 'prompt\tarm\tstatus\tgeneration_tps\n' > "$OUT_DIR/performance.tsv"

ARMS=(sequential dspark_strict dspark_replay fast_commit_generic fast_commit_repaired)

for prompt_name in warehouse intervals; do
  prompt_file="$OUT_DIR/prompts/$prompt_name.txt"
  for arm in "${ARMS[@]}"; do
    run_arm "$prompt_name" "$prompt_file" "$arm"
    record_perf "$prompt_name" "$arm"
  done

  for arm in dspark_strict dspark_replay fast_commit_generic fast_commit_repaired; do
    compare_to_sequential "$prompt_name" "$arm"
  done
done

echo
echo "QUALITY_MATRIX"
column -t -s $'\t' "$OUT_DIR/comparisons.tsv" 2>/dev/null || cat "$OUT_DIR/comparisons.tsv"

echo
echo "PERFORMANCE_MATRIX"
column -t -s $'\t' "$OUT_DIR/performance.tsv" 2>/dev/null || cat "$OUT_DIR/performance.tsv"

echo
echo "results: $OUT_DIR"
echo "fast_commit_* bypasses replay only on full accepts; partial accepts still replay."
echo "DIFF in fast_commit_* is experimental data, not a script failure; command failures still fail the run."
