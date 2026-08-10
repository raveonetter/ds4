#!/usr/bin/env bash
set -euo pipefail

# Fine-grained production search around the generic DSpark fast-commit path.
#
# This deliberately does NOT run family repairs.  It reuses the exact prompt
# files from an existing quality-matrix run and searches only:
#
#   configuration = fast_generic + dspark confidence
#
# Each configuration is repeated so generation throughput can be aggregated by
# median and token trajectory stability can be checked across repeated runs.
# The sequential oracle is repeated as well.
#
# Required:
#   SOURCE_RUN_DIR=quality-results/dspark-matrix/<run-id>
#
# Optional:
#   CONFIDENCES="0.50 0.55 0.60 0.65 0.70 0.75 0.80"
#   REPEATS=5
#   OUT_ROOT=quality-results/dspark-confidence
#   RUN_ID=...
#   DS4_BIN=./ds4
#   GENERIC_BIN=./ds4
#   MODEL=./ds4flash.gguf
#   DSPARK_MODEL=./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
#   TOKENS=768

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR to an existing quality-matrix run}
DS4_BIN=${DS4_BIN:-./ds4}
GENERIC_BIN=${GENERIC_BIN:-$DS4_BIN}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
CONFIDENCES=${CONFIDENCES:-"0.50 0.55 0.60 0.65 0.70 0.75 0.80"}
REPEATS=${REPEATS:-5}
OUT_ROOT=${OUT_ROOT:-./quality-results/dspark-confidence}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR="$OUT_ROOT/$RUN_ID"
FAST_COMMIT_SENTINEL=${FAST_COMMIT_SENTINEL:-DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT}
ALLOW_UNVERIFIED_FAST_COMMIT=${ALLOW_UNVERIFIED_FAST_COMMIT:-0}

if [[ -z "${TOKENS:-}" && -f "$SOURCE_RUN_DIR/config.txt" ]]; then
  TOKENS=$(awk -F= '$1=="TOKENS" {print $2; exit}' "$SOURCE_RUN_DIR/config.txt")
fi
TOKENS=${TOKENS:-768}

[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be a positive integer" >&2; exit 2; }
[[ -x "$DS4_BIN" ]] || { echo "missing executable $DS4_BIN" >&2; exit 2; }
[[ -x "$GENERIC_BIN" ]] || { echo "missing executable $GENERIC_BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model $MODEL" >&2; exit 2; }
[[ -f "$DSPARK_MODEL" ]] || { echo "missing DSpark support model $DSPARK_MODEL" >&2; exit 2; }
[[ -d "$SOURCE_RUN_DIR/prompts" ]] || { echo "missing prompts directory $SOURCE_RUN_DIR/prompts" >&2; exit 2; }

PROMPT_FILES=("$SOURCE_RUN_DIR"/prompts/*.txt)
[[ -e "${PROMPT_FILES[0]}" ]] || { echo "no prompt files under $SOURCE_RUN_DIR/prompts" >&2; exit 2; }

fast_commit_available() {
  local bin=$1
  if [[ "$ALLOW_UNVERIFIED_FAST_COMMIT" == "1" ]]; then
    return 0
  fi
  command -v strings >/dev/null 2>&1 || return 1
  strings "$bin" 2>/dev/null | grep -F "$FAST_COMMIT_SENTINEL" >/dev/null
}

if ! fast_commit_available "$GENERIC_BIN"; then
  echo "fast-commit hook '$FAST_COMMIT_SENTINEL' not found in $GENERIC_BIN" >&2
  exit 2
fi

clear_experiment_env() {
  unset DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT DS4_DSPARK_STATS
  unset DS4_FAMILY_REPAIRS
  unset DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
  unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
}

extract_tps() {
  local log=$1
  local value
  value=$(grep -Eo '[0-9]+([.][0-9]+)?[[:space:]]*t/s' "$log" 2>/dev/null |
          tail -n 1 | sed -E 's/[[:space:]]*t\/s//' || true)
  [[ -n "$value" ]] && printf '%s' "$value" || printf 'NA'
}

confidence_tag() {
  local conf=$1
  conf=${conf//./p}
  printf 'c%s' "$conf"
}

record_sample() {
  local prompt=$1 arm=$2 conf=$3 repeat=$4 status=$5 tps=$6 out=$7 log=$8
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$prompt" "$arm" "$conf" "$repeat" "$status" "$tps" "$out" "$log" >> "$OUT_DIR/samples.tsv"
}

run_sequential() {
  local prompt_name=$1 prompt_file=$2 repeat=$3
  local dir="$OUT_DIR/$prompt_name/sequential"
  local out="$dir/r${repeat}.txt"
  local log="$dir/r${repeat}.log"
  local status tps prompt
  mkdir -p "$dir"
  prompt=$(cat "$prompt_file")
  echo "[$prompt_name] sequential repeat=$repeat"
  if (
    clear_experiment_env
    "$DS4_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink -p "$prompt"
  ) >"$out" 2>"$log"; then
    status=PASS
    tps=$(extract_tps "$log")
  else
    status=FAIL
    tps=NA
  fi
  record_sample "$prompt_name" sequential NA "$repeat" "$status" "$tps" "$out" "$log"
}

run_generic() {
  local prompt_name=$1 prompt_file=$2 conf=$3 repeat=$4
  local tag arm dir out log status tps prompt
  tag=$(confidence_tag "$conf")
  arm="fast_generic_${tag}"
  dir="$OUT_DIR/$prompt_name/$arm"
  out="$dir/r${repeat}.txt"
  log="$dir/r${repeat}.log"
  mkdir -p "$dir"
  prompt=$(cat "$prompt_file")
  echo "[$prompt_name] $arm repeat=$repeat"
  if (
    clear_experiment_env
    env DS4_DSPARK_STATS=1 DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1 \
      "$GENERIC_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink \
      --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$conf" -p "$prompt"
  ) >"$out" 2>"$log"; then
    status=PASS
    tps=$(extract_tps "$log")
  else
    status=FAIL
    tps=NA
  fi
  record_sample "$prompt_name" "$arm" "$conf" "$repeat" "$status" "$tps" "$out" "$log"
}

mkdir -p "$OUT_DIR"
printf 'prompt\tarm\tconfidence\trepeat\tstatus\tgeneration_tps\toutput_file\tlog_file\n' > "$OUT_DIR/samples.tsv"

cat > "$OUT_DIR/config.txt" <<EOF_CONFIG
SOURCE_RUN_DIR=$SOURCE_RUN_DIR
DS4_BIN=$DS4_BIN
GENERIC_BIN=$GENERIC_BIN
MODEL=$MODEL
DSPARK_MODEL=$DSPARK_MODEL
TOKENS=$TOKENS
CONFIDENCES=$CONFIDENCES
REPEATS=$REPEATS
FAST_COMMIT=DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1
FAMILY_REPAIRS=DISABLED
EOF_CONFIG

for repeat in $(seq 1 "$REPEATS"); do
  for prompt_file in "${PROMPT_FILES[@]}"; do
    prompt_name=$(basename "$prompt_file" .txt)
    run_sequential "$prompt_name" "$prompt_file" "$repeat"
    for conf in $CONFIDENCES; do
      run_generic "$prompt_name" "$prompt_file" "$conf" "$repeat"
    done
  done
done

python3 ./scripts/dspark_confidence_pareto.py \
  --run-dir "$OUT_DIR" \
  --ds4-bin "$DS4_BIN" \
  --model "$MODEL" \
  --expected-repeats "$REPEATS"

echo
echo "CONFIDENCE_SEARCH_DIR=$OUT_DIR"
echo "CONFIDENCE_SAMPLES=$OUT_DIR/samples.tsv"
echo "CONFIDENCE_ROBUST_PARETO=$OUT_DIR/robust-pareto.tsv"
