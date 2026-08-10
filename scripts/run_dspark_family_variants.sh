#!/usr/bin/env bash
set -euo pipefail

# Append additional fast-commit family configurations to an existing
# run_dspark_quality_matrix.sh result directory. Prompts are reused verbatim.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

RUN_DIR=${RUN_DIR:?set RUN_DIR to an existing quality-matrix run directory}
REPAIRED_BIN=${REPAIRED_BIN:-${DS4_BIN:-./ds4}}
MODEL=${MODEL:-./ds4flash.gguf}
DSPARK_MODEL=${DSPARK_MODEL:-./gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
TOKENS=${TOKENS:-768}
CONFIDENCES=${CONFIDENCES:-"0 0.3 0.5 0.7"}
FAMILY_SWEEP_MODE=${FAMILY_SWEEP_MODE:-cumulative}

F3=FAMILY_F16_BATCH_EXT_VS_SINGLE_MV
F4=FAMILY_ROUTER_WEIGHT_NORMALIZATION_BATCH_REDUCE_VS_SINGLE_KERNEL
F5=FAMILY_ROUTED_MOE_IQ2_XXS_Q2_K_BATCH_VS_SINGLE
F6=FAMILY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD_VS_SINGLE_FUSED_HC
F7=FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV

[[ -f "$RUN_DIR/performance.tsv" ]] || { echo "missing $RUN_DIR/performance.tsv" >&2; exit 2; }
[[ -x "$REPAIRED_BIN" ]] || { echo "missing executable $REPAIRED_BIN" >&2; exit 2; }
[[ -f "$MODEL" && -f "$DSPARK_MODEL" ]] || { echo "missing model/support model" >&2; exit 2; }

tag_conf() { local x=${1//./p}; printf 'c%s' "$x"; }

extract_tps() {
  local log=$1
  local v
  v=$(grep -Eo '[0-9]+([.][0-9]+)?[[:space:]]*t/s' "$log" 2>/dev/null | tail -n 1 | sed -E 's/[[:space:]]*t\/s//' || true)
  [[ -n "$v" ]] && printf '%s' "$v" || printf 'NA'
}

base_tps() {
  local prompt=$1
  awk -F '\t' -v p="$prompt" '$1==p && $2=="sequential" {print $5; exit}' "$RUN_DIR/performance.tsv"
}

speedup() {
  local x=$1 b=$2
  [[ "$x" == NA || -z "$b" || "$b" == NA || "$b" == 0 ]] && { printf 'NA'; return; }
  awk -v x="$x" -v b="$b" 'BEGIN {printf "%.4f", x/b}'
}

arm_exists() {
  local prompt=$1 arm=$2
  awk -F '\t' -v p="$prompt" -v a="$arm" '$1==p && $2==a {found=1} END {exit !found}' "$RUN_DIR/performance.tsv"
}

run_variant() {
  local prompt_name=$1 conf=$2 label=$3 repairs=$4 use_f1=$5
  local tag arm prompt_file out log status tps base sp prompt
  tag=$(tag_conf "$conf")
  arm="fast_${label}_${tag}"
  prompt_file="$RUN_DIR/prompts/$prompt_name.txt"
  out="$RUN_DIR/$prompt_name/$arm.txt"
  log="$RUN_DIR/$prompt_name/$arm.log"
  [[ -f "$prompt_file" ]] || { echo "missing prompt $prompt_file" >&2; exit 2; }
  arm_exists "$prompt_name" "$arm" && return 0

  # Match command-substitution semantics used by the legacy matrix: trailing
  # newlines from the prompt file are removed before -p.
  prompt=$(cat "$prompt_file")
  echo "[$prompt_name] $arm"
  (
    unset DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT DS4_DSPARK_STATS
    unset DS4_FAMILY_REPAIRS DS4_FAMILY1_REPAIR DS4_FAMILY2_REPAIR DS4_FAMILY3_REPAIR
    unset DS4_FAMILY4_REPAIR DS4_FAMILY5_REPAIR DS4_FAMILY6_REPAIR DS4_FAMILY7_REPAIR
    if [[ "$use_f1" == 1 ]]; then
      env DS4_DSPARK_STATS=1 DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1 \
          DS4_FAMILY1_REPAIR=ALL DS4_FAMILY_REPAIRS="$repairs" \
          "$REPAIRED_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink \
          --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$conf" -p "$prompt"
    else
      env DS4_DSPARK_STATS=1 DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT=1 \
          DS4_FAMILY_REPAIRS="$repairs" \
          "$REPAIRED_BIN" -m "$MODEL" --temp 0 --tokens "$TOKENS" --nothink \
          --mtp "$DSPARK_MODEL" --dspark --dspark-confidence "$conf" -p "$prompt"
    fi
  ) >"$out" 2>"$log" && status=PASS || status=FAIL

  [[ "$status" == PASS ]] && tps=$(extract_tps "$log") || tps=NA
  base=$(base_tps "$prompt_name")
  sp=$(speedup "$tps" "$base")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$prompt_name" "$arm" "$conf" "$status" "$tps" "$sp" >> "$RUN_DIR/performance.tsv"
}

cumulative_variants=(
  "f1_f3_f4|$F3,$F4|1"
  "f1_f3_f4_f5|$F3,$F4,$F5|1"
  "f1_f3_f4_f5_f6|$F3,$F4,$F5,$F6|1"
  "f1_f3_f4_f5_f6_f7|$F3,$F4,$F5,$F6,$F7|1"
)
individual_variants=(
  "f3|$F3|0"
  "f4|$F4|0"
  "f5|$F5|0"
  "f6|$F6|0"
  "f7|$F7|0"
)

case "$FAMILY_SWEEP_MODE" in
  cumulative) variants=("${cumulative_variants[@]}") ;;
  individual) variants=("${individual_variants[@]}") ;;
  all) variants=("${individual_variants[@]}" "${cumulative_variants[@]}") ;;
  off|none|0) exit 0 ;;
  *) echo "unknown FAMILY_SWEEP_MODE=$FAMILY_SWEEP_MODE" >&2; exit 2 ;;
esac

for prompt in warehouse intervals; do
  for conf in $CONFIDENCES; do
    for spec in "${variants[@]}"; do
      IFS='|' read -r label repairs use_f1 <<< "$spec"
      run_variant "$prompt" "$conf" "$label" "$repairs" "$use_f1"
    done
  done
done

# Family 2 (flash-attention batch-direct family) is intentionally absent:
# family_repair manifest on this experiment line marks it NOT_IMPLEMENTED.
