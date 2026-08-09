#!/bin/bash
# 监控 ds4f q2-q4 主模型下载：curl 一断就续传 + 防睡眠
# 日志写到 ~/ds4_download_watch.log
set -u
LOG=~/ds4_download_watch.log
PART="/Users/sijiaguo/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf.part"
FINAL="${PART%.part}"
URL="https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# 是否已经有 curl 在跑（针对该 part 文件）
curl_running() {
  pgrep -f "curl.*DeepSeek-V4-Flash-Layers37-42Q4KExperts" >/dev/null 2>&1
}

launch() {
  log "未检测到下载进程，启动续传 (caffeinate 防空闲睡眠)..."
  # -C - 续传；用 caffeinate -i 跟随 curl 进程防睡眠
  nohup bash -c "caffeinate -i curl -fL --progress-meter -C - -o '$PART' '$URL'" \
    >>"$LOG" 2>&1 &
  log "已启动续传 curl PID=$!"
}

log "==== 监控启动 ===="
log "PART=$PART"

# 主循环：每 60 秒检查一次
while true; do
  # 已完成：part 不在了且出现正式 gguf
  if [ ! -e "$PART" ] && [ -e "$FINAL" ]; then
    log "下载完成：出现 $FINAL"
    break
  fi
  # part 消失但也没正式文件 —— 可能被脚本移动了，检查两种可能
  if [ ! -e "$PART" ] && [ ! -e "$FINAL" ]; then
    log "注意：part 与 final 都不存在，可能下载被外部中断/移动，尝试重新拉取"
    launch
    sleep 60
    continue
  fi
  if ! curl_running; then
    SIZE=$(stat -f%z "$PART" 2>/dev/null || echo 0)
    log "检测到 curl 中断 (当前已下 ${SIZE} 字节)，准备续传"
    launch
  fi
  sleep 60
done

log "==== 监控结束 ===="
