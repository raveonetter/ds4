#!/bin/sh
# 自愈下载循环: 下载 Layers37-42 (97.6GB) 直到完成, 断了自动续传重启
# 退出条件: 文件大小 >= TOTAL (curl 续传完后会 mv 成 final, 但循环里直接判 size)

# 走系统代理(8001), 否则 DNS 无法解析 huggingface.co
export http_proxy="http://127.0.0.1:8001"
export https_proxy="http://127.0.0.1:8001"
export HTTP_PROXY="http://127.0.0.1:8001"
export HTTPS_PROXY="http://127.0.0.1:8001"

OUT_DIR="/Users/sijiaguo/ds4/gguf"
FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
PART="$OUT_DIR/$FILE.part"
FINAL="$OUT_DIR/$FILE"
URL="https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/$FILE"
TOTAL=97591747456

LOG="$OUT_DIR/dl_resume_loop.log"
exec >>"$LOG" 2>&1

echo "=== 自愈循环启动 $(date '+%Y-%m-%d %H:%M:%S') ==="

round=0
while true; do
    round=$((round+1))

    # 已完成?
    if [ -s "$FINAL" ]; then
        sz=$(stat -f%z "$FINAL" 2>/dev/null || echo 0)
        if [ "$sz" -ge "$TOTAL" ]; then
            echo "[$(date '+%H:%M:%S')] 第$round轮: FINAL 已存在且完整 ($sz >= $TOTAL), 下载完成 ✅"
            break
        fi
    fi

    # part 已完整(下载完但还没 mv)?
    if [ -s "$PART" ]; then
        sz=$(stat -f%z "$PART" 2>/dev/null || echo 0)
        if [ "$sz" -ge "$TOTAL" ]; then
            echo "[$(date '+%H:%M:%S')] 第$round轮: PART 已完整, 重命名为 FINAL"
            mv -f "$PART" "$FINAL"
            echo "[$(date '+%H:%M:%S')] 下载完成 ✅"
            break
        fi
    fi

    cur=$(stat -f%z "$PART" 2>/dev/null || echo 0)
    pct=$(awk "BEGIN{printf \"%.2f\", $cur/$TOTAL*100}")
    echo "[$(date '+%H:%M:%S')] 第$round轮: 从 $cur bytes ($pct%) 续传..."

    # 单次 curl: 续传(-C -), 连接超时30s, 失败重试5次, 每次最多跑不设上限
    # 若中途卡死/连接断, curl 退出 -> 循环重来, -C - 从上次位置继续
    curl -fL --connect-timeout 30 --retry 5 --retry-delay 5 --retry-all-errors \
         --progress-meter -C - -o "$PART" "$URL"

    rc=$?
    echo "[$(date '+%H:%M:%S')] 第$round轮 curl 退出码=$rc"

    if [ "$rc" -ne 0 ]; then
        # 非零退出: 可能断网/代理抖断, 等几秒再续
        echo "[$(date '+%H:%M:%S')] 退出码$rc, 休眠5s后重试..."
        sleep 5
    else
        # 退出码0: 可能是下载完成(文件齐了)或服务器正常关闭连接
        # 下一轮 while 顶部会判断是否完整
        echo "[$(date '+%H:%M:%S')] curl 正常返回, 进入下一轮检查..."
    fi
done

echo "=== 自愈循环结束 $(date '+%Y-%m-%d %H:%M:%S') ==="
