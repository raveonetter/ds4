#!/bin/sh
# 监控 ds4f-q2-q2-q4 (97.6GB Layers37-42) 下载进度
# 修复: 1) ETA 用滑动平均避免单点抖动  2) 连续多次零增长才判假死

OUT_DIR="/Users/sijiaguo/ds4/gguf"
PART="$OUT_DIR/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf.part"
FINAL="$OUT_DIR/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"

TOTAL=97591747456   # 服务端实测 97.59 GB
TS=$(date '+%Y-%m-%d %H:%M:%S')

# 进程存活?
if pgrep -f "download_model.sh ds4f-q2-q4" >/dev/null 2>&1 || pgrep -f "Layers37-42.*\.part" >/dev/null 2>&1; then
    ALIVE="ALIVE"
else
    ALIVE="DEAD"
fi

if [ -s "$FINAL" ]; then
    SIZE=$(stat -f%z "$FINAL" 2>/dev/null || echo 0)
    PCT=$(awk "BEGIN{printf \"%.2f\", $SIZE/$TOTAL*100}")
    echo "[$TS] FINAL-EXISTS size=$SIZE ($PCT%) process=$ALIVE"
    if [ "$SIZE" -ge "$TOTAL" ]; then
        echo "[$TS] DOWNLOAD COMPLETE ✅"
    fi
    rm -f "$OUT_DIR/.dl_prev" "$OUT_DIR/.dl_ts" "$OUT_DIR/.dl_stall"
    exit 0
fi

if [ -s "$PART" ]; then
    SIZE=$(stat -f%z "$PART" 2>/dev/null || echo 0)
    PCT=$(awk "BEGIN{printf \"%.2f\", $SIZE/$TOTAL*100}")
    PREV=$(cat "$OUT_DIR/.dl_prev" 2>/dev/null || echo 0)
    PREV_TS=$(cat "$OUT_DIR/.dl_ts" 2>/dev/null || echo 0)
    NOW_TS=$(date +%s)
    if [ "$PREV" -gt 0 ] && [ "$SIZE" -ge "$PREV" ] && [ "$PREV_TS" -gt 0 ]; then
        DELTA=$((SIZE-PREV))
        ELAPSED=$((NOW_TS-PREV_TS))
        if [ "$ELAPSED" -gt 0 ]; then
            RATE=$(awk "BEGIN{printf \"%.4f\", $DELTA/$ELAPSED/1024/1024}")
            # 滑动平均: 保留最近一次速率到 .dl_rate,与历史均值混合
            OLD=$(cat "$OUT_DIR/.dl_rate" 2>/dev/null || echo 0)
            AVG=$(awk "BEGIN{printf \"%.4f\", ($RATE+$OLD)/2}")
            echo "$AVG" > "$OUT_DIR/.dl_rate"
            if [ "$(awk "BEGIN{print ($AVG>0)}")" = "1" ]; then
                ETA=$(awk "BEGIN{rem=($TOTAL-$SIZE)/1024/1024/$AVG/3600; if(rem>=1){printf \"%.1f h\", rem}else{printf \"%.0f min\", rem*60}}")
            else
                ETA="n/a"
            fi
            SPEED="+$AVG MB/s (avg) ETA=$ETA"
        else
            SPEED="(间隔0)"
        fi
        # 假死检测: 连续3次零增长才告警
        if [ "$SIZE" -eq "$PREV" ]; then
            STALL=$(cat "$OUT_DIR/.dl_stall" 2>/dev/null || echo 0)
            STALL=$((STALL+1))
            echo "$STALL" > "$OUT_DIR/.dl_stall"
        else
            echo "0" > "$OUT_DIR/.dl_stall"
            STALL=0
        fi
    else
        SPEED="(基线)"
        echo "0" > "$OUT_DIR/.dl_stall"
    fi
    echo "[$TS] PART size=$SIZE ($PCT%) process=$ALIVE $SPEED"
    echo "$SIZE" > "$OUT_DIR/.dl_prev"
    echo "$NOW_TS" > "$OUT_DIR/.dl_ts"
    # 假死告警
    STALL=$(cat "$OUT_DIR/.dl_stall" 2>/dev/null || echo 0)
    if [ "$STALL" -ge 3 ]; then
        echo "[$TS] ⚠️  STALL: 连续3次采样零增长, 可能代理假死! 当前进程=$ALIVE"
    fi
else
    echo "[$TS] NO PART FILE process=$ALIVE"
fi

# 进程死了且 part 不完整 -> 告警
if [ "$ALIVE" = "DEAD" ] && [ ! -s "$FINAL" ]; then
    echo "[$TS] ⚠️  DOWNLOAD PROCESS DEAD AND INCOMPLETE!"
fi
