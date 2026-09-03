#!/usr/bin/env bash
# 22 の全 (seed, A) を順に通す。段は冪等なので、落ちたら同じ組を引き直す。
#
#   bash experiments/22_alloc_baselines/drive.sh            # 既定の段
#   STAGES=grid bash .../drive.sh                           # 格子だけ後から
#   COMBOS="0 6|1 6|2 6" STAGES=grid bash .../drive.sh      # 25% だけ格子
#
# 順序は「25% の3 seed を先に終える」ようにしてある。区間を出すには3 seed 要る
# ので、途中で止まったときに使えるのは「1 seed の両スパース率」ではなく
# 「3 seed の片方のスパース率」の側である。
set -u
cd "$(dirname "$0")/../.."

STAGES="${STAGES:-probe,alloc,score,bench}"
RETRIES="${RETRIES:-3}"
LOG_DIR="result_logs/exp22_drive"
mkdir -p "$LOG_DIR"

COMBOS="${COMBOS:-0 6|1 6|2 6|0 4|1 4|2 4}"
IFS='|' read -ra COMBO_LIST <<< "$COMBOS"
for combo in "${COMBO_LIST[@]}"; do
  set -- $combo
  seed=$1; nactive=$2
  log="$LOG_DIR/seed${seed}_a${nactive}.log"
  for attempt in $(seq 1 "$RETRIES"); do
    echo "=== seed $seed A=$nactive  試行 $attempt/$RETRIES  $(date +%H:%M:%S) ===" \
      | tee -a "$log"
    if uv run python experiments/22_alloc_baselines/run.py \
         --seed "$seed" --nactive "$nactive" --stages "$STAGES" >>"$log" 2>&1; then
      echo "=== seed $seed A=$nactive  済み $(date +%H:%M:%S) ===" | tee -a "$log"
      break
    fi
    echo "!!! seed $seed A=$nactive 試行 $attempt が落ちた。$log の末尾を見ること" \
      | tee -a "$log"
    if [ "$attempt" -eq "$RETRIES" ]; then
      echo "!!! seed $seed A=$nactive を $RETRIES 回で諦めた。次の組へ進む"
    fi
    sleep 20
  done
done
echo "=== drive.sh 終わり $(date +%H:%M:%S) ==="
