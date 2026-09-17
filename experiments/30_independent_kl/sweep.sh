#!/usr/bin/env bash
# 5 seed × 2動作点を1枚で順に回す。済んだ段は run.py が飛ばすので、止まったら同じコマンドで再開できる
set -euo pipefail
cd "$(dirname "$0")/../.."
for seed in 0 1 2 3 4; do
  for nactive in 6 4; do
    echo "=== $(date '+%F %T') seed $seed A=$nactive ==="
    uv run python experiments/30_independent_kl/run.py --seed "$seed" --nactive "$nactive"
  done
done
echo "=== $(date '+%F %T') 完了 ==="
