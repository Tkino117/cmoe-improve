#!/usr/bin/env bash
# 3 seed を1枚で順に回す。済んだ段は run.py が飛ばすので、止まったら同じコマンドで再開できる。
# 探索を先に3 seed ぶん通してから PPL へ入る（配分だけ先に見たい、が主目的のため）
set -euo pipefail
cd "$(dirname "$0")/../.."
SEEDS=(5 6 7)
for seed in "${SEEDS[@]}"; do
  echo "=== $(date '+%F %T') 探索 seed $seed ==="
  uv run python experiments/31_calib_amount/run.py --seed "$seed" --search-only
done
for seed in "${SEEDS[@]}"; do
  echo "=== $(date '+%F %T') PPL seed $seed ==="
  uv run python experiments/31_calib_amount/run.py --seed "$seed"
done
echo "=== $(date '+%F %T') 完了 ==="
