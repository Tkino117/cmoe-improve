#!/usr/bin/env bash
# 5 seed を1枚で順に回す。済んだ段は run.py が飛ばすので、止まったら同じコマンドで再開できる。
# 探索を先に全 seed ぶん通してから PPL へ入る（配分だけ先に見たい、が主目的のため）
set -euo pipefail
cd "$(dirname "$0")/../.."
SEEDS=(5 6 7 8 9)
for seed in "${SEEDS[@]}"; do
  echo "=== $(date '+%F %T') 探索 seed $seed ==="
  uv run python experiments/33_independent_n64/run.py --seed "$seed" --search-only
done
for seed in "${SEEDS[@]}"; do
  echo "=== $(date '+%F %T') PPL seed $seed ==="
  uv run python experiments/33_independent_n64/run.py --seed "$seed"
done
echo "=== $(date '+%F %T') 完了 ==="
