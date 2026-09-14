#!/usr/bin/env bash
# 29 Mistral-7B の比較手法を埋める。GPU 1枚で順に回す。
#
#   bash experiments/29_mistral_baselines.sh
#   SEEDS="0 1 2" NACTIVES="6 4" bash experiments/29_mistral_baselines.sh
#
# 埋めるのは EW-rule / LLaMA-MoE(Random) / LLaMA-MoE-v2 の3本である。dense は
# experiments/25 の ``--check`` が測ってあるので測らない（各 run が取り込む）。
#
# **現行 CMoE の分割（control）も同じマシンで測り直す。** report/26 は既存の
# ``bench_slimpajama*`` を対照に流用したが、Mistral のそれは report/27 を測った
# 別マシンの出力である。比べる3つの配分（uniform0 / uniform1 / 提案 幅4）だけを
# ここで引き直して、分割の比較を1台に閉じる。
#
# 順序は「25%(A=6) の3 seed を先に終える」。区間を出すには3 seed 要るので、
# 途中で止まったときに使えるのは片方のスパース率が3 seed 揃った側である。
#
# **段の前に GPU が空くのを待つ。** 1枚しか無いので、前の段の子プロセスが残って
# いると次が確実に OOM で落ちる（`cmoe run` は親を殺しても生き残ることがある）。
set -u
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mistral-7b}"
SEEDS="${SEEDS:-0 1 2}"
NACTIVES="${NACTIVES:-6 4}"
RETRIES="${RETRIES:-2}"
FREE_MIB="${FREE_MIB:-40000}"     # これだけ空いていれば 7B は載る
WAIT_MAX="${WAIT_MAX:-30}"        # 空くのを待つ上限（分）
LOG_DIR="result_logs/exp29_drive"
mkdir -p "$LOG_DIR"

wait_for_gpu() {
  for minute in $(seq 1 "$WAIT_MAX"); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$FREE_MIB" ] && return 0
    echo "  GPU の空きが ${free}MiB しか無い。60秒待つ（${minute}/${WAIT_MAX}）"
    sleep 60
  done
  echo "!!! GPU が ${WAIT_MAX} 分空かなかった。誰かが掴んだままである"
  return 1
}

attempt() {   # attempt <ログ> <札> <コマンド...>
  local log="$1" name="$2"; shift 2
  for try in $(seq 1 "$RETRIES"); do
    wait_for_gpu | tee -a "$log" || return 1
    echo "=== $name 試行 $try/$RETRIES  $(date +%F_%H:%M:%S) ===" | tee -a "$log"
    if "$@" >>"$log" 2>&1; then
      echo "=== $name 済み $(date +%F_%H:%M:%S) ===" | tee -a "$log"
      return 0
    fi
    echo "!!! $name の試行 $try が落ちた。$log の末尾を見ること" | tee -a "$log"
    sleep 30
  done
  return 1
}

seed_list=$(echo "$SEEDS" | tr ' ' ',')

for nactive in $NACTIVES; do
  # 28（LLaMA-MoE v1 / v2 と現行 CMoE の対照）。run.py 側が済んだ段を飛ばす
  attempt "$LOG_DIR/moe28_${MODEL}_a${nactive}.log" "28 $MODEL A=$nactive" \
    uv run python experiments/28_llama_moe/run.py \
      --model "$MODEL" --method control,v1,v2 --seeds "$seed_list" \
      --nactives "$nactive"

  # 21（EW-rule）。(seed, A) ごとに1本
  for seed in $SEEDS; do
    log="$LOG_DIR/ew_${MODEL}_a${nactive}_seed${seed}.log"
    out="result_logs/ew_bench_${MODEL}$([ "$nactive" = 6 ] || echo "_a${nactive}")_seed${seed}"
    if [ -f "$out/summary.json" ]; then
      echo "=== EW $MODEL A=$nactive seed=$seed: 済み ===" | tee -a "$log"
      continue
    fi
    attempt "$log" "EW $MODEL A=$nactive seed=$seed" \
      uv run python experiments/21_ew_rule/run.py \
        --model "$MODEL" --seed "$seed" --nactive "$nactive"
  done
done

# EW-rule と**平均 x を揃えた一様**。report/21 の結論（「EW-rule ≈ 同じ平均 x の
# 一様配分」）を Mistral でも言うのに要る。A=6 は EW の run 自身が uniform3 を
# 同梱しているので足りており、揃っていないのは A=4 だけである（EW の平均 x は
# 2.78〜3.00 で、run 内の対照は uniform2）。
if [ "${MATCHED:-1}" = 1 ]; then
  for seed in $SEEDS; do
    out="result_logs/moe28_matched_${MODEL}_a4_seed${seed}"
    [ -f "$out/summary.json" ] && continue
    attempt "$LOG_DIR/matched_${MODEL}_a4_seed${seed}.log"       "揃えた一様 uniform3 A=4 seed=$seed"       uv run cmoe run --model mistralai/Mistral-7B-v0.1 --carver cmoe         --calib slimpajama --nsamples 16 --nexperts 8 --nactive 4         --seeds "$seed" --datasets wikitext2,c4-new --bench         --bench-batch-size 32 --alloc uniform3 --out "$out"         --bench-reference         result_logs/exp25_check/bench_slimpajama_mistral-7b_seed0/bench/dense.json
  done
fi
echo "=== 29 終わり $(date +%F_%H:%M:%S) ==="
