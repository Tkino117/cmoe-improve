# 03 SA配分キャリブレーションの適正量

## 目的

SA配分の探索に使うキャリブレーションデータが n=8 で足りているかを見極める。
seed だけを変えて探索を繰り返し、出てきた配分を評価したときの **PPL のばらつき
（標準偏差）** を測る。ばらつきが小さければ n=8 で足りている。大きければ
キャリブレーション量から設計し直す。

seed が入りうる箇所は [docs/02](../../docs/02_seed-effects.md) に一覧がある。

## 手順

1. seed を振って配分を探す（①だけを動かす）

   ```
   cmoe search --search beam --width 2 --oracle <TBD> --seed s
   ```

2. 出た配分を評価する（②は seed 0 に固定）

   ```
   cmoe run --alloc <配分> --seeds 0
   ```

3. seed ごとの PPL を並べ、標準偏差を見る

基準線として `uniform3` を `--seeds 0` で1回測る。①を通らないので単一値。

## 固定するもの

| | |
|---|---|
| モデル | meta-llama/Llama-2-7b-hf |
| N / A | 8 / 6 |
| nsamples | 8 |
| seqlen | 2048 |
| calib | wikitext2 |
| 探索 | beam 幅2 |
| router | cmoe |
| 変換用 carve seed（②） | 0 |
| 評価セット | wikitext2, c4-new（どちらも seed 非依存） |

## 振るもの

探索用 carve seed（①）のみ。

## 未決定

- **オラクル** — `local_error` は全32層で実測15.8分。`suffix_kl` は未計測で数倍以上
- **seed の本数** — 標準偏差を見るのが目的なので3本では足りない。5本以上

## 前提

キャリブレーショントークンが決まれば carve も PPL も決定的で、測定ノイズは無い
（探索の再測定で差 0、`suffix_kl` の非決定性の床も 0）。したがって観測される
ばらつきはすべて①由来である。
