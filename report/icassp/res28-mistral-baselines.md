# 28 Mistral-7B の比較手法

2026-09-13 / 実装 `experiments/21_ew_rule/`・`experiments/28_llama_moe/`・
`experiments/29_mistral_baselines.sh` / 出力
`result_logs/moe28_{control,v1random,v2}_mistral-7b_a{6,4}_seed{0,1,2}`・
`result_logs/ew_bench_mistral-7b{,_a4}_seed{0,1,2}`

[report/26](res26-llama-moe-baselines.md) の Mistral 版。分割だけの差・対応のある
差・考察は実験ノート側（`report/28_mistral-baselines.md`）にある。

## 設定

| | |
|---|---|
| モデル / 層数 | `mistralai/Mistral-7B-v0.1` / 32 |
| N / A | 8 / 6（スパース率25%）・8 / 4（50%） |
| 校正 | slimpajama 16本×2048、seed 0 / 1 / 2（全手法で同一） |
| 分割 | 現行 CMoE / LLaMA-MoE(Random) / LLaMA-MoE-v2 |
| 配分 | 提案（beam 幅4）/ EW-rule / 一様 |
| ルーター | 現行 CMoE（k_act 10 / bias_speed 0.001）。固定 Top-K、全手法で同一 |
| 評価 | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot・全件。PPL は WikiText-2 / C4-new |
| dense の基準 | `result_logs/exp25_check/bench_slimpajama_mistral-7b_seed0/bench/dense.json`（5タスク）と `result_logs/dense_ppl_mistral-7b-v0.1/`（PPL） |
| 追加学習 | 行わない |

手法の中身（移送の範囲、原典と違えた点）は report/26 と report/21 に書いたもの
から変えていない。配分ベクトルは report/27 の探索結果をそのまま読んでいる。

**対照（現行 CMoE の分割）は測り直した。** report/26 は既存の
`bench_slimpajama*` を対照に流用したが、Mistral のそれは report/27 を測った
別マシンの出力である。比べる配分だけをこのマシン（RTX PRO 5000 Blackwell 1枚）
で引き直し、比較を1台に閉じた。dense の基準だけは取り込みである。

## 結果（3 seed 平均）

PPL は WikiText-2 / C4-new、以降は各タスクの正答率(%)と、その平均。

**スパース率 25%（A=6）**

| 手法 | 配分 | WikiText-2 | C4 | PIQA | WinoG. | ARC-e | ARC-c | HellaS. | Avg. |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 提案（beam 幅4） | beam 幅4 | 8.10 | 11.94 | 73.50 | 67.85 | 68.17 | 37.37 | 51.23 | 59.62 |
| EW-rule | 規則が出す層別配分 | 8.16 | 12.21 | 73.45 | 66.04 | 68.90 | 37.17 | 51.08 | 59.33 |
| LLaMA-MoE（Random） | uniform0 | 54.71 | 68.04 | 68.28 | 59.41 | 54.48 | 31.31 | 42.95 | 51.28 |
| LLaMA-MoE-v2 | uniform1 | 8.56 | 12.66 | 73.34 | 65.82 | 68.80 | 37.54 | 51.35 | 59.37 |
| dense（変換前） | — | 5.25 | 8.38 | 80.36 | 75.06 | 80.22 | 49.66 | 61.51 | 69.36 |

**スパース率 50%（A=4）**

| 手法 | 配分 | WikiText-2 | C4 | PIQA | WinoG. | ARC-e | ARC-c | HellaS. | Avg. |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 提案（beam 幅4） | beam 幅4 | 17.49 | 25.18 | 62.66 | 56.67 | 45.74 | 23.27 | 37.14 | 45.09 |
| EW-rule | 規則が出す層別配分 | 20.07 | 30.41 | 61.15 | 55.30 | 43.18 | 21.79 | 35.18 | 43.32 |
| LLaMA-MoE（Random） | uniform0 | 899.04 | 696.02 | 56.84 | 52.35 | 37.68 | 23.63 | 32.64 | 40.63 |
| LLaMA-MoE-v2 | uniform1 | 30.95 | 50.95 | 60.43 | 52.20 | 44.16 | 22.41 | 33.54 | 42.55 |
| dense（変換前） | — | 5.25 | 8.38 | 80.36 | 75.06 | 80.22 | 49.66 | 61.51 | 69.36 |
