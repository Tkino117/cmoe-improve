# 26 LLaMA-MoE / LLaMA-MoE-v2 の分割規則との比較

2026-09-10 / 実装 `src/cmoe/carve/llama_moe.py`・`experiments/28_llama_moe/` /
出力 `result_logs/moe28_*`・`result_logs/llama_moe_v2_probe_seed{0,1,2}`

## 設定

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（25%）・8 / 4（50%） |
| 校正 | slimpajama 16本×2048、seed 0 / 1 / 2（全手法で同一） |
| 分割 | 現行 CMoE / LLaMA-MoE(Random) / LLaMA-MoE-v2 |
| ルーター | 現行 CMoE（k_act 10 / bias_speed 0.001）。固定 Top-K、全手法で同一 |
| 評価 | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot・全件。PPL は WikiText-2 / C4-new |
| dense の基準 | `result_logs/bench_h4/wikitext2_seed0/bench/dense.json` |
| ブートストラップ | 10,000回 / seed 20260813。層は (seed × タスク)、単位は問題 |
| 追加学習 | 行わない |

**動かしたのは分割（軸3）だけである。** 両手法は学習前提（v1 は継続事前学習
200B トークン、v2 は post-training 約7B トークン）で、ここに移したのは分割規則の
み。CMoE 最新版 (ACL 2026) も分割だけを再実装して比較している（§5.1、Table 6 の
"† Split-only; training time not included"）。

現行 CMoE の分割で同じ配分を測ったものは `bench_slimpajama{,_a4}_seed<N>` の
既存の測定を用い、測り直していない。MMLU は測らない。

### 移送

**LLaMA-MoE v1**（pjlab-sys4nlp/llama-moe の `RandomSplit`）: `[0..E-1]` を
`i//E` 回並べて shuffle する等サイズ・非重複のランダム割り当て。校正データも重みも
読まない。種は層ごとに変える。原典は Clustering / Co-activation Graph / Gradient も
試したうえで Random を最良としており、Random のみを移した。原典に shared expert は
無く、動作点は `uniform0` が対応する。

**LLaMA-MoE-v2**（OpenSparseLLMs/LLaMA-MoE-v2 の `GradientSplitResidual`、
`criterion="max"`, `share_neurons=False`）: 3段。

1. **ゲート** … 層ごとに MLP 入力の隠れ状態を routed expert と同数のクラスタへ
   balanced k-means で分ける
2. **重要度** … LM 損失を backward し、各トークンを `argmax(h · center^T)` で
   クラスタへ割り当て、中間活性 f について `|f ⊙ ∇_f L|` を**クラスタ内で**平均する
   （重要度はクラスタごとに1本）
3. **分割** … residual（shared）には各クラスタの上位 `expert_size` 件を数えて
   **選んだクラスタ数が多いニューロン**から詰め、残りを重要度そのものの貪欲な
   最大値選択で routed へ等サイズに配る

x は原典の `expert_num_residual` に対応し、主構成 `1+7top1` は x=1。
**クラスタ数 = routed expert 数**なので、x を変えるとプローブを取り直す必要がある。

原典と違えた点: (a) ゲートのクラスタリングは `k_means_constrained.KMeansConstrained`
（jitter 0.4）ではなく同じ上下限を使う貪欲な近似で、クラスタリングに使うトークンは
4096 に間引いた（重要度は全トークン）、(b) 校正は slimpajama、(c) ルーターは
現行 CMoE 固定（v2 のゲートは分割にのみ使う）、(d) 追加学習なし、(e) **Attention
MoE は移していない** — 原典 Table 1 の構成も `MLP-MoE (8top2)` /
`MLP-MoE (1+7top1)` で、attention と併用したモデルは主結果に無い。

## 結果1: 手法の構成どうし（3 seed 平均）

PPL は WikiText-2 / C4-new、以降は各タスクの正答率(%)と、その平均。

**スパース率 25%（A=6）**

| 手法 | 配分 | WikiText-2 | C4 | PIQA | WinoG. | ARC-e | ARC-c | HellaS. | Avg. |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 提案（beam 幅4） | beam 幅4 | 7.33 | 9.92 | 73.94 | 66.32 | 67.73 | 34.44 | 50.29 | 58.55 |
| LLaMA-MoE（Random） | uniform0 | 29.90 | 19.31 | 67.10 | 60.59 | 57.80 | 30.89 | 43.45 | 51.97 |
| LLaMA-MoE-v2 | uniform1 | 7.86 | 10.48 | 71.62 | 61.96 | 65.07 | 32.96 | 49.11 | 56.14 |
| dense（変換前） | — | 5.47 | 7.27 | 78.02 | 69.53 | 75.67 | 42.92 | 57.08 | 64.65 |

**スパース率 50%（A=4）**

| 手法 | 配分 | WikiText-2 | C4 | PIQA | WinoG. | ARC-e | ARC-c | HellaS. | Avg. |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 提案（beam 幅4） | beam 幅4 | 16.36 | 20.88 | 63.04 | 58.59 | 47.53 | 23.78 | 37.71 | 46.13 |
| LLaMA-MoE（Random） | uniform0 | 608.19 | 367.30 | 56.02 | 50.72 | 31.82 | 21.16 | 28.26 | 37.60 |
| LLaMA-MoE-v2 | uniform1 | 24.35 | 32.71 | 61.32 | 53.20 | 41.69 | 23.24 | 33.54 | 42.60 |
| dense（変換前） | — | 5.47 | 7.27 | 78.02 | 69.53 | 75.67 | 42.92 | 57.08 | 64.65 |

## 結果2: 分割だけの差（同じ配分どうし、3 seed 平均）

**スパース率 25%（A=6）**

| 配分 | 分割 | WikiText-2 | C4 | PIQA | WinoG. | ARC-e | ARC-c | HellaS. | Avg. |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| uniform0 | 現行 CMoE | 86.10 | 68.05 | 69.99 | 59.25 | 61.46 | 32.03 | 45.66 | 53.68 |
| uniform0 | LLaMA-MoE（Random） | 29.90 | 19.31 | 67.10 | 60.59 | 57.80 | 30.89 | 43.45 | 51.97 |
| uniform1 | 現行 CMoE | 7.77 | 10.47 | 72.62 | 63.59 | 66.48 | 33.42 | 48.85 | 56.99 |
| uniform1 | LLaMA-MoE-v2 | 7.86 | 10.48 | 71.62 | 61.96 | 65.07 | 32.96 | 49.11 | 56.14 |
| beam 幅4 | 現行 CMoE | 7.33 | 9.92 | 73.94 | 66.32 | 67.73 | 34.44 | 50.29 | 58.55 |
| beam 幅4 | LLaMA-MoE（Random） | 9.65 | 12.29 | 71.94 | 62.48 | 64.20 | 33.76 | 47.90 | 56.06 |

**スパース率 50%（A=4）**

| 配分 | 分割 | WikiText-2 | C4 | PIQA | WinoG. | ARC-e | ARC-c | HellaS. | Avg. |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| uniform0 | 現行 CMoE | 5442.10 | 1648.23 | 55.46 | 49.30 | 31.14 | 21.22 | 28.18 | 37.06 |
| uniform0 | LLaMA-MoE（Random） | 608.19 | 367.30 | 56.02 | 50.72 | 31.82 | 21.16 | 28.26 | 37.60 |
| uniform1 | 現行 CMoE | 21.04 | 28.65 | 59.25 | 54.80 | 40.26 | 21.93 | 34.18 | 42.08 |
| uniform1 | LLaMA-MoE-v2 | 24.35 | 32.71 | 61.32 | 53.20 | 41.69 | 23.24 | 33.54 | 42.60 |
| beam 幅4 | 現行 CMoE | 16.36 | 20.88 | 63.04 | 58.59 | 47.53 | 23.78 | 37.71 | 46.13 |
| beam 幅4 | LLaMA-MoE（Random） | 1131.43 | 1301.55 | 56.49 | 50.01 | 31.73 | 21.42 | 27.72 | 37.48 |

## 結果3: 対応のある差

符号は「候補 − 対照」。`acc` は正、`gold_nll` は負が候補の勝ち。`*` は95%区間が
0をまたがない。`n/15` は候補が改善した (seed × タスク) の層数。

**対照 = 提案（beam 幅4）**

| スパース率 | 候補 | `acc` | `gold_nll` |
|---|---|--:|--:|
| 25% | LLaMA-MoE（Random）@ uniform0 | −0.0658* 0/15 | +4.236* 0/15 |
| 25% | LLaMA-MoE-v2 @ uniform1 | −0.0240* 0/15 | +0.266* 0/15 |
| 50% | LLaMA-MoE（Random）@ uniform0 | −0.0853* 0/15 | +9.150* 0/15 |
| 50% | LLaMA-MoE-v2 @ uniform1 | −0.0353* 1/15 | +1.812* 0/15 |

**対照 = 現行 CMoE の分割（同じ配分）**

| スパース率 | 配分 | 候補 | `acc` | `gold_nll` |
|---|---|---|--:|--:|
| 25% | uniform0 | Random | −0.0171* 8/15 | +2.725* 6/15 |
| 25% | uniform1 | v2 | −0.0085* 3/15 | +0.046* 3/15 |
| 25% | beam 幅4 | Random | −0.0249* 1/15 | +0.543* 0/15 |
| 50% | uniform0 | Random | +0.0054 [−0.0010, +0.0120] | −1.896* 10/15 |
| 50% | uniform1 | v2 | +0.0051 [−0.0007, +0.0110] | +0.425* 0/15 |
| 50% | beam 幅4 | Random | −0.0865* 0/15 | +6.875* 0/15 |
