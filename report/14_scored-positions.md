# 14 校正の読む位置を採点位置に寄せる

2026-08-30 / 実験 `experiments/13_scored_positions/`（段1）と
`experiments/14_scored_alloc_search/`（段2）/
出力 `result_logs/bench_scored_seed{0,1,2}`、`result_logs/benchqa_s05_w{2,3}_seed0`、
`result_logs/score_uniform_benchqa_s05_x2to6_seed0`、`result_logs/bench_benchqa_s05_seed0` /
対照は [report/11](11_*bench-calibration.md)（`result_logs/bench_benchtrain_seed{0,1,2}`、
`result_logs/benchtrain_w{2,3}_n8_seed0`）

## 問い

評価（lm-eval）が読むのは、選択肢のトークンの対数尤度を作っている位置だけである。
校正の側をそこへ寄せると何か動くか。動かせる場所が2つあるので、2段に分けて測った。

1. **分割**を採点位置の活性だけで作る（配分は `uniform4` に固定）
2. **探索の目的関数**でも答え部分に重みを置き、層ごとの x を探す

段1で `benchqa`（1問1系列・採点位置に印）と `--profile-positions scored` を、
段2で `--scored-weight`（答え部分が目的関数に占める割合）を入れた。

## 設定

**共通**

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（スパース率25%） |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| ベンチ | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot・全件、batch 32 |
| ブートストラップ | 10,000回 / seed 20260813 |
| dense の基準 | report/04 のものを取り込み（macro `acc` 0.6464 / `acc_norm` 0.6847） |
| コード | `d826f44` + 未コミットの `--profile-positions` / `--scored-weight` |

**段1（分割だけを寄せる）**

| | |
|---|---|
| 校正 | `benchqa` 8 × 2048 = 採点位置 16,498（総 364,844トークン、1,852問）、seed 0 / 1 / 2 |
| 数える位置 | 採点位置のみ（`--profile-positions scored`） |
| 配分 | `uniform4` 固定 |
| 対照 | report/11 の `benchtrain`（16,384トークン）× `uniform4`、同 seed |
| 所要 | 1 seed あたり 12.8分（3 seed） |

**段2（探索の目的関数も寄せる）**

| | |
|---|---|
| 校正 | `benchqa` 1 × 1100 = 採点位置 1,157（総 16,214トークン、134問）。report/11 と**総トークン数**を揃えた |
| 重みの内訳 | 答え部分 1,157 位置 / 文脈 3,720 位置 / 埋め 11,337 位置 |
| 答え部分の重み | `--scored-weight 0.5`（群の合計が半分ずつ。1位置あたりでは答えが文脈の3.21倍、埋めは0） |
| オラクル / 探索 | `suffix_kl` / beam 幅2・幅3 |
| 採点した一様 | x = 2, 3, 4, 5, 6（探索と同じオラクル・同じ重み） |
| 測定した配分 | `uniform3`（対照）、`uniform2`、beam2、beam3 |
| seed | 0 のみ |
| 所要 | 探索 幅2 12.9分 / 幅3 20.0分、一様5本 5.0分、測定4構成 37.5分 |

段2 の `--seqlen` は 1100 なので、PPL の窓幅が report/11（2048）と違う。PPL は
実験内でだけ比べている。`acc` は窓幅に依らない。

## 結果

### 段1 分割を採点位置で作る（`uniform4` 固定、3 seed）

report/11 の `benchtrain` × `uniform4` との、seed を層とするペア付き再抽出
（10,000回）。`*` は 95% 区間が0をまたがない。

| 指標 | 対照（benchtrain） | 候補（benchqa+scored） | 対照比 |
|---|---|---|---|
| `acc` | 0.6160 | 0.6175 | +0.001534 [−0.002972, +0.005827] 9/15 |
| `acc_norm` | 0.6524 | 0.6546 | +0.002147 [−0.002270, +0.006521] 9/15 |
| `gold_nll` | 3.6199 | 3.6023 | −0.017587 [−0.034488, −0.000685]* 10/15 |
| `margin` | 1.5063 | 1.5608 | +0.054557 [+0.027147, +0.082220]* 10/15 |
| `ref_kl` | 0.2103 | 0.2061 | −0.004259 [−0.009957, +0.001485] 11/15 |
| `ref_agreement` | 0.8535 | 0.8593 | +0.005819 [+0.001015, +0.010571]* 9/15 |

タスクごとの `acc`（3 seed 平均、対照 → 候補）:
PIQA 0.7670 → 0.7597 / WinoGrande 0.6619 → 0.6690 / ARC-e 0.7226 → 0.7274 /
ARC-c 0.3956 → 0.3976 / HellaSwag 0.5329 → 0.5339。

PPL（差は共通評価塊の NLL、負なら候補が良い）:
WikiText-2 7.7160 → 7.7511（+0.004549 [+0.002212, +0.006856]* 1/3 seed 改善）、
C4-new 10.1573 → 10.1506（−0.000671 [−0.002202, +0.000860] 1/3）。

### 段2 答え部分に重み 0.5 を置いて探す（seed 0）

**オラクルの score**（同じ校正トークン・同じ重みの上での値。小さいほど良い）

| 配分 | 平均 x | score |
|---|---|---|
| beam3 | 2.72 | **0.204446** |
| beam2 | 2.81 | 0.204757 |
| `uniform2` | 2.00 | 0.240480 |
| `uniform3` | 3.00 | 0.246468 |
| `uniform6` | 6.00 | 0.278327 |
| `uniform4` | 4.00 | 0.281743 |
| `uniform5` | 5.00 | 0.286029 |

出た配分:

```
beam2  5,1,2,5,6,1,2,5,3,5,0,0,3,2,3,6,2,1,3,3,1,5,2,2,1,4,3,4,0,1,3,6
beam3  5,2,1,3,6,1,1,2,3,1,3,3,1,3,4,5,2,1,4,3,1,1,2,3,6,6,2,5,0,4,2,1
```

**ベンチ**（マクロ平均。対照は `uniform3`、タスクを層とするペア付き再抽出）

| 配分 | `acc` | 対照比 `acc` | `acc_norm` | `gold_nll` |
|---|---|---|---|---|
| `uniform3`（対照） | 0.6000 | — | 0.6320 | 3.9256 |
| `uniform2` | 0.5970 | −0.003055 [−0.011684, +0.005915] 2/5 | 0.6290 | 3.9088 |
| beam2 | 0.5872 | **−0.012834 [−0.021554, −0.004216]*** 0/5 | 0.6290 | 3.9696 |
| beam3 | 0.5912 | −0.008785 [−0.017740, +0.000080] 1/5 | 0.6271 | 3.9523 |

**PPL**（実験内のみ）

| 配分 | WikiText-2 | C4-new |
|---|---|---|
| `uniform3` | 8.9347 | **11.0529** |
| `uniform2` | 8.9503 | 11.0668 |
| beam2 | 8.9347 | 11.1004 |
| beam3 | **8.8808** | 11.1410 |

### 参考 report/11 の同 seed（重みなし、`benchtrain`、`--seqlen 2048`）

score は校正トークンが違うので段2の値とは比べられない。

| 配分 | 平均 x | score | `acc` |
|---|---|---|---|
| `uniform4` | 4.00 | 0.136491 | 0.6170 |
| `uniform5` | 5.00 | 0.134521 | 0.6179 |
| `uniform6` | 6.00 | 0.136983 | 0.6187 |
| beam2 | 4.28 | 0.119483 | 0.6123 |
| beam3 | 4.22 | 0.118864 | 0.6113 |

この2段でこの線の測定を終えた。

## 保存されているもの

- 段1: `result_logs/bench_scored_seed{0,1,2}/`、段の一覧 `result_logs/exp13_stages_seed{0,1,2}.json`
- 段2: `result_logs/benchqa_s05_w{2,3}_seed0/`（`search.json`）、
  `result_logs/score_uniform_benchqa_s05_x2to6_seed0/`（`score.json`）、
  `result_logs/bench_benchqa_s05_seed0/`（`summary.json` と問題ごとの生の対数尤度）、
  段の一覧 `result_logs/exp14_stages_seed0.json`
- 一様 x=4,5,6 と x=2,3 を別々に採点した先の出力も
  `result_logs/score_uniform_benchqa_s05{,_low}_seed0/` に残っている

再現:

```bash
uv run python experiments/13_scored_positions/run.py --seed 0   # 1,2 も同様
uv run python experiments/compare_runs.py \
    --base result_logs/bench_benchtrain_seed0 \
    --base result_logs/bench_benchtrain_seed1 \
    --base result_logs/bench_benchtrain_seed2 --base-alloc uniform4 \
    --cand result_logs/bench_scored_seed0 \
    --cand result_logs/bench_scored_seed1 \
    --cand result_logs/bench_scored_seed2 --cand-alloc uniform4

uv run python experiments/14_scored_alloc_search/run.py --seed 0 --with-bench
```
