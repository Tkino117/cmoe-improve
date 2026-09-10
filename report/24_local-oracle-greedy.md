# 24 層ローカルのオラクルで貪欲に決めると、ほぼ一様配分に戻る

2026-08-25 測定 / 2026-09-09 記録 /
出力 `result_logs/order_local_error_{forward,reverse}_slim_n16_seed0`、
`result_logs/order_score_{local_error_forward,suffix_kl}_slim_n16_seed0` /
対照は [report/07](07_*slimpajama-alloc-3seeds.md) の幅2・3・4 と
[report/23](23_greedy-slimpajama.md) の幅1（**どれも同じ校正トークンの上**）

**この4 run はこれまでどのレポートにも入っていない。** [report/22](22_alloc-baselines.md)
が LExI 相当の移送の裏取りとして存在にだけ触れている。ここで数値を記録する。

## 何を測ったものか

当時の CLI には `--order` があった（今のソースには無い）。`--oracle local_error
--search greedy` で、層を前から決める向き（`forward`）と後ろから決める向き
（`reverse`）の2本を走らせている。

- **`forward`** — 変換済みの接頭辞の上を進む貪欲。層 ℓ の候補は層 0..ℓ−1 の
  選択に依存する
- **`reverse`** — 未決定の層は上流にあり dense のまま。層ローカルのオラクルは
  下流を見ないので、**層ごとのスコアが他の層の選択に依存しない独立な argmin に
  なる**（run.txt にその注記が残っている）。report/22 の LExI 相当が出した配分と
  32層すべてで一致する

`local_error` が測るのはその層の出力の相対二乗誤差

    L = Σ_t ‖ (h_t ⊙ missed_t) W_down^T ‖² / Σ_t ‖ h_t W_down^T ‖²

で、**後続の層を1つも走らせない**。`suffix_kl`（最終出力の分布の KL）とは
測っているものが違う。

## 設定

report/07 の seed 0 と同一。動かしたのはオラクルと探索だけである。

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（スパース率25%） |
| 校正 | slimpajama n=16 × 2048、seed 0。トークンのハッシュ `37738f7c8d47` |
| オラクル / 探索 | `local_error` / greedy（幅1）、向き forward と reverse |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| コード | 2026-08-25 の作業木。`--order` を持つ版で、現在のソースには無い |

**校正トークンのハッシュが report/07・23 と同一**なので、下の `suffix_kl` の
採点は同じ表に並べて読める。非決定性の床は `suffix_kl` の採点で 0.000e+00、
勝った配分の測り直しの差は2本とも 0.000e+00 だった。

## 結果

### 出た配分は、ほぼ全層が x=6 である

```
forward  4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,4,5,2,5,6   平均 x 5.69
reverse  4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,6,3,6,6   平均 x 5.75
```

| | forward | reverse |
|---|--:|--:|
| x について単調非増加だった層 | 26/32 | 28/32 |
| 層ごとの argmin が x=6 だった層 | 27/32 | 29/32 |
| 相対 margin（2位と1位の差 / 1位）の中央値 | 2.49e-03 | 2.34e-03 |
| 同 最小 | 1.7e-06 | 9.9e-08 |
| 平均 x | 5.69 | 5.75 |

**`local_error` は x についてほぼ単調に減る。** したがって層ごとの argmin は
ほとんど常に上限の x=6 になり、配分は `uniform6`（平均 x=6）のすぐ隣に落ちる。
x=6 の層は Top-K=0 なのでルーターが出力に効かない（report/13）。

差が付く層でも margin は小さく、中央値で 2e-03、最小では 1e-07 台である。

forward と reverse は**層27〜31 の5層でだけ食い違う**（`…,4,5,2,5,6` 対
`…,3,6,3,6,6`）。baseline を変換済みに取るか dense に取るかは、この動作点で実際に
答えを変える。

### 同じ校正トークンの上で `suffix_kl` を測る

`order_score_suffix_kl_slim_n16_seed0` が6本を採点している。そこに
report/07 の幅2・3・4 と report/23 の幅1 を足したものが下の表である
（**全部 seed 0・同じ 32,768トークン**）。

score の悪い順に並べた（下ほど良い）。

| 配分の決め方 | 平均 x | `suffix_kl` | `uniform6` 比 |
|---|--:|--:|--:|
| `uniform3` | 3.00 | 0.207316 | +7.9% |
| `uniform4` | 4.00 | 0.195045 | +1.5% |
| `uniform6`（seed 0 で最良の一様） | 6.00 | 0.192136 | — |
| **`local_error` 貪欲 reverse（独立 argmin）** | 5.75 | **0.190275** | −1.0% |
| **`local_error` 貪欲 forward（接頭辞）** | 5.69 | **0.189543** | −1.3% |
| CMoE-ref の beam プリセット（wikitext2 校正） | 4.88 | 0.185115 | −3.7% |
| `suffix_kl` 幅2（report/07） | 4.75 | 0.183640 | −4.4% |
| `suffix_kl` 幅1（貪欲、report/23） | 4.97 | 0.182807 | −4.9% |
| `suffix_kl` 幅3（report/07） | 4.50 | 0.181855 | −5.4% |
| `suffix_kl` 幅4（report/07） | 4.41 | **0.180591** | −6.0% |

一様の3行（0.207316 / 0.195045 / 0.192136）は report/07 の採点段の値と
完全に一致する。

**同じ貪欲な歩き方でも、尺度を替えると届く先が変わる。** `local_error` 貪欲は
最良の一様を 1.0〜1.3% 下回るだけで、`suffix_kl` 幅1 の 4.9% には届かない。
探索の幅は両方とも1 である。

### コスト

| | オラクル | 単位 | コスト | 評価回数 | 所要 |
|---|---|---|--:|--:|--:|
| `local_error` 貪欲 forward | 層ローカル | `layer_ffn_evals` | 224 | 224 | 12.5分 |
| `local_error` 貪欲 reverse | 層ローカル | `layer_ffn_evals` | 224 | 224 | 12.6分 |
| `suffix_kl` 幅1（report/23） | 後続まで通す | `layer_forwards` | 3,696 | 224 | 17.7分 |
| `suffix_kl` 幅4（report/07） | 後続まで通す | `layer_forwards` | 14,112 | 875 | 69.3分 |

**単位が違うので `spent` の列は縦に比べられない**（層ローカルは後続の層を走らせ
ないので、1回の評価の中身がそもそも違う）。比べられるのは評価回数と所要時間で、
幅1 どうしなら 224回 対 224回・12.5分 対 17.7分である。

## 保存されているもの

- `result_logs/order_local_error_{forward,reverse}_slim_n16_seed0/` —
  `search.json`（全層 × 全候補（x=0〜6）のスコア、層ごとの margin、
  勝った配分の測り直し、校正の内訳）と `run.txt`（層ごとの表）
- `result_logs/order_score_local_error_forward_slim_n16_seed0/` —
  2本の配分を `local_error` で採点し直したもの
- `result_logs/order_score_suffix_kl_slim_n16_seed0/` — 6本を `suffix_kl` で
  採点したもの（上の表の出どころ）
- `report/figures/fig_allocation_bands.{py,pdf,png}` — 論文の図1。この報告書の
  「独立 argmin」と、校正が選んだ一様・提案（幅4）の配分を色帯で並べたもの。
  3本の校正トークンが同一であることを検査してから描く
  （`uv run --with matplotlib python report/figures/fig_allocation_bands.py`）

## まだ測っていないもの

- **ベンチと PPL。** この4 run は採点だけで、モデルを組んで測ってはいない。
  ただし `reverse` の配分は report/22 の LExI 相当と 32層すべてで一致するので、
  そちらの測定（3 seed × 2動作点）がそのまま読める
- **seed 1・2。** これは seed 0 の1本だけである
- **50%（A=4）。** 測っていない

## 考察・結論

本レポートでは書かない。素の観測結果のみを記録した。
