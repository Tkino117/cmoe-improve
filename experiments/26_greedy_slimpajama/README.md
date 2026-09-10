# 26 slimpajama 校正で幅1（貪欲）を測る

## 問い

[res15](../../report/icassp/res15-layer-interaction.md) の (b) は「**候補を1本に
絞る（＝層ごとに独立に確定する）と、目的関数も下流の `acc` も最も悪くなる**」で、
根拠は [report/11](../../report/11_*bench-calibration.md) の1系しかない。その系の
校正は `benchtrain`（5タスクの train split、n=8）である。res15 自身が注意として
書いているとおり、**`result_logs/` に `*_w1_*` は `benchtrain_w1_n8_seed{0,1,2}`
しか無い**ので、査読で「幅1 が負けるのは校正のせいでは」と問われると埋められない。

ここは**校正を slimpajama n=16 に替えて幅1 を測る**。幅2・3・4 は
[report/07](../../report/07_*slimpajama-alloc-3seeds.md) に 3 seed 揃っているので、
足すのは幅1 の1本だけである。

**問いは「幅1 が最下位のままか」だけ**である。探索が一様配分に勝つかは別の問い
（report/07・res09 が扱っている）で、ここでは扱わない。

## 固定するもの

report/07 と同一。動かしたのは幅だけである。

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（スパース率25%）。`--nactive 4` で 50% も回せる |
| 校正 | slimpajama n=16 × 2048、seed 0 / 1 / 2 |
| 採点オラクル / 探索 | `suffix_kl` / beam **幅1**（= 貪欲。幅1のビームそのもので、別実装ではない） |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001）。固定 Top-K |
| ベンチ | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot・全件、batch 32 |
| PPL | wikitext2 / c4-new |
| ブートストラップ | 10,000回 / seed 20260813 |

dense の基準は測らない。report/04 の
`result_logs/bench_h4/wikitext2_seed0/bench/dense.json` を取り込む。

## 走らせる前に決めたこと

- **主判定は探索の score（`suffix_kl`）**である。幅1 と幅2/3/4 は**同じ校正
  トークンの上**で測るので直接比べられ、ベンチを通さずに済む。判定は
  「**3 seed すべてで 幅1 が最下位か**」の1本
- ベンチと PPL は記述用に測る。report/07 の効果量（`acc` で ±0.005、区間の
  半幅 ±0.005）からして、**幅1 と幅4 の差がベンチで有意にならないのは
  この設計では失敗ではない**
- 探索が一様に勝つかは**ここでは問わない**。report/11 の系では幅2・3 も
  `uniform4` に負けており、その但し書きは res15 に既にある
- 平均 x は揃わない（探索が決める）。順序対照は置かない — 置くなら
  report/22 の枠組みで、それは別の実験である

## 段と、済みの判定

1 seed は2段。**済んだ段は飛ばす**。印はファイルの存在ではない。

| 段 | コマンド | 出力 | 済みの印 |
|---|---|---|---|
| 1 探索 | `cmoe search --search beam --width 1` | `slimpajama_w1_n16_seed<N>/search.json` | `allocation` と `recheck` がある |
| 2 測定 | `cmoe run --bench` 3構成 | `bench_slimpajama_greedy_seed<N>/summary.json` | `summary` と `bench_summary` があり、構成数が合い、`failures` が無い |

A=4 では出力名が `slimpajama_a4_w1_n16_seed<N>` /
`bench_slimpajama_greedy_a4_seed<N>` になる（report/09 の命名に合わせた）。

## 段2 の3構成

先頭が `cmoe run` の中の対応のある比較の基準になる。

| 構成 | 何のために |
|---|---|
| `uniform3`（A=4 なら `uniform2`） | **再現の検査。** 既存の `bench_slimpajama_seed<N>` の同じ配分と Δ=0 になることを確かめる。合わなければ run をまたぐ比較を読んではいけない（report/22 と同じ理由） |
| 幅4 の配分（report/07 の探索結果を読む） | 対照。**候補と同じ実行の中で**測るので、幅1 対 幅4 は run をまたがない |
| 幅1 の配分（段1） | 候補 |

幅2・幅3 は測り直さない。上の再現の検査が通れば、既存の
`bench_slimpajama_seed<N>` から run をまたいで読める。

## 走らせ方

**本実行の前に必ず `--smoke` を通すこと。**

```bash
uv run python experiments/26_greedy_slimpajama/run.py --smoke    # 2層・7本・8問
uv run python experiments/26_greedy_slimpajama/run.py --seed 0   # 約1.0時間
uv run python experiments/26_greedy_slimpajama/run.py --seed 1
uv run python experiments/26_greedy_slimpajama/run.py --seed 2

# seed をまたいだまとめ（GPU 不要）
uv run python experiments/26_greedy_slimpajama/summarize.py --seeds 0,1,2
```

## 見積もり（既存の実測からの積み上げ、A=6）

| 段 | 内訳 | 見積り |
|---|---|---|
| 1 | 探索 幅1。コストは幅について線形で、幅2 の 7,168 に対し **3,696 `layer_forwards`**（`benchtrain_w1_n8_seed0` の実測値）。report/07 の幅2 が n=16 で 35分 | 約18分 |
| 2 | 3構成 × 11.1分（変換2.2 + PPL 1.4 + ベンチ 7.6。report/06 の実測） | 約33分 |
| | **1 seed** | **約51分** |
| | **3 seed** | **約2.6時間** |

A=4 は report/09 の実測で1割ほど速い。
