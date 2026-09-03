# 20 配分探索 × 可変 Top-K の 2×2

## 問い

[report/07](../../report/07_*slimpajama-alloc-3seeds.md) は**層方向**の割り付け
（層ごとの shared 数 x）を校正データから決めると PPL が下がることを示し、
[report/19](../../report/19_routing-improvements.md) は**トークン方向**の割り付け
（routed の起動個数）を固定 Top-K から解放すると PPL と `acc` が下がることを
示した。ICASSP 原稿はこの2つを「同じ予算を2方向に配り直す」という1本の主張で
束ねている。

**その主張は、2つを同時に入れたときに成り立つか。**

先例がある。report/19 の方式7（低ランク score）と方式8（可変K）は軸としては
独立だったのに、**両方入れると単独のどちらにも届かなかった**。「独立な軸だから
足せるはず」は、この系では通らない。

## 事前の予測（走らせる前に書く）

**可変Kの効き目は、探索配分の下では uniform3 のときより小さくなると予想する。**
理由は配分探索が routing を減らす方向へ動くことで、既に測ってある幅4 の配分から
そのまま読める。

| seed | 平均 x | 平均 K（= 6 − 平均 x） | K=0 の層（32層中） |
|---|--:|--:|--:|
| 0 | 4.406 | 1.594 | 9 |
| 1 | 4.656 | 1.344 | 12 |
| 2 | 4.750 | 1.250 | 13 |

方式8 を測った動作点（`uniform3`）は平均 K=3・K=0 の層は0 である。探索配分では
**K=0 の層で可変Kが恒等**になり（`Converter` は Top-K=0 の層で全方式に同じ基準
ルーターを渡す）、残る層でも配れる幅が狭い。

予測が当たった場合に「失敗」と書くのではなく、**交互作用の大きさを測ることが
この実験の目的**である。足し算にならないなら、原稿 §4 の「両方向」は一本の主張
ではなく「一つの原理の2つの適用例」に落とす。

## 動作点をどこに置くか

A=6（スパース率25%）。両方の軸に既存の測定がある唯一の動作点で、report/07 の
探索（A=6）と report/19 の方式8（S3A3E8 = `uniform3`・A=6）が直接つながる。

校正は **slimpajama n=16**（report/07 と同じ）にする。配分探索は校正データを
選ぶ — wikitext2 / c4 校正では探索が最良の一様に勝てなかった（report/04）。
2×2 を1つの校正の上で組む以上、探索が成立する側に揃えるほかない。

**副産物として、方式8 を slimpajama 校正の上で測り直すことになる。** report/19 は
wikitext2 校正の1点でしか測っていないので、しきい値の較正が別の校正データでも
転移するかがここで分かる。

## 水準

**配分 3水準 × ルーター 2水準 = 6構成 / seed。**

| 配分 | 出どころ | 平均 K | 役割 |
|---|---|--:|---|
| `uniform3` | 固定 | 3.00 | report/07 の固定対照であり、**report/19 で方式8 を測った動作点そのもの**。2×2 の一方の脚 |
| `uniform5` | 固定 | 1.00 | 記述用。探索配分と平均 K が近い一様。**効き目の差が「K の大きさ」から来るのか「配分の形」から来るのかを分ける** |
| 探索 幅4 | report/07 の `slimpajama_w4_n16_seed<N>/search.json` を読む。**探索は走らせない** | 1.25〜1.59 | 2×2 のもう一方の脚 |

| ルーター | 中身 |
|---|---|
| `cmoe` | 固定 Top-K（対照） |
| `dynamic_cmoe` | 可変 K。score は現行のまま、しきい値1個で選ぶ。追加の積和は無い |

**1回の変換の中に2つのルーターを載せる。** carve も expert 重みも分割も共有され、
差は `MoE.gate` だけになる（report/19 と同じ組み方）。`MoE` が次層へ伝播させるのは
先頭の `cmoe` である — これも report/19 と同じで、揃えないと比べられない。

**「校正で選んだ一様」は水準に入れない。** slimpajama の3 seed で選ばれたのは
uniform6 / uniform5 / uniform6 で、uniform6 は x=A すなわち全層 Top-K=0、
つまり**可変Kが定義上まったく効かない**。seed ごとに水準の意味が変わる列を 2×2 に
混ぜない。この事実自体は考察に書く。

## 固定するもの

report/07 と report/19 の**両方**に揃える。食い違う箇所は report/07 側を採る
（配分探索の側が校正データに敏感なため）。

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（スパース率25%） |
| 校正 | slimpajama、carve 16 / fit 64 / validation 64 × 2048、seed 0 / 1 / 2 |
| 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001、router 正規化あり） |
| しきい値 | 層ごとに1個。fit の上で平均選択数が K になる分位点（`calibrate_threshold`）。床は 0 |
| ベンチ | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot・全件、batch 32 |
| PPL | wikitext2 / c4-new |
| ブートストラップ | 10,000回 / seed 20260813 |
| dense の基準 | report/04 の `result_logs/bench_h4/wikitext2_seed0/bench/dense.json` を取り込む |

## 走らせる前に決めたこと

6構成 × 8指標 × 2評価セットあるので、走らせてから拾えば必ず何か言える。先に書く。
判定の枠組みは report/07・08 と同じにする。

- **主要評価は PPL**（wikitext2 / c4-new）。ベンチの主要指標は `gold_nll` **1本**。
  `acc` / `acc_norm` / `margin` / `ref_kl` / `ref_agreement` は記述用で判定に使わない
- **判定**: 3 seed すべてで符号が改善方向、かつブートストラップ95%区間が0をまたがない
- **この実験の主語は交互作用**である。報告する主要な量は次の2本の差で、
  足し算になっているかを見る

      Δ_uniform3 = (uniform3 + 可変K) − (uniform3 + 固定K)
      Δ_探索     = (探索幅4 + 可変K) − (探索幅4 + 固定K)

- **`realized_load` は必須の記録**である。可変Kは予算を平均でしか守らないので、
  評価データの上で実際に走った expert 数が対照以下であることを示さなければ、
  この比較は成立しない。層ごとに K が違う探索配分の下でも守られているかは
  測っていない
- **言わないこと**: report/19（wikitext2 校正・S3A3E8）の数字との差には判定を
  当てない。別の校正・別の実行で、対応の取れる単位が無い。記述として並べるだけ
  にする
- **予測が当たること（可変Kの効き目が縮むこと）は失敗ではない。** それが観測結果で
  あり、原稿の組み立てを決める材料である

## 走らせる前に足した実装

**方式8 は fit 分割を要求するのに、slimpajama には分割器が無かった。**
report/07 の探索は `--router cmoe` だけだったので `load_calibration` の経路を
通っており、fit を一度も要求していない。可変Kを足した瞬間に `load_splits` の
経路へ移り、`SPLIT_SETS` に `wikitext2` しか登録されていないことで止まる。

`src/cmoe/data/slimpajama.py` に `splits()` を足した。要点は1つで、

* **carve は `calibration(model, seqlen, carve_count, seed)` と1トークンも
  違わない。** 種に札を足さず、避ける文書も無い経路をそのまま通す
  （`draw_component(part=None, exclude=())`）。report/07 が探した配分は、この
  carve から作られた分割の上でしか意味を持たないので、これは飾りではなく要件
  である。守れていることは token hash `37738f7c8d47`（seed 0）で確かめてある
  （`tests/test_data_slimpajama.py`、`CMOE_SLIMPAJAMA_INTEGRATION=1`）
* fit と validation は別の札の種で引き、**carve が採った文書を丸ごと避ける**。
  wikitext2 の分割は窓の重なりだけを避けるが、slimpajama は成分ごとの母集団が
  小さいので文書の単位で分ける
* validation も訓練側のシャードから引く。SlimPajama-6B の1シャードに
  train / validation の区別は無い。この実験は `--diagnostics` を使わないので
  読まれないが、3本組の型を満たすために作る

**したがって、この実験の `cmoe`（固定Top-K）の行は report/07 の同じ配分の行と
一致するはずである。** 一致しなければ、carve が動いている。照合する値
（`result_logs/bench_slimpajama_seed<N>/summary.json` から引いた PPL）:

| seed | 配分 | wikitext2 | c4-new |
|---|---|--:|--:|
| 0 | `uniform3` | 7.504356 | 10.160854 |
| 0 | `uniform5` | 7.589600 | 10.171712 |
| 1 | `uniform3` | 7.437901 | 9.972234 |
| 1 | `uniform5` | 7.418673 | 9.904146 |
| 2 | `uniform3` | 7.454722 | 10.091785 |
| 2 | `uniform5` | 7.330173 | 9.988071 |

探索配分（幅4）の行は report/07 の `beam 幅4` と同じ配分なので、そちらも一致する
はずである。

## 走らせ方

**本実行の前に必ず `--smoke` を通すこと。**

```bash
uv run python experiments/20_alloc_x_dynamick/run.py --smoke   # 2層・8問
uv run python experiments/20_alloc_x_dynamick/run.py --seed 0
uv run python experiments/20_alloc_x_dynamick/run.py --seed 1
uv run python experiments/20_alloc_x_dynamick/run.py --seed 2

# seed をまたいだまとめ（GPU 不要）
uv run python experiments/20_alloc_x_dynamick/summarize.py --seeds 0,1,2
```

探索の段は無い。report/07 の `search.json` から幅4 の配分を読むだけで、
**その seed の探索が無ければ止まる**（勝手に走らせない）。

## 見積もり

report/07 の測定段が 10構成で 108分（1構成あたり約11分、変換3回を含む）。
ここは6構成・変換3回なので **1 seed あたり約80分**、3 seed で **約4時間**。

## 保存されるもの

- `bench_alloc_dynamic_seed<N>/summary.json` … 全引数、PPL の塊ごとの平均 NLL を
  全件、ベンチの集計と対応のある比較、**`realized_load`**
- `bench_alloc_dynamic_seed<N>/bench/run<NNN>_<router>.json` … 問題ごと・選択肢
  ごとの生の対数尤度（fp32）。1構成 3.7MB
- `exp20_stages_seed<N>.json` … 段の一覧、読んだ配分ベクトル、測ったコードのコミット
