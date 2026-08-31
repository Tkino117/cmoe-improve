# 16 スパース率75%（N=8 / A=2）で、06・07 と同じ配分探索を回す

## 目的

スパース率を3点そろえる。

| | 探索が PPL で勝つか | ベンチで検出できるか |
|---|---|---|
| 25%（A=6、[report/07](../../report/07_slimpajama-alloc-3seeds.md)） | 勝つ（6箇所すべて） | できない |
| 50%（A=4、[report/08](../../report/08_sparsity50-alloc-3seeds.md)） | 勝つ | できない |
| **75%（A=2、ここ）** | ？ | ？ |

25% → 50% で動作点を倍に厳しくしてもベンチの効果量は伸びきらなかった。ここは
**もう一段だけ厳しくする**。A=2 では1トークンあたりに走る expert が8個中2個に
なり、捨てるぶんが 75% になる。配分の効き目はここで最大になるはずなので、
伸びなければ「どの動作点でも、この設計のベンチでは見えない」が3点で言える。

## 走らせる前に決めておくこと

**A=2 では一様が3本しかない。** x=0 は全層 routed、x=2 は全層 routing 無し、
その間に x=1 が1本あるだけである。層ごとの候補も3個で、探索が選べる余地は
25%・50%よりはっきり狭い。

そのため、**効果量が伸びなかったときに「配分は効かない」と「選べる余地が無い」
は分けられない**。report を書くときにこの2つを混ぜない。分けたければ N を上げて
（A/N を保ったまま粒度を細かくして）測り直すことになるが、それはこの実験の外で
ある。

## 固定するもの

A **以外は report/07・08 と1つも変えない**。

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | **8 / 2**（スパース率75%） |
| 校正 | slimpajama n=16 × 2048、seed 0 / 1 / 2 |
| 採点オラクル / 探索 | `suffix_kl` / beam 幅2・3・4 |
| 配分 | 一様 x=0〜2 と、探索の3幅 |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| ベンチ | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot 全件、batch 32 |
| PPL | wikitext2 / c4-new |
| ブートストラップ | 10,000回 / seed 20260813 |

**N は動かさない。** N を変えると expert 1個あたりのニューロン数（分割の粒度）
まで動いて、比べているものが「配分」でなくなる。

**MMLU はここでは測らない。** 3つのスパース率で揃っていないと並べられないので、
25% / 50% / 75% をまとめて [`experiments/17_mmlu_bench`](../17_mmlu_bench) が
別に足す。

dense の基準は測らない。report/04 の
`result_logs/bench_h4/wikitext2_seed0/bench/dense.json` を取り込む（dense は
変換していないモデルなので、**A にも**校正にも seed にも配分にも依らない）。

## 判定

report/07・08 と同じ枠組みにする。動作点を変えた効果を見る実験で、読み方まで
変えたら比べられない。

- **主要評価は PPL**（wikitext2 / c4-new）。ベンチの主要指標は `gold_nll`
  **1本だけ**。`acc` / `acc_norm` / `margin` / `ref_kl` / `ref_agreement` は
  記述用に出すが、判定には使わない
- **主役の幅は4**（幅2・3 は「幅が効く」ことの記述用）
- **判定**: 3 seed すべてで符号が改善方向、かつブートストラップ95%区間が
  0をまたがない → 「改良」と書く
- **対等な比較は「校正で選んだ一様」との差**である
- 25% / 50% の数字は**記述として並べるだけ**にする。別の実行・別の動作点で、
  対応の取れる単位が無い

## 対照を3つ置く

06・07 と同じ3つ。固定対照だけが A に連動して変わる。

| 対照 | 選び方 | 読み方 |
|---|---|---|
| `uniform1` | 固定 | 選び方に評価指標が入らない。A の半分＝ shared と routed が半々で、A=6 の `uniform3`・A=4 の `uniform2` と同じ立ち位置 |
| **校正で選んだ一様** | 一様3本を探索と同じ `suffix_kl` で採点した最良 | 両側とも「校正だけで選んだ1本」。**対等な比較はこれ** |
| 評価指標で最良の一様 | その指標を見てから3本の最良 | 探索に不利な側の下限 |

## 走らせ方

**本実行の前に必ず `--smoke` を通すこと。**

```bash
uv run python experiments/16_sparsity75/run.py --smoke   # 2層・2本・8問
uv run python experiments/16_sparsity75/run.py --seed 0  # 約2.4時間
uv run python experiments/16_sparsity75/run.py --seed 1
uv run python experiments/16_sparsity75/run.py --seed 2

# seed をまたいだまとめ（GPU 不要）
uv run python experiments/16_sparsity75/summarize_seeds.py --seeds 0,1,2
```

**seed 0 も探索から回す。** A が違えば探索の結果も違うので、A=6 / A=4 の出力は
使えない。出力先の名前に `a2` が入っているのはそのためである。

## 段と、済みの判定

1 seed は3段。**済んだ段は飛ばす**。印はファイルの存在ではない（開始直後から
真になる）ので、段ごとに中身を見る。

| 段 | コマンド | 出力 | 済みの印 |
|---|---|---|---|
| 1 探索 | `cmoe search` 幅2/3/4 | `slimpajama_a2_w<W>_n16_seed<N>/search.json` | `allocation` と `recheck` がある |
| 2 採点 | `cmoe score` 一様 x=0..2 | `score_uniform_slimpajama_a2_n16_seed<N>/score.json` | `best` があり、採点した本数が合う |
| 3 測定 | `cmoe run --bench` 6構成 | `bench_slimpajama_a2_seed<N>/summary.json` | `summary` と `bench_summary` があり、構成数が合い、`failures` が無い |

印の無いディレクトリは `*.partial<N>` へ退避してから引き直す。別の実験の出力
（引数が食い違う）だった場合は退避もせずに止まる。

## 見積もり（report/07・08 の実測からの外挿）

層ごとの候補が x=0..6 の7個から x=0..2 の**3個**に減るので、探索の評価回数は
3/7 になる。測る構成も 10 → 6 に減る。

| 段 | A=6 の実測 | A=2 の見込み |
|---|---|---|
| 1 探索 幅2 / 3 / 4 | 35 + 53 + 69 = 157分 | 15 + 23 + 30 = **68分** |
| 2 一様の採点（1本 528 layer forwards） | 22分（7本） | **10分**（3本） |
| 3 測定（1構成 約11分） | 108分（10構成） | **66分**（6構成） |
| | 4.8時間 | **約2.4時間 / seed** |

seed 3本で**約7時間**。

## 保存されるもの

- `slimpajama_a2_w<W>_n16_seed<N>/search.json` … 全層 × 全親 × 全候補のスコア、
  落選候補も含む枝の系譜、勝った配分の測り直し、校正の成分ごとの内訳
- `score_uniform_slimpajama_a2_n16_seed<N>/score.json` … 一様3本の score と層ごとの内訳
- `bench_slimpajama_a2_seed<N>/summary.json` … 全引数、PPL の塊ごとの平均 NLL を全件、
  ベンチの集計と対応のある比較
- `bench_slimpajama_a2_seed<N>/bench/run<NNN>_cmoe.json` … 問題ごと・選択肢ごとの
  生の対数尤度（fp32）
- `exp16_stages_seed<N>.json` … 段の一覧と、測ったコードのコミット
- `exp16_seeds/seeds.md` / `seeds.json` … seed をまたいだまとめ

指標を後から足すのに測り直しは要らない。`cmoe.eval.bench.load_samples()` で
読み戻せば、`bench_stats` の全指標が GPU なしで出る。
