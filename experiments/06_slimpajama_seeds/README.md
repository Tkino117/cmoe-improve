# 06 slimpajama 校正の配分探索を、校正 seed 3本で回す

## 目的

[report/06](../../report/06_slimpajama-alloc-benchmark.md) は slimpajama n=16 の
**seed 1本**で、探索配分が PPL で**どの一様配分よりも良い**ことを見た。
[report/04](../../report/04_alloc-search-benchmark.md)（wikitext2 / c4 校正、
seed 3本）では逆に、探索配分は最良の一様を**どの指標でも上回らなかった**。

ここで見るのは「seed 0 の観測が他の seed でも出るか」だけである。

**言わないこと。** report/04 との差の原因は帰属しない。あちらは校正コーパスが
wikitext2 / c4 で本数が n=8、こちらは slimpajama の n=16 と、2つが同時に違う。

## 固定するもの

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（**x をどう振っても1トークンあたりに走る expert 数は同じ**＝スパース率25%） |
| 校正 | slimpajama n=16 × 2048、seed 0 / 1 / 2 |
| 採点オラクル / 探索 | `suffix_kl` / beam 幅2・3・4 |
| 配分 | 一様 x=0〜6 と、探索の3幅 |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| ベンチ | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag、0-shot 全件、batch 32 |
| PPL | wikitext2 / c4-new |
| ブートストラップ | 10,000回 / seed 20260813 |

dense の基準は測らない。report/04 の
`result_logs/bench_h4/wikitext2_seed0/bench/dense.json` を取り込む（dense は
変換していないモデルなので、校正にも seed にも配分にも依らない）。

## 走らせる前に決めたこと

走らせてから指標を選べば、18箇所の中の1つは必ず何か言う。だから先に書く。

- **主要評価は PPL**（wikitext2 / c4-new）。ベンチの主要指標は `gold_nll`
  **1本だけ**。`acc` / `acc_norm` / `margin` / `ref_kl` / `ref_agreement` は
  記述用に出すが、判定には使わない
- **主役の幅は4**（seed 0 で最良、幅3 とは 28/32 層が一致）。幅2・3 は「幅が
  効く」ことの記述用
- **判定**: 3 seed すべてで符号が改善方向、かつブートストラップ95%区間が
  0をまたがない → 「改良」と書く。PPL だけが満たしてベンチが null なら
  「PPL では改良、ベンチでは検出できず」と書く
- **効き目の限界**: ブートストラップは問題や塊を再抽出するだけで、seed ごとに
  配分ベクトルが変わるぶんの分散は3本では測れない。そこは per-seed の符号一致
  （3/3 か）で補う。ベンチの効果量は seed 0 で `acc` +0.005 程度、区間の半幅が
  ±0.008 で、3本にしても ±0.005 前後までしか縮まない。**ベンチで区間が
  0をまたいだままなのは、この設計では失敗ではない**

## 対照を3つ置く

探索は**校正データだけを見て**配分を1本選ぶ。その相手を「評価指標を見てから
一様7本の最良」で選ぶと、対照の側にだけたまたまの上振れが乗る。report/04 で
最良の一様が seed と指標ごとに変わった（`acc` は uniform3/5/3、`acc_norm` は
uniform6/6/6 …）のがその現れである。

| 対照 | 選び方 | 読み方 |
|---|---|---|
| `uniform3` | 固定 | 選び方に評価指標が入らない。report/04・06 と同じ基準 |
| **校正で選んだ一様** | 一様7本を探索と同じ `suffix_kl` で採点した最良 | 両側とも「校正だけで選んだ1本」。**対等な比較はこれ** |
| 評価指標で最良の一様 | その指標を見てから7本の最良 | 探索に不利な側の下限 |

2番目のために `cmoe score` を足した。探索を1回も呼ばず、与えた配分を
`score_allocation` で頭から採点する — 探索が勝った配分を測り直すのと同じ関数を
通るので、出る数は `search` の score と直接比べられる。

## 走らせ方

**本実行の前に必ず `--smoke` を通すこと。**

```bash
uv run python experiments/06_slimpajama_seeds/run.py --smoke   # 2層・7本・8問
uv run python experiments/06_slimpajama_seeds/run.py --seed 1  # 約4.8時間
uv run python experiments/06_slimpajama_seeds/run.py --seed 2

# seed をまたいだまとめ（GPU 不要）
uv run python experiments/06_slimpajama_seeds/summarize_seeds.py --seeds 0,1,2
```

seed 0 は探索とベンチが済んでいる（report/05・06 の出力）ので、`--seed 0` で
走るのは一様の採点だけである。

## 段と、済みの判定

1 seed は3段。**済んだ段は飛ばす**。印はファイルの存在ではない（開始直後から
真になる）ので、段ごとに中身を見る。

| 段 | コマンド | 出力 | 済みの印 |
|---|---|---|---|
| 1 探索 | `cmoe search` 幅2/3/4 | `slimpajama_w<W>_n16_seed<N>/search.json` | `allocation` と `recheck` がある |
| 2 採点 | `cmoe score` 一様 x=0..6 | `score_uniform_slimpajama_n16_seed<N>/score.json` | `best` があり、採点した本数が合う |
| 3 測定 | `cmoe run --bench` 10構成 | `bench_slimpajama_seed<N>/summary.json` | `summary` と `bench_summary` があり、構成数が合い、`failures` が無い |

印の無いディレクトリは `*.partial<N>` へ退避してから引き直す（書きかけを「済み」と
取り違えない）。ただし**別の実験の出力**（引数が食い違う）だった場合は退避もせずに
止まる。`cmoe run` に途中再開は無いので、1構成でも失敗した段は全構成を引き直す。

## 見積もり（seed 0 の実測からの積み上げ）

| 段 | 内訳 | 実測 |
|---|---|---|
| 1 | 探索 幅2 / 3 / 4 | 35 + 53 + 69 = 157分 |
| 2 | 一様7本の採点（1本 528 layer forwards） | 約18分 |
| 3 | 10構成 × 11.1分（変換2.2 + PPL 1.4 + ベンチ 7.6） | 111分 |
| | **1 seed** | **約4.8時間** |

seed 1・2 で約9.5時間、これに seed 0 の段2（18分）を足す。

## 保存されるもの

- `slimpajama_w<W>_n16_seed<N>/search.json` … 全層 × 全親 × 全候補のスコア、
  落選候補も含む枝の系譜、勝った配分の測り直し、校正の成分ごとの内訳
- `score_uniform_slimpajama_n16_seed<N>/score.json` … 一様7本の score と層ごとの内訳
- `bench_slimpajama_seed<N>/summary.json` … 全引数、PPL の塊ごとの平均 NLL を全件、
  ベンチの集計と対応のある比較
- `bench_slimpajama_seed<N>/bench/run<NNN>_cmoe.json` … 問題ごと・選択肢ごとの
  生の対数尤度（fp32）。1構成 3.7MB
- `exp06_stages_seed<N>.json` … 段の一覧と、測ったコードのコミット
- `exp06_seeds/seeds.md` / `seeds.json` … seed をまたいだまとめ

指標を後から足すのに測り直しは要らない。`cmoe.eval.bench.load_samples()` で
読み戻せば、`bench_stats` の全指標が GPU なしで出る。
