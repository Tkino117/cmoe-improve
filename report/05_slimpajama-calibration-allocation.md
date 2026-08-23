# 05 混合コーパスで校正して配分を探す

2026-08-23 / 出力 `result_logs/slimpajama_w{2,3,4}_n16_seed0`、
`result_logs/wikitext2_w2_n16_seed0`

## 目的

校正データを wikitext2（＝Wikipedia 単体）から、事前学習コーパスの混合物に
差し替えて配分を探す。出てきた配分ベクトルを wikitext2 のものと並べる。

## 使ったデータセット

`DKYoon/SlimPajama-6B`（`cerebras/SlimPajama-627B` の 10% サンプル）の
**1シャード**を固定して使う。

| | |
|---|---|
| repo | `DKYoon/SlimPajama-6B` |
| シャード | `data/train-00000-of-00048-ab2b35705f029d94.parquet`（292MB） |
| 文書数 / 文字数 | 114,355 / 492,988,695 |

本家 `cerebras/SlimPajama-627B` と `togethercomputer/RedPajama-Data-1T-Sample` は
2026-08 時点で認証なしには引けない。`togethercomputer/RedPajama-Data-1T` は
引けるが、Books スライス（`urls/book.txt`）が削除されている。

### 成分の内訳と、引いた本数

成分名は `meta.redpajama_set_name`。本数は**各成分に最低1本、残りをこの
シャードで実測した文字数比の最大剰余法**で配る（`cmoe.data.slimpajama.allocate`）。

| 成分 | 文書数 | 文字数 | 比率 | 引いた本数（n=16） |
|---|---|---|---|---|
| CommonCrawl | 36,264 | 268.4M | 54.45% | 6 |
| C4 | 62,534 | 140.6M | 28.52% | 4 |
| GitHub | 4,124 | 22.4M | 4.55% | 2 |
| ArXiv | 285 | 17.8M | 3.62% | 1 |
| Book | 38 | 16.9M | 3.44% | 1 |
| Wikipedia | 5,251 | 12.6M | 2.55% | 1 |
| StackExchange | 5,859 | 14.1M | 2.87% | 1 |

### 窓の切り方

成分ごとに、文書をシャッフルして必要量（窓の総面積の8倍）だけ連結し、
**重ならない**開始位置から 2048トークンの窓を切る。

c4 の引き方（seqlen 以上の文書だけ採る）は使っていない。このシャードの文書長の
中央値は C4 が 1,235文字・Wikipedia が 1,128文字で、2048トークンに届く文書は
C4 で 3.9% しか無いため。

wikitext2 の引き方（`draw_starts`、重なりを許す）も使っていない。成分ごとに
必要量だけを連結するので母集団が小さく、そのまま引くと n=16 の CommonCrawl で
78%、いずれかの成分で 93% の確率で窓が重なる（重複トークン約10%。wikitext2 の
n=16 は 0.7%）。

### 引けた校正セット（seed 0）

`slimpajama-train-calib` [16, 2048]、32,768トークン、`37738f7c8d47`。

| 成分 | 本数 | 連結した母集団 | 文字/トークン | 使った文書 |
|---|---|---|---|---|
| CommonCrawl | 6 | 133,908 tok | 3.80 | 61 |
| C4 | 4 | 66,350 tok | 3.96 | 114 |
| GitHub | 2 | 58,241 tok | 2.26 | 24 |
| ArXiv | 1 | 25,069 tok | 3.27 | 1 |
| Book | 1 | 23,046 tok | 3.65 | 1 |
| Wikipedia | 1 | 20,872 tok | 3.18 | 29 |
| StackExchange | 1 | 19,556 tok | 3.38 | 25 |

ArXiv と Book は1文書が必要量を超えるため、それぞれ1文書から切っている。

## 設定

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N（expert 数）/ A（1トークンあたり） | 8 / 6 |
| 採点オラクル | `suffix_kl` |
| 探索 | beam。slimpajama は幅2 / 3 / 4、対照は幅2 |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| seqlen | 2048 |
| seed | 0 |
| 環境 | RTX PRO 5000 Blackwell、Python 3.11.14 / PyTorch 2.8.0+cu128 / transformers 4.47.1 / datasets 2.21.0 |
| コード | `cdc90c6`。slimpajama 幅2 と wikitext2 n=16 は同内容の未コミット状態で測った。wikitext2 n=8 は report/01 当時のコード |

対照を2つ置く。どちらも幅2 だけである。

- **wikitext2 n=8** — report/01・report/03 の既存測定をそのまま引いた
  （`result_logs/calib_seed_variance_suffix_kl_w2/search_seed0`）
- **wikitext2 n=16** — 本数だけを揃えた対照。今回引き直した。
  n を変えると窓の引き方が変わるので、n=8 とは別物である

```bash
for width in 2 3 4; do
  uv run cmoe search --calib slimpajama --nsamples 16 --oracle suffix_kl \
    --search beam --width $width --seed 0 \
    --out result_logs/slimpajama_w${width}_n16_seed0
done
uv run cmoe search --calib wikitext2 --nsamples 16 --oracle suffix_kl \
  --search beam --width 2 --seed 0 --out result_logs/wikitext2_w2_n16_seed0
```

## 結果

### 出てきた配分（層0から順の x）

```
wikitext2  n=8   幅2  4,4,5,6,6,4,6,6,6,1,3,5,0,3,4,4,3,5,5,6,3,3,5,6,2,5,4,2,6,6,1,2
wikitext2  n=16  幅2  3,5,6,6,4,5,5,6,4,2,3,6,0,2,5,3,5,6,6,6,6,6,5,6,6,6,5,4,4,4,4,2
slimpajama n=16  幅2  6,6,6,6,6,6,5,5,6,0,4,3,3,3,4,5,4,6,6,4,4,6,2,6,4,6,6,6,5,3,4,6
slimpajama n=16  幅3  4,6,6,5,6,6,5,6,4,2,4,2,3,3,4,4,5,6,5,4,6,4,5,5,3,6,6,6,5,2,4,2
slimpajama n=16  幅4  4,6,6,5,6,6,5,6,4,2,4,2,3,3,4,4,5,6,5,4,6,4,5,5,3,4,6,6,4,2,2,4
```

| 校正 | 幅 | 本数 | トークン | トークンのハッシュ | 平均 x | score (`suffix_kl`) | コスト | 評価回数 | 測り直しの差 | 所要 |
|---|---|---|---|---|---|---|---|---|---|---|
| wikitext2 n=8 | 2 | 8 | 16,384 | `65a051f78ade` | 4.0938 | 1.882924e-01 | 7,168 | 441 | 0.000e+00 | 21.1 分 |
| wikitext2 n=16 | 2 | 16 | 32,768 | `5453d5d910b0` | 4.5625 | 1.947526e-01 | 7,168 | 441 | 0.000e+00 | 35.9 分 |
| slimpajama n=16 | 2 | 16 | 32,768 | `37738f7c8d47` | 4.7500 | 1.836402e-01 | 7,168 | 441 | 0.000e+00 | 35.2 分 |
| slimpajama n=16 | 3 | 16 | 32,768 | `37738f7c8d47` | 4.5000 | 1.818547e-01 | 10,640 | 658 | 0.000e+00 | 53.0 分 |
| slimpajama n=16 | 4 | 16 | 32,768 | `37738f7c8d47` | 4.4062 | 1.805907e-01 | 14,112 | 875 | 0.000e+00 | 69.3 分 |

**score は校正をまたいで比べられない。** 各行はそれぞれ自分の校正トークンの上で
測った KL である（slimpajama の3行は同じトークンなので、その3行どうしは比べられる）。

コストの単位は `layer_forwards`。機械の非決定性の床はどの探索でも 0.000e+00 で、
5本すべて、勝った配分を頭から測り直した差は 0.000e+00 だった。失敗した探索は無い。

### 配分どうしの一致

**校正のあいだ**（どれも幅2）

| 比較 | 一致した層 | 平均 \|差\| |
|---|---|---|
| wikitext2 n=8 vs wikitext2 n=16 | 8/32 | 1.22 |
| wikitext2 n=16 vs slimpajama n=16 | 9/32 | 1.31 |
| wikitext2 n=8 vs slimpajama n=16 | 6/32 | 1.53 |

**幅のあいだ**（どれも slimpajama n=16）

| 比較 | 一致した層 | 平均 \|差\| |
|---|---|---|
| 幅2 vs 幅3 | 16/32 | 0.81 |
| 幅3 vs 幅4 | 28/32 | 0.22 |
| 幅2 vs 幅4 | 13/32 | 0.91 |

幅3 と幅4 は層0〜24 が同一で、分かれるのは層25 以降の4層である。

## 保存されているもの

- `result_logs/slimpajama_w{2,3,4}_n16_seed0/search.json`、
  `result_logs/wikitext2_w2_n16_seed0/search.json` — 全層 × 全親 × 全候補
  （x=0〜6）のスコア、落選候補も含む枝の系譜、全引数、勝った配分の測り直し
- 同 `search.json` の `calibration.components` — 成分ごとの本数・開始位置・
  使った文書の番号・連結した母集団のトークン数・実測の文字/トークン・
  シャード名。ここから同じトークン列を引き直せる
- `result_logs/slimpajama_smoke/` — 2層だけの動作確認
- `result_logs/calib_seed_variance_suffix_kl_w2/search_seed0/` — wikitext2 n=8
  の既存測定（report/01）

## 考察・結論

本レポートでは書かない。素の観測結果のみを記録した。
