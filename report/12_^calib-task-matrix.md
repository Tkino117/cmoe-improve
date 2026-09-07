# 12 校正タスクと評価タスクの対応（4×4）

2026-08-29 / 実験 `experiments/11_calib_task_matrix/` /
出力 `result_logs/bench_calibtask_{piqa,winogrande,arc,arc_easy,arc_challenge,hellaswag,flanv2}_seed{0,1,2}` /
対照は [report/11](11_bench-calibration.md) の5タスク混合
（`result_logs/bench_benchtrain_seed{0,1,2}`）と
[report/04](04_alloc-search-benchmark.md) の wikitext2 校正
（`result_logs/bench_h4/wikitext2_seed{0,1,2}`）

## 目的

report/11 は「校正を選択問題の train split に替えると効く」ことを示した
（`acc` +0.0361、dense との差の 54% が埋まった）。だがそこで使った校正は5
タスクを混ぜたもので、効いているのが**測るタスクそのもの**なのか、それとも
**選択問題らしい形**であって中身は問わないのか、が分かれていない。

校正を1タスクに絞った系を作り、それぞれを全タスクで測る。1回の測定が5タスクを
採点するので、行を走らせれば表が埋まる。

## 設定

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N（expert 数）/ A（1トークンあたり） | 8 / 6（スパース率25%） |
| 校正 | 8本 × 2048トークン（= 16,384トークン。**行が変わっても総量は同じ**）、seed 0 / 1 / 2 |
| 配分 | `uniform4` に固定 |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| ベンチ | PIQA / WinoGrande / ARC / HellaSwag、0-shot・全件、batch 32 |
| PPL | wikitext2 / c4-new |
| ブートストラップ | 10,000回 / seed 20260813 |
| dense の基準 | report/04 のものを取り込み（`bench_h4/wikitext2_seed0/bench/dense.json`） |
| コード | `b5f5013` + 未コミット（`benchtrain` のタスク部分集合、`cmoe.data.flanv2`、`experiments/11_calib_task_matrix/`。作業ツリーには実験10の途中作業も同居している） |
| 所要 | 単一タスク4行 + ARC-e / ARC-c を分けた2行 + `flanv2` の計21構成、1構成あたり 9.5〜9.8分（計 3.4時間）。失敗0 |

**探索の段は無い。** report/11 で、配分を替えて動く幅（±0.003）は校正を替えて
動いた幅（+0.036）の10分の1以下だった。行ごとに探索すると配分が交絡するので、
`uniform4` に固定して校正だけを動かす。

**ARC-e と ARC-c は1タスクとして扱う。** 同じ AI2-ARC を、当時のベースライン
2種が解けたかどうかで割ったもので、出題元・形式・語彙は同じ、問題は重複しない。
校正データとしても交換可能だった（後述）。ExpertWeaver Table 8 の数え方に
合わせて `ARC` 1列・1行にする。列は保存済みの尤度を突き合わせて束ねたもの
（ARC-e 2,376問 + ARC-c 1,172問 = 3,548問）、行は測り直しである（校正トークン
が変わるので束ねられない。8本を ARC-e 4本 / ARC-c 4本に配る）。

**`flanv2` の行は ExpertWeaver の校正である。** この軸（校正データを何にするか）
での既存の最良は ExpertWeaver（arXiv:2602.15521）の Flan 多タスク校正なので、
同じモデル・同じトークン量・同じ配分で1行として並べる。写したのは彼らの
Appendix H Table 8 に実際に並んでいる 39 データセットのうち、タスク別ミラー
（`Muennighoff/flan`、FLAN 2021・10テンプレート）から引ける **36 タスク**である
（欠: duorc / xsum / siqa / winogrande / race / winogender。wmt14 の en-de・en-ro
は wmt16 の同言語対で代替）。**評価タスクのうち PIQA / HellaSwag / ARC はこの
校正に入り、WinoGrande だけが入らない。** Flan-v2 は公開データセットを
instruction 形式に整形した混合なので、これらの train 側が校正に入ることは元の
設定どおりである。

なお論文の中でタスク数が揃っていない — 本文 §2.2.2 は 48 タスク、§3 は 42
タスク、Appendix G は「各タスク10サンプル」、Table 8 の実列挙は 39 データセット
である。ここが写したのは実列挙の方である。系列長やトークン総量は書かれて
いないので、量は写せない。**量ではなく中身を写し、量は他の行に揃えた。**

**`wikitext2` の行は現行 CMoE そのものである。** 校正を差し替えていない出発点
で、表の外側にある基準として置いてある。`mix(5)` の行が report/11 である。

### 行ごとの校正の中身（seed 0）

引くのは各タスクの train split だけで、採点に使う split は1問も入らない。

| 行 | 使った問題数 / train 全体 | 連結したトークン数 | 16,384トークンが占める割合 |
|---|---|---|---|
| `piqa` | 3,400 / 16,113 | 148,790 | 11% |
| `winogrande` | 5,033 / 40,398 | 137,595 | 12% |
| `arc` | 1,729 / 2,251（ARC-e）+ 1,119 / 1,119（ARC-c） | 63,627 + 47,196 | 15% |
| `hellaswag` | 1,478 / 39,905 | 131,606 | 12% |

ARC は train split をほぼ使い切っており（ARC-c は全問）、窓が母集団に占める
割合が他の行より大きい。

## 結果

### 4×4（`acc`、3 seed 平均）

● は対角（校正と評価が同じタスク）、* は列の最良、◎ は両方。

| 校正 \ 評価 | PIQA | Wino | ARC | HSwag | macro |
|---|---|---|---|---|---|
| `piqa` | **0.7713**◎ | 0.6469 | 0.5996 | 0.5283 | 0.6365 |
| `winogrande` | 0.7568 | 0.6575● | 0.5701 | 0.5198 | 0.6260 |
| `arc` | 0.7608 | 0.6596 | **0.6160**◎ | 0.5162 | 0.6381 |
| `hellaswag` | 0.7657 | 0.6501 | 0.5920 | **0.5390**◎ | 0.6367 |
| `flanv2`（ExpertWeaver） | 0.7425 | 0.6585 | 0.5884 | 0.5041 | 0.6234 |
| `mix(5)`（report/11） | 0.7670 | **0.6619*** | 0.6146 | 0.5329 | **0.6441** |
| `wikitext2`（現行 CMoE） | 0.7356 | 0.6406 | 0.5704 | 0.4955 | 0.6105 |
| dense（変換なし） | 0.7802 | 0.6953 | 0.6485 | 0.5708 | 0.6737 |

### `gold_nll`（3 seed 平均、小さいほど良い）

| 校正 \ 評価 | PIQA | Wino | ARC | HSwag | macro |
|---|---|---|---|---|---|
| `piqa` | **2.0652**◎ | 0.7283 | 1.9962 | 10.9247 | 3.9286 |
| `winogrande` | 2.1752 | 0.7223● | 2.1170 | 11.2121 | 4.0566 |
| `arc` | 2.1788 | **0.6991*** | **1.8955**◎ | 11.3131 | 4.0216 |
| `hellaswag` | 2.1049 | 0.7047 | 2.0280 | **10.5607**◎ | **3.8496** |
| `flanv2` | 2.2638 | 0.7013 | 2.0686 | 11.7457 | 4.1948 |
| `mix(5)` | 2.0836 | 0.7055 | 1.9376 | 10.7112 | 3.8595 |
| `wikitext2` | 2.3464 | 0.7315 | 2.1071 | 11.9886 | 4.2934 |
| dense | 1.9185 | 0.6288 | 1.5893 | 8.8646 | 3.2503 |

### 行効果と列効果を引いた残差（`acc`）

行ごとに「どの列にも効く強さ」が違うので、加法モデル（行平均 + 列平均 − 全体
平均）からの残差で見る。

| 校正 \ 評価 | PIQA | Wino | ARC | HSwag |
|---|---|---|---|---|
| `piqa` | **+0.0055**● | −0.0088 | +0.0030 | +0.0003 |
| `winogrande` | +0.0015 | **+0.0123**● | −0.0160 | +0.0023 |
| `arc` | −0.0066 | +0.0023 | **+0.0178**● | −0.0134 |
| `hellaswag` | −0.0003 | −0.0058 | −0.0048 | **+0.0108**● |

対角の残差の平均は **+0.0116**、非対角は **−0.0039**。**残差では対角が 4/4 列
で最良**である。Wino 列で `arc` 校正が生の値で上に来るのは、`arc` 行がどの列に
も強いこと（行平均が単一4行で最大）で説明が付く。

## 保存されているもの

- 測定: `result_logs/bench_calibtask_<行>_seed{0,1,2}/`（`summary.json` と、
  問題ごと・選択肢ごとの生の対数尤度 `bench/*.json`、校正トークンのハッシュと
  引いた問題の番号）
- 段の一覧: `result_logs/exp11_stages_seed{0,1,2}.json`（最後に走らせた行のもの）

生の対数尤度が全件残っているので、別の指標を足すのも、ARC のように列を束ねる
のも、測り直しは要らない。

再現:

```bash
uv run python experiments/11_calib_task_matrix/run.py --seed 0   # 1,2 も同様
uv run python experiments/11_calib_task_matrix/run.py --seed 0 --calibs benchtrain:arc
uv run python experiments/11_calib_task_matrix/run.py --seed 0 --calibs flanv2
uv run python experiments/11_calib_task_matrix/summarize.py --seeds 0,1,2
uv run python experiments/11_calib_task_matrix/summarize.py --seeds 0,1,2 --arc split

# ExpertWeaver 校正との対応のある比較（マクロ、ARC は分けたまま）
uv run python experiments/compare_runs.py \
    --base result_logs/bench_h4/wikitext2_seed0 \
    --base result_logs/bench_h4/wikitext2_seed1 \
    --base result_logs/bench_h4/wikitext2_seed2 --base-alloc uniform4 \
    --cand result_logs/bench_calibtask_flanv2_seed0 \
    --cand result_logs/bench_calibtask_flanv2_seed1 \
    --cand result_logs/bench_calibtask_flanv2_seed2 --cand-alloc uniform4
```
