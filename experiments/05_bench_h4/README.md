# 05 H4 — 探索した配分を、選択問題ベンチマークで測る

## 目的

[report/03](../../report/03_alloc-search-vs-uniform-3seeds.md) は PPL だけで
「探索した配分はどの一様配分よりも良い」を見た。ここは**同じ配分・同じ seed・
同じ校正**をベンチマークにかける（[docs/03](../../docs/03_expermient-design.md) の H4）。

タスクは PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag の 0-shot。CMoE 最新版
（ACL 2026, arXiv:2502.04416）Table 1 と同じ並びで、ExpertWeaver
（arXiv:2602.15521）Table 2 との共通部分でもある。どちらの論文も主表の動作点は
スパース率25%で、この実験の N=8 / A=6 がちょうどそれに当たる。

## 何を見るのか

**正答率だけでは差が出ない見込みである。** PPL の差は 0.03（1トークンあたり
0.004 nat）で、選択問題の正答率はマージンの符号しか見ないため、この程度のずれで
符号をまたぐ問題はごく一部しかない。1万問でも標本誤差は ±1ポイント弱ある。

そこで測るのは6指標で、**問題ごと・選択肢ごとの生の対数尤度**を全件残す。

| 指標 | 中身 | 向き |
|---|---|---|
| `acc` / `acc_norm` | 論文の表に載る正答率 | 大 |
| `gold_nll` | 選択肢だけで softmax した正解確率の −log | 小 |
| `margin` | 正解 − 最良の不正解の対数尤度 | 大 |
| `ref_kl` | dense との選択肢上の分布の KL | 小 |
| `ref_agreement` | dense と同じ選択肢を選んだか | 大 |

**この実験が答えるのは「探索した配分は下流タスクでも良いか」であって、
「正答率で有意差が出るか」ではない。** `acc` の区間が 0 をまたいだまま
`gold_nll` の区間がまたがない、という結果はこの設計では失敗ではない。

## 固定するもの

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6（全構成で推論コストは同一＝スパース率25%） |
| 校正 | 8本 × 2048トークン、seed 0 / 1 / 2 |
| 配分 | 一様 x=0〜6 と、report/03 の beam 幅2 / 3 / 4 |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| ベンチ | 5タスク 0-shot 全件、batch 32、lm-eval-harness 0.4.12 |
| PPL | wikitext2 / c4-new（report/03 と6桁一致するかの点検を兼ねる） |
| ブートストラップ | 10,000回 / seed 20260813 |

**探索は走らせない。** report/03 の18本は
`result_logs/h1h3_pilot{,_seed1,_seed2}/h{2,3}_search_w{2,3,4}/search.json` に
残っており、そこからベクトルを読んで `--alloc` に渡す。探索がこの実験の所要時間の
大半だったので、これで12時間ぶんが要らなくなる。読むときは校正・幅・seed が
一致することを確かめる。

**dense の基準は1回だけ測って使い回す。** dense は seed にも配分にも校正にも
依らないので、段ごとに測り直すのは丸損である（8.8分 × 5 = 44分）。最初の段が測り、
残りは `--bench-reference` で取り込む。取り違えを防ぐため、モデル名・タスク・
shot 数・limit が一致しなければ取り込まずに止まる。

## 走らせ方

**本実行の前に必ず `--smoke` を通すこと。**

```bash
uv run python experiments/05_bench_h4/run.py --smoke    # 2層・8問。経路の確認
uv run python experiments/05_bench_h4/run.py --seed 0   # seed 0（約3.9時間）
uv run python experiments/05_bench_h4/run.py --seed 1
uv run python experiments/05_bench_h4/run.py --seed 2

# seed をまたいだまとめ（GPU 不要）
uv run python experiments/05_bench_h4/summarize_seeds.py --seeds 0,1,2
```

出力は `result_logs/bench_h4/<校正>_seed<N>/`。

## 見積もり（実測値からの積み上げ）

| 内訳 | 実測 |
|---|---|
| 変換（32層） | 1.3分 |
| PPL（wikitext2 + c4-new） | 1.4分 |
| ベンチ5タスク全件（batch 32） | 8.8分 |
| **1構成** | **11.5分** |

配分10種 × 校正2種 × seed 3本 = 60構成 → **約11.5時間**（+ dense 8.8分を1回）。
seed 1本あたり約3.9時間。

## 途中結果の残り方

段（1校正 × 1seed）ごとに別ディレクトリで、**完了印がある段は飛ばす**。印は
「`summary` と `bench_summary` があり、構成数が合い、`failures` が無い」こと。
印の無いディレクトリは `*.partial<N>` へ退避してから引き直す（書きかけを
「済み」と取り違えない）。ただし**別の実験の出力**（引数が食い違う）だった場合は
退避もせずに止まる。

`cmoe run` に途中再開は無いので、1構成でも失敗した段は次の起動で全構成を引き直す。

## 保存されるもの

- `<段>/summary.json` … 全引数、PPL の塊ごとの平均 NLL 全件、ベンチの集計と
  対応のある比較
- `<段>/bench/run<NNN>_cmoe.json` … **問題ごと・選択肢ごとの生の対数尤度**
  （fp32）、`doc_hash`、正解番号、選択肢の文字数、lm-eval 自身の集計値。1構成 3.7MB
- `<段>/bench/dense.json` … 基準。取り込んだ段には `dense_from.txt` も残る
- `stages.json` … 段の一覧と、測ったコードのコミット

指標をあとから足すのに測り直しは要らない。`cmoe.eval.bench.load_samples()` で
読み戻し、`bench_stats` の全指標が GPU なしで出る。**保存ファイルだけから
信頼区間まで含めて集計を再構成できることは実測で確認してある。**
