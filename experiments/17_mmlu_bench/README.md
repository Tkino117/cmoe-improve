# 17 既に測った構成を、MMLU でもう一度測る

## 目的

MMLU（test 14,042問）はベンチに足したばかりで、slimpajama 校正の測定にはどれも
入っていない。25% と 50% に足す。

| スパース率 | 元の実行 | 構成数 | MMLU の出力 |
|---|---|---|---|
| 25%（A=6） | `bench_slimpajama_seed<N>`（report/07） | 10 | `bench_mmlu_slimpajama_seed<N>` |
| 50%（A=4） | `bench_slimpajama_a4_seed<N>`（report/08） | 8 | `bench_mmlu_slimpajama_a4_seed<N>` |
| ~~75%（A=2）~~ | 測らない（下記） | — | — |

MMLU は ExpertWeaver が shared 割合 α の掃引に使っている指標である。5タスクでは
効果量が `acc` 0.005 程度で seed 3本では桁が足りなかったので、問題数が
14,042 と桁の違う MMLU で同じ差がどう見えるかを取りに行く。

## 測り直すのは MMLU だけ

PPL と5タスクは既に測ってあり、同じ引数で引き直しても同じ数字が出るだけである。
`--no-ppl --bench-tasks mmlu` で MMLU の生の尤度だけを取りに行き、**既存の
`bench_slimpajama*` は1つも触らない**。

代わりに、マクロ平均は実行の中では揃わない。report を書くときに
`cmoe.eval.bench.load_samples()` で5タスクぶんと束ねる（生の対数尤度が残って
いるので、指標も対応のある比較も GPU なしで作り直せる）。

## 構成は既存の実行からそのまま引く

探索が出した配分は seed ごとに違うので、ここで作り直すと別物になる。元の
`bench_.../summary.json` の `arguments.alloc` を**順番ごと**写して渡す（先頭が
`cmoe run` の対応のある比較の基準になるので、順番も意味を持つ）。

元の実行が無い、あるいは終わっていない動作点は、その場で止まる。

## 75%（A=2）は測らない

[`experiments/16_sparsity75`](../16_sparsity75) で、75% は5タスクの `acc` が
chance（macro 0.35）に張り付くところまで壊れていることが分かった。MMLU の
chance は 0.25 なので、ここで測っても**床を確かめるだけ**になる。`--all` は
25% と 50% だけを回す（`ALL_NACTIVES`）。

必要になったら `--nactive 2 --seed <N>` で1本ずつ回せる。配分は
`bench_slimpajama_a2_seed<N>` から引かれる。

## dense の基準

既存の dense（`bench_h4/wikitext2_seed0/bench/dense.json`）は**使えない**。
5タスクぶんしか無く、`cmoe run` はタスクの顔ぶれが違えば取り込みを断る。

**最初の1回だけ測る**。置き場所は A=6 / seed 0 の実行の中
（`bench_mmlu_slimpajama_seed0/bench/dense.json`）で、残りは全部これを取り込む。
dense は変換していないモデルなので、A にも校正にも seed にも配分にも依らない。
`--all` は dense を測る1本を必ず先頭に並べる。

## 固定するもの

元の実行と1つも変えない。変えるのは測るタスクだけである。

| | |
|---|---|
| モデル / 層数 | `meta-llama/Llama-2-7b-hf` / 32 |
| N / A | 8 / 6・4 |
| 校正 | slimpajama n=16 × 2048、seed 0 / 1 / 2 |
| 配分 | 元の実行が測ったものをそのまま |
| ルーター / 分割 | 現行 CMoE（k_act 10 / bias_speed 0.001） |
| ベンチ | **MMLU のみ**、0-shot 全件（14,042問）、batch 32 |
| PPL | 測らない |
| ブートストラップ | 10,000回 / seed 20260813 |

## 走らせ方

**本実行の前に必ず `--smoke` を通すこと。**

```bash
uv run python experiments/17_mmlu_bench/run.py --smoke              # 2層・科目あたり2問

uv run python experiments/17_mmlu_bench/run.py --nactive 6 --seed 0 # dense もここで測る
uv run python experiments/17_mmlu_bench/run.py --nactive 6 --seed 1
uv run python experiments/17_mmlu_bench/run.py --nactive 6 --seed 2
uv run python experiments/17_mmlu_bench/run.py --nactive 4 --seed 0
...

# まとめて 25% と 50%（dense を測る1本が自動で先頭に来る）
uv run python experiments/17_mmlu_bench/run.py --all
```

**MMLU の `--bench-limit` は科目あたりに効く。** 57科目が lm-eval では別タスク
なので、`--bench-limit 2` は 114問になる。smoke がこれを使う。

## 済みの判定

`--no-ppl` なので `summary`（PPL のまとめ）は残らない。見るのは
`bench_summary` があり、構成数が合い、`failures` が無いこと。印の無い
ディレクトリは `*.partial<N>` へ退避してから引き直す。

## 見積もり

MMLU の尤度計算は 14,042問 × 4選択肢 = 56,168本で、5タスク（約60,600本）と
ほぼ同じ量である。5タスクが1構成 約7分だったので、変換（約1分）と合わせて
**1構成 8〜10分**の見当。

| 動作点 | 構成 × seed | 見込み |
|---|---|---|
| 25%（A=6） | 10 × 3 = 30 | 約5時間 |
| 50%（A=4） | 8 × 3 = 24 | 約4時間 |
| dense | 1 | 約10分 |
| | **53回** | **約8時間** |

## 保存されるもの

- `bench_mmlu_slimpajama*_seed<N>/summary.json` … 全引数、MMLU の集計と対応の
  ある比較
- `bench_mmlu_slimpajama*_seed<N>/bench/run<NNN>_cmoe.json` … 問題ごと・
  選択肢ごとの生の対数尤度（fp32）
- `bench_mmlu_slimpajama_seed0/bench/dense.json` … dense の基準。残りの実行が
  取り込む
- `exp17_stages_a<A>_seed<N>.json` … 段の一覧と、測ったコードのコミット
