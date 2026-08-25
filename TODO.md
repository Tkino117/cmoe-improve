# TODO

設計は [docs/01](docs/01_experiment-platform-design.html)、使い方は [README](README.md)。

## 完了

6つの差し替え軸（モデル / データ / 分割 / ルーター / 配分 / 評価）と組み立て役・CLI。
ルーター方式は現行 CMoE・方式2・3・4・6 と診断用 `|h|` オラクル。
配分は採点オラクル4種（mass / mass_squared / local_error / suffix_kl）と
探索2種（beam / greedy）。

選択問題ベンチマーク（`--bench`）。PIQA / WinoGrande / ARC-e / ARC-c /
HellaSwag の 0-shot を lm-eval-harness 0.4.12 で通し、**問題ごと・選択肢ごとの
生の対数尤度**を残す。指標は `acc` / `acc_norm` / `gold_nll` / `margin` /
`ref_kl` / `ref_agreement` の6つで、後から GPU なしで足せる。

検証: `uv run pytest tests/ -q`（CPU、数秒。オラクルは `CMoE-ref/xsearch` と
ビット一致を確認）、`CMOE_BENCH_INTEGRATION=1 uv run pytest tests/test_bench.py -q`
（本物の lm-eval を5タスク通す。約20秒）、
`CMOE_SLIMPAJAMA_INTEGRATION=1 uv run pytest tests/test_data_slimpajama.py -q`
（SlimPajama のシャードを実際に落として引く。初回は 292MB のダウンロード）、
`uv run python experiments/00_anchor.py`（GPU、約25分、既存測定と6桁一致）。

探索は Llama-2-7B で実走済み（`result_logs/gputest_*`）。全32層 × `local_error`
× beam 幅2 が 15.8 分、勝った配分の測り直しは差 0。出た配分の PPL は
wikitext2 で **6.992** に対し `uniform3` が **7.072**（対照比 NLL -0.0113
[-0.0147, -0.0080]、seed 1本）。`suffix_kl` では層0 の2番手から伸びた枝が勝ち、
幅が効くことも実機で確認した。

## 残り

1. 計算量を落とす探索 — 安いオラクルで候補を絞り、高いオラクルで決める段構え。
   予算（`--budget`）は入れてあるので、同じ予算での比較はすぐ回せる
2. 未測定の実験（走らせるだけ）: carve n=64 での 2×2 再測定、方式6 との組み合わせ
3. H4（ベンチマーク）は済んだ — [report/04](report/04_alloc-search-benchmark.md)。
   探索配分は最良の一様配分を**どの指標でも上回らなかった**。対照の取り方は
   決めた — 一様7本を探索と同じ `suffix_kl` で採点して選ぶ（`cmoe score`）。
   両側とも「校正だけを見て選んだ1本」になる。
   slimpajama 校正の seed 3本は済んだ — [report/07](report/07_slimpajama-alloc-3seeds.md)。
   PPL は3対照とも幅4 で 3/3 seed・区間も0をまたがない。ベンチの `gold_nll` は
   固定 uniform3 に対してのみ判定を満たし、校正で選んだ一様に対しては区間が
   0をまたぐ
   スパース率50%（N=8 / A=4）でも同じ格子を回した —
   [report/08](report/08_sparsity50-alloc-3seeds.md)。PPL は 3対照 × 3幅 ×
   2評価セットの18箇所すべてが判定を満たす。ベンチの `gold_nll` は区間が
   0をまたがなくなった（幅4 で −0.160 [−0.182, −0.138]）が、seed の符号が 2/3
   で判定は満たさない。`acc` は依然 +0.004 [−0.001, +0.008] で埋もれている
4. `suffix_kl` の全32層はまだ走らせていない（既定の幅4 なら数時間。`presets.py`
   の `beam` はこの目的関数の産物なので、突き合わせる相手がある）

## 注意

- `CMoE-ref/` は参照専用。import しない
- ビーム配分は20層で Top-K=0 になり、配分とルーターの効果は足し算にならない
- スパース率は `--nactive` で決まる（N=8 で A=6 が25%、A=4 が50%）。50%では
  校正が選ぶ一様は3 seed とも `uniform4` — Top-K=0、つまり routing 無しである
- beam は幅を広げても最終スコアが単調に下がる保証を持たない。層 L で残す上位 w 本
  は幅が広いほど母集団も広がるので、狭い beam が残していた接頭辞が押し出される
  （report/08 の seed 1 で幅4 が幅2・3 に負けた）
- 回収率は PPL を予測しない。採否は PPL で決める
- 接頭辞のスコアは残りの層が dense のままの値である。実在するモデルを表すのは
  最終層で測った値だけで、それより前は「どの接頭辞を生かすか」の当て推量
- 層ローカルのオラクル（mass / local_error）は A < N でないと何も測らない。
  A = N ではどの候補も routed を全部走らせる（オラクル生成時に断る）
- 配分名とベクトルを並べて測るには `--alloc` を繰り返す。
  `--alloc uniform3,3,6,...` は1つのカンマ区切りに両方を入れることになり通らない
