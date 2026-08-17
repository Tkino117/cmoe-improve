# TODO

設計は [docs/01](docs/01_experiment-platform-design.html)、使い方は [README](README.md)。

## 完了

6つの差し替え軸（モデル / データ / 分割 / ルーター / 配分 / 評価）と組み立て役・CLI。
ルーター方式は現行 CMoE・方式2・3・4・6 と診断用 `|h|` オラクル。
配分は採点オラクル4種（mass / mass_squared / local_error / suffix_kl）と
探索2種（beam / greedy）。

検証: `uv run pytest tests/ -q`（CPU、数秒。オラクルは `CMoE-ref/xsearch` と
ビット一致を確認）、`uv run python experiments/00_anchor.py`（GPU、約25分、
既存測定と6桁一致）。

探索は Llama-2-7B で実走済み（`result_logs/gputest_*`）。全32層 × `local_error`
× beam 幅2 が 15.8 分、勝った配分の測り直しは差 0。出た配分の PPL は
wikitext2 で **6.992** に対し `uniform3` が **7.072**（対照比 NLL -0.0113
[-0.0147, -0.0080]、seed 1本）。`suffix_kl` では層0 の2番手から伸びた枝が勝ち、
幅が効くことも実機で確認した。

## 残り

1. 計算量を落とす探索 — 安いオラクルで候補を絞り、高いオラクルで決める段構え。
   予算（`--budget`）は入れてあるので、同じ予算での比較はすぐ回せる
2. 未測定の実験（走らせるだけ）: carve n=64 での 2×2 再測定、方式6 との組み合わせ
3. 上の PPL 差は seed 1本・wikitext2 のみ。採否を言うには seed 3本 × c4-new が要る
4. `suffix_kl` の全32層はまだ走らせていない（既定の幅4 なら数時間。`presets.py`
   の `beam` はこの目的関数の産物なので、突き合わせる相手がある）

## 注意

- `CMoE-ref/` は参照専用。import しない
- ビーム配分は20層で Top-K=0 になり、配分とルーターの効果は足し算にならない
- 回収率は PPL を予測しない。採否は PPL で決める
- 接頭辞のスコアは残りの層が dense のままの値である。実在するモデルを表すのは
  最終層で測った値だけで、それより前は「どの接頭辞を生かすか」の当て推量
- 層ローカルのオラクル（mass / local_error）は A < N でないと何も測らない。
  A = N ではどの候補も routed を全部走らせる（オラクル生成時に断る）
- 配分名とベクトルを並べて測るには `--alloc` を繰り返す。
  `--alloc uniform3,3,6,...` は1つのカンマ区切りに両方を入れることになり通らない
