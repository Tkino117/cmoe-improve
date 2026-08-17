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

## 残り

1. 計算量を落とす探索 — 安いオラクルで候補を絞り、高いオラクルで決める段構え。
   予算（`--budget`）は入れてあるので、同じ予算での比較はすぐ回せる
2. 未測定の実験（走らせるだけ）: carve n=64 での 2×2 再測定、方式6 との組み合わせ
3. GPU での探索の実走確認。CPU の極小モデルまでしか通していない

## 注意

- `CMoE-ref/` は参照専用。import しない
- ビーム配分は20層で Top-K=0 になり、配分とルーターの効果は足し算にならない
- 回収率は PPL を予測しない。採否は PPL で決める
- 接頭辞のスコアは残りの層が dense のままの値である。実在するモデルを表すのは
  最終層で測った値だけで、それより前は「どの接頭辞を生かすか」の当て推量
