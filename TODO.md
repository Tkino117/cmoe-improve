# TODO

設計は [docs/01](docs/01_experiment-platform-design.html)、使い方は [README](README.md)。

## 候補

- **校正データを測る対象に近づける（experiments/09）** — ベンチ5タスクの train split で校正する `benchtrain`。動くなら次は KL を答え部分のトークンだけで取る
- **expert 粒度を上げる（S6A6E16）** — 先に N を振ってオラクル L の床だけ測り、下がるなら本測定へ
- **配分とルーターを同時に動かす** — 足し算にならないことが分かっているので、まず 2×2 で交互作用を見る
