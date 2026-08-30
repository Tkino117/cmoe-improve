# TODO

設計は [docs/01](docs/01_experiment-platform-design.html)、使い方は [README](README.md)。

## 候補

- **タスク特化の主張を詰める（docs/04）** — 校正の読む位置を採点位置に寄せる線は [report/14](report/14_*scored-positions.md) で打ち切った
- **同じコストなら幅と深さのどちらか（追加測定なし）** — 深さ1（コスト 7.57倍）と、既にある幅4・深さ0（同 2倍）を [report/15](report/15_*search-depth.md) の対照の隣に並べる
- **expert 粒度を上げる（S6A6E16）** — 先に N を振ってオラクル L の床だけ測り、下がるなら本測定へ
- **配分とルーターを同時に動かす** — 足し算にならないことが分かっているので、まず 2×2 で交互作用を見る
