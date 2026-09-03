# 19 ルーティングの改良

report/18 が測ったのは「選ぶ前に真の活性 |h| を全部計算してから Top-K を取る」
オラクル `oracle_abs` で、これは選択問題の `acc` をマクロ +0.0221 上げた（dense
との差の 33.5%）。ただし配備できない。report/10 で測った配備できる5方式は、
どれも `acc` を動かせなかった。

この実験は「オラクルを安く真似れば `acc` が上がるか」を追い、途中で**もっと効く
軸**を見つけた。中身は3本の道具でできている。

## 道具

| ファイル | 何をするか | GPU |
|---|---|---|
| `dump.py` | 変換と同じ順で層を歩き、指定した層の z・分割・重みを落とす | 要る（約3分） |
| `probe.py` | 落とした材料の上で、score 関数を並べてオラクル一致・質量・層出力誤差を採点する | 要る（案あたり数秒） |
| `dynamic_k.py` | 同じ平均コストで expert をトークン間に配り直したときの層出力誤差 | 要る（同上） |
| `run.py` | 本測定（PPL と選択問題ベンチマーク） | 要る（数時間） |
| `per_task.py` | 本測定の生の尤度から、タスクごとの対照比を出し直す | 要らない |

`probe.py` と `dynamic_k.py` はモデルを走らせないので、案を1つ試すのに数秒しか
かからない。ベンチマークまで行く前に案を落とすためにある。

```bash
uv run python experiments/19_spectral_router/dump.py --layers 0,3,7,11,15,19,23,27,31
uv run python experiments/19_spectral_router/probe.py --scorers cmoe,wlowrank32,oracle
uv run python experiments/19_spectral_router/dynamic_k.py --scorers cmoe,oracle \
    --modes fixed,tau
uv run python experiments/19_spectral_router/run.py --smoke
uv run python experiments/19_spectral_router/run.py
uv run python experiments/19_spectral_router/per_task.py \
    --run result_logs/spectral_router_bench --metrics acc,gold_nll
```

落とす先は既定で `/tmp/claude-1000/probe/dump`（`CMOE_PROBE_DIR` で変えられる）。
1層あたり 539 MB あるので、リポジトリの中には置かない。

## 試して落とした案

`probe.py` に残っているのは report/19 に載った案だけである。落とした案は実装ごと
消してあるので、何をどこまで試したかをここに置く。数字は層 15（一部は 3層または
9層の平均）の、オラクルとの Top-K 完全一致率と gap 回収である。

| 案 | 中身 | 結果 |
|---|---|---|
| 無作為 m 本 | expert から無作為に m 本引いて \|h\| を足す（不偏推定） | m=16 で 8.7%。平均 \|h\| の上位から取る方が良い（14.5%） |
| 共有部分空間 | x の主成分 r 次元へ落としてから厳密なノルムを計算 | r=256 で 29.3%。expert ごとの基底（rank 32、同じコスト）の 42% に負ける |
| 共有基底（重み由来） | routed 全体の白色化 SVD の上位 r 本を全 expert で共有 | r=256 で 22.6%。共有基底は判別に効かない方向へ予算を使う |
| PCA + 2次形式 | r 次元へ落とし、expert ごとに2次形式をリッジ回帰で当てる | r=64 で 29.1%。ノルムの積（4次）に対して2次では足りない |
| shared expert の活性 | 常時走る shared の \|h\|（4,128次元）から線形に読む。**追加コストは現行ルーターの半分** | 32.0%（9層平均）。安さの割に良いが、当てはめが校正データに寄る |
| 標本 + 低ランクの合成 | 上の2つを最小二乗で足す | 単独の低ランクより悪い（26.1% 対 42.2%）。二乗誤差は順位付けの目的関数として合わない |
| 双一次形式 | \|h\| の符号を行ごとに固定して和を x の2次形式1枚に落とす | 符号が安定せず、形にならなかった |
| expert ごとの倍率較正 | Cauchy–Schwarz の緩みを平均比で消す | 42.4% 対 42.0%。誤差の範囲。可変Kと合わせるとむしろ悪化（平均 K が 2.69 に落ちる） |
| 出力寄与で重み付け | 行の重みに down_proj の列ノルムも掛け、`Σ‖d_i‖·\|h_i\|` を近似する | 41.7% 対 41.9%。変わらない |

## 出てきた方式

`src/cmoe/router/methods/` に2つ入れた。

* **方式7 `spectral_mass`** — 代表ニューロン1本をやめ、expert の活性質量を
  `‖silu(G_e x)‖·‖U_e x‖`（Cauchy–Schwarz の上界）で見積もる。重み行列の低ランク
  近似で作るので、増えるのは expert あたり `2·r·hidden` の積和だけ。
  `spectral_mass:64` と書くと rank を指定できる。
* **方式8 `dynamic_cmoe` / `dynamic_spectral`** — 固定 Top-K をやめ、score の
  しきい値1個で選ぶ。平均だけを予算に合わせ、トークンごとの個数は変える。
  `dynamic_cmoe` は score を現行のまま使うので、**追加の積和が1回も無い**。
