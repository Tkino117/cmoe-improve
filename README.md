# cmoe-improve

CMoE（学習済み FFN を学習なしで MoE に組み替える手法）の研究を続けるための実験基盤。

差し替えたいもの — モデル・データセット・分割・ルーター・配分 — が、それぞれ独立に
差し替えられることを設計の中心に置いている。設計の意図は
[docs/01_experiment-platform-design.html](docs/01_experiment-platform-design.html) にある。

## 使い方

```bash
uv sync

# 一様 x=3・現行ルーターで変換して PPL を測る
uv run cmoe run --alloc uniform3 --router cmoe

# 配分とルーターの直積。report/13 の 2×2 表はこの形で出る
uv run cmoe run --alloc uniform3,beam --router cmoe,oracle_recovery --seeds 0,1,2

# ルーターの診断だけ（回収率・Oracle 一致率）
uv run cmoe run --router oracle_recovery --diagnostics --no-ppl

# 選択問題ベンチマークも測る。dense を基準に取ってから各構成を測る
uv run cmoe run --alloc uniform4 --alloc beam --bench

# 層ごとの x を探す。出た配分はそのまま run --alloc に貼れる
uv run cmoe search --oracle suffix_kl --search beam --width 4

# 安い層ローカル指標で探す（後続の層を走らせない）
uv run cmoe search --oracle local_error --search greedy --width 1

# 移送が既存の測定と同じ数字を出すかの確認
uv run python experiments/00_anchor.py
```

探索は採点オラクルと探索アルゴリズムを別々に選ぶ。「同じ目的関数で探索を替える」
「同じ探索で安いオラクルに替える」がどちらも1行の違いになり、それが
「計算量を落とす探索」の実験点になる。

結果は `result_logs/<name>/` に入る（`run.txt` = 表示したものすべて、
`summary.json` = 全精度の数値とトークンのハッシュ）。終わった結果がある
ディレクトリへの上書きは拒否される。

## 構造

```
src/cmoe/
  moe/        変換後の層に載る実体（MoE / Router / SubMLP）
  adapters/   [軸1] モデル差異。層の並び、attention の進め方、FFN の差し替え
  data/       [軸2] キャリブレーションと評価のトークン列
  carve/      [軸3] ニューロン分割（活性プロファイル + クラスタリング）
  router/     [軸4] ルーター方式
  alloc/      [軸5] SA 配分。search（探索）と oracles（採点）に分かれる
  eval/       [軸6] PPL・選択問題ベンチマークと、対応のある比較の統計
  assemble.py 組み立て役。軸どうしを繋ぐ知識はここにしか無い
  cli.py      唯一のドライバ（run = 変換して測る / search = 配分を探す）
experiments/  1実験1ファイルの薄い設定
tests/        CPU で数秒で回る動作確認
```

依存は一方向:
`moe ← adapters ← carve ← {router, alloc} ← assemble ← eval ← cli/experiments`。
`router` と `alloc` は互いを import しない。

## 状態

6つの軸すべてが移送済み。

- 校正セット: `wikitext2`、`c4`、`slimpajama`（SlimPajama の7成分を層化して引く。
  成分ごとの本数を固定するので、seed を振っても組成が変わらない）
- アダプタ: `llama`（既存の測定を再現する経路）、`auto`（同じ層構造の他モデル）
- ルーター方式: 現行 CMoE、頻度重心、Oracle 相関、回収率の共同最適化、
  expert 平均（素/centered）、診断用 `|h|` オラクル
- ルーター診断: `|h|` 回収率、Oracle gap 回収率、Top-K の recall と完全一致率
- 採点オラクル: `mass`（活性質量の回収率）、`mass_squared`、`local_error`（層の
  出力誤差 L）、`suffix_kl`（残りを dense のまま走らせた出力分布の KL）
- 配分の探索: `beam`（幅を指定）、`greedy`（幅1のビームそのもの）
- 評価: PPL と、選択問題ベンチマーク5タスク（`--bench`）

### 選択問題ベンチマーク

既定のタスクは PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag の 0-shot で、
CMoE 最新版（ACL 2026, arXiv:2502.04416）Table 1 と同じ並びであり、
ExpertWeaver（arXiv:2602.15521）Table 2 との共通部分でもある。どちらの論文も
主表の動作点はスパース率25%で、ここの N=8 / A=6 がちょうどそれに当たる。

**正答率だけを見ない。** 選択問題の採点は選択肢ごとの対数尤度の argmax であり、
正答率はマージンの**符号**しか見ない。ここで問題になる差（PPL 0.03 ≒ 1トークン
0.004 nat）でマージンの符号をまたぐ問題はごく一部で、正答率の標本誤差に埋もれる。
そこで残すのは集計値ではなく**問題ごと・選択肢ごとの生の対数尤度**で、指標は
そこから後で作る（`bench_stats`）。

| 指標 | 中身 | 向き |
|---|---|---|
| `acc` / `acc_norm` | 論文の表に載る正答率（後者は選択肢の文字数で割ってから argmax） | 大 |
| `gold_nll` | 選択肢だけで softmax した正解確率の −log。K択分類の交差エントロピー | 小 |
| `margin` | 正解 − 最良の不正解の対数尤度。符号が `acc` そのもの | 大 |
| `ref_kl` | dense との、選択肢上の分布の KL。正誤と切り離して「壊した量」を見る | 小 |
| `ref_agreement` | dense と同じ選択肢を選んだか | 大 |

信頼区間は **(seed × タスク)** を層とした、問題単位の対応のある再抽出。タスクごとに
問題数が 1,200〜10,000 と一桁違うので、層を等しく重み付けしてマクロ平均にする。

生の尤度は `result_logs/<name>/bench/` に構成ごとの JSON で残る（`dense.json` が
基準）。指標を足すのに測り直しは要らない。

ベンチのデータセットは `.cache/hf-datasets` に持つ（`--bench-cache-dir`）。共有の
HuggingFace キャッシュには新しい `datasets` が書いた索引が混じっており、このリポジトリ
が固定している 2.21.0 では ARC と HellaSwag が読めないため。

探索を走らせずに使う既知の配分は `src/cmoe/alloc/presets.py` に定数として
置いてある。探索の再実装が同じベクトルを出すことは求めていない（浮動小数の
順序ひとつで分岐が変わる）。

元実装と過去のレポートは `CMoE-ref/`（追跡対象外）にある。
