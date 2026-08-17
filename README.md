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

# 移送が既存の測定と同じ数字を出すかの確認
uv run python experiments/00_anchor.py
```

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
  eval/       [軸6] PPL と対応のある比較の統計
  assemble.py 組み立て役。軸どうしを繋ぐ知識はここにしか無い
  cli.py      唯一のドライバ
experiments/  1実験1ファイルの薄い設定
tests/        CPU で数秒で回る動作確認
```

依存は一方向:
`moe ← adapters ← carve ← {router, alloc} ← assemble ← eval ← cli/experiments`。
`router` と `alloc` は互いを import しない。

## 状態

移送済み: 軸1〜4 と6 の全体、軸5 の適用側（配分ベクトルをモデルに反映する経路）。

- アダプタ: `llama`（既存の測定を再現する経路）、`auto`（同じ層構造の他モデル）
- ルーター方式: 現行 CMoE、頻度重心、Oracle 相関、回収率の共同最適化、
  expert 平均（素/centered）、診断用 `|h|` オラクル
- ルーター診断: `|h|` 回収率、Oracle gap 回収率、Top-K の recall と完全一致率

未移送: 配分の探索アルゴリズム（beam / greedy）と採点オラクル。既知の配分は
`src/cmoe/alloc/presets.py` に定数として置いてある。

元実装と過去のレポートは `CMoE-ref/`（追跡対象外）にある。
