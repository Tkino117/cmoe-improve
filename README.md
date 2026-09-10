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

# 校正を1問1系列で引き、採点に効く位置だけで活性を数える
uv run cmoe run --calib benchqa --profile-positions scored --alloc uniform4 --bench

# 探索の目的関数でも答え部分だけを見る（--scored-weight 1 で答え部分が全部）
uv run cmoe search --calib benchqa --profile-positions scored \
    --scored-weight 1.0 --search beam --width 2

# 親モデルとの KL ではなく、選択問題の採点そのもので探す
uv run cmoe search --calib benchchoice --oracle margin --search beam --width 2

# 探索を走らせず、与えた配分を同じオラクルで採点する。対照の一様配分を
# 「評価指標を見て選ぶ」のではなく、探索と同じ目的関数で選ぶのに使う
uv run cmoe score --oracle suffix_kl --alloc uniform3 --alloc uniform4

# 静的プルーニングの対照。原典の定義（attn+FFN 全体の比）で 25% と 50%
uv run cmoe prune --method flap,llm_pruner --sparsity 0.25,0.5 --scope block --bench

# 同じ手法を FFN だけに限り、トークンあたりの活性パラメータを MoE 変換に揃える
uv run cmoe prune --method flap --sparsity 0.25 --scope mlp --bench

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
  prune/      [対照] 静的な構造化プルーニング（FLAP / LLM-Pruner）。MoE 変換を
              通らない比較手法で、軸ではない
  assemble.py 組み立て役。軸どうしを繋ぐ知識はここにしか無い
  cli.py      唯一のドライバ（run = 変換して測る / search = 配分を探す /
              score = 与えた配分を採点する / prune = 静的プルーニングの対照を測る）
experiments/  1実験1ファイルの薄い設定
tests/        CPU で数秒で回る動作確認
```

依存は一方向:
`moe ← adapters ← carve ← {router, alloc} ← assemble ← eval ← cli/experiments`。
`router` と `alloc` は互いを import しない。

## 状態

6つの軸すべてが移送済み。

- 校正セット: `wikitext2`、`c4`、`slimpajama`（SlimPajama の7成分を層化して引く。
  成分ごとの本数を固定するので、seed を振っても組成が変わらない）、`benchtrain`
  （選択問題5タスクの train split を層化して引く。評価に使う split は入らない）、
  `benchqa`（同じ train split を**1問1系列**で引き、採点される対数尤度を作って
  いる位置に印を付ける。印を使うかは `--profile-positions` が決める）、
  `benchchoice`（同じ問題を**選択肢ごとに1系列**で引き、行と問題の対応を持つ。
  選択肢どうしを比べる目的関数が要る唯一のセット。予算はタスクごとの問題数で
  数える）、`benchtrain:mmlu`（MMLU には train split が無いので dev + validation
  から引く。採点する test は入らない）
- アダプタ: `llama`（既存の測定を再現する経路）、`auto`（同じ層構造の他モデル）
- ルーター方式: 現行 CMoE、頻度重心、Oracle 相関、回収率の共同最適化、
  score 校正（代表を凍結し expert ごとの gain と offset を座標上昇で合わせる）、
  expert 平均（素/centered）、診断用 `|h|` オラクル、
  `spectral_mass`（代表ニューロン1本をやめ、expert の活性質量を重み行列の低ランク
  近似 `‖silu(G_e x)‖·‖U_e x‖` で見積もる。`spectral_mass:64` で rank を指定。
  白色化を切った `spectral_plain`、校正を一切読まない `spectral_weight` もある）、
  `dynamic_cmoe` / `dynamic_spectral`（固定 Top-K をやめ、score のしきい値1個で
  選ぶ。平均だけを予算に合わせるので、`dynamic_cmoe` は追加の積和がゼロ。
  評価データの上で実際に走った expert 数は `summary.json` の `realized_load` に
  残る）
- ルーター診断: `|h|` 回収率、Oracle gap 回収率、Top-K の recall と完全一致率
- 採点オラクル: `mass`（活性質量の回収率）、`mass_squared`、`local_error`（層の
  出力誤差 L）、`suffix_kl`（残りを dense のまま走らせた出力分布の KL）、
  `margin`（選択問題の採点そのもの。正解と、最も惜しい不正解の対数尤度の差。
  `--margin-beta` の両端で `bench_stats` の `margin` と `gold_nll` に一致する。
  `benchchoice` が要る）
- 活性を数える位置: 全位置（既定）と、採点される対数尤度を作っている位置だけ
  （`--profile-positions scored`。印を持つ校正セットが要る）
- オラクルが採点する位置: `--scored-weight` が、目的関数のうち答え部分が占める
  割合を決める（1 で答え部分だけ、0 で文脈だけ）。省略しても、印を持つ校正
  セットでは埋めを数えない。分割を決める `--profile-positions` とは別の軸で、
  「どの位置で切り分けるか」と「どの位置で採点するか」を別々に動かせる
- 配分の探索: `beam`（幅を指定）、`greedy`（幅1のビームそのもの）。探索を
  走らせずに配分を採点するだけの `cmoe score` もある（対照を校正で選ぶため）
- 配分の対照（先行研究を配分軸だけ移したもの。**どれも提案手法ではない**）:
  `alloc/ew_rule.py`（ExpertWeaver の式4・5・6。report/21）、
  `alloc/owl.py`（OWL の外れ値比率）、
  `alloc/score_rule.py`（層別スカラー → x の写像。EW と OWL の共通の器）、
  `alloc/lexi.py`（LExI の感度プロファイル。層ローカル誤差の独立な argmin）、
  `alloc/order.py`（順序対照。配分の多重集合を保ったまま層の並びだけ崩す）。
  統計を作るのは `experiments/22_alloc_baselines/{owl,lexi}_probe.py` で、
  そこから先の写像は CPU だけで足りる
- 静的プルーニングの対照（**MoE 変換を通らない**先行研究。どれも提案手法では
  ない）: `prune/flap.py`（FLAP の WIFV + AL-AM + bias 補償）、
  `prune/llm_pruner.py`（LLM-Pruner の block-wise Taylor。**追加学習は行わない**）。
  ExpertWeaver Table 2 の training-free 比較と同じ顔ぶれである。スパース率は
  `--scope` が2つの土俵を持つ — `block` は原典の定義（attn+FFN 全体の比）、
  `mlp` は FFN だけを刈って**トークンあたりの活性パラメータを MoE 変換に揃える**。
  この基盤の「25%」は FFN ニューロンの比なのでブロック全体では 16.71% にしか
  ならず、同じ「25%」と書くと対照だけが1.5倍削ることになる。`summary.json` の
  `accounting` が名目比と実効比の両方を残す。report/25
- 評価: PPL と、選択問題ベンチマーク5タスク（`--bench`）

### 選択問題ベンチマーク

既定のタスクは PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag の 0-shot で、
CMoE 最新版（ACL 2026, arXiv:2502.04416）Table 1 と同じ並びであり、
ExpertWeaver（arXiv:2602.15521）Table 2 との共通部分でもある。どちらの論文も
主表の動作点はスパース率25%で、ここの N=8 / A=6 がちょうどそれに当たる。動作点は
`--nactive` で替えられる（N=8 で A=4 なら50%。[report/08](report/08_sparsity50-alloc-3seeds.md)）。

**MMLU も測れる**（`--bench-tasks mmlu`、test 14,042問）。lm-eval では57科目が
別々のタスクなので、`harness.subtasks` が名前を科目に開き、持ち帰った尤度を
`bench.concatenate` が1タスクに束ね直す（並びは科目名順に固定されるので、構成を
またいだ問題の対応は崩れない）。ExpertWeaver が shared 割合 α の掃引に使って
いる指標がこれである。校正に使うときは `--calib benchtrain:mmlu`。

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
