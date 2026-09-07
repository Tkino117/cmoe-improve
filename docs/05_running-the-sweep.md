# 05 新しいマシンで本実験を回す手順

4枚の GPU を積んだマシンで、Docker の中だけで完結させる。ここに書いてあるのは
**論文の主表を作るための20ジョブ**（2モデル × 2スパース率 × 5 seed）の回し方で、
1つ1つの実験の中身は `experiments/25_model_seeds/run.py` の冒頭にある。

## 何を回すのか

| | |
|---|---|
| モデル | `meta-llama/Llama-2-7b-hf` と `mistralai/Mistral-7B-v0.1`（どちらも32層、intermediate が8で割り切れる） |
| 動作点 | N=8 / A=6（スパース率25%）と N=8 / A=4（50%） |
| seed | 0・1・2・3・4 の5本。**探索も seed ごとに引き直す**ので、配分そのものが seed ごとに独立に決まる |
| 校正 | slimpajama n=16 × 2048 |
| 評価 | PIQA / WinoGrande / ARC-e / ARC-c / HellaSwag / MMLU の `acc` と `gold_nll`、WikiText-2 と C4-new の PPL |
| ジョブ数 | 2 × 2 × 5 = **20本** |

1ジョブは4段（探索 幅2/3/4 → 一様の採点 → 5タスクとPPL → MMLU）。旧マシン
（RTX PRO 5000 Blackwell）の実測では **A=6 が 5.4〜6.1時間、A=4 が 4.5〜4.8時間**
（`result_logs/exp0{6,7}_stages_seed*.json` と `bench_mmlu_*/summary.json` の合計）。
Mistral は Llama より1割ほど遅い。**合計およそ110時間、4枚に配って実時間28時間**
が目安になる。

## 前提

- GPU 4枚（各48GB。7B の bf16 と探索の中間テンソルには余裕がある）
- NVIDIA ドライバ 525.60.13 以上。torch 2.8 は cu128 のホイールで、CUDA の
  minor version compatibility に頼っている。測定先の 535.309.01（CUDA 12.2）は
  満たしている。版そのものは `nvidia-smi` で見る。実際に動くかは手順2 で確かめる
- Docker と NVIDIA Container Toolkit（`docker run --gpus` が通ること）
- HuggingFace のモデルを引けること。**Llama-2 は gated** なので、手順2 の
  「HuggingFace の認証」を通しておく。通っていないと preflight のモデルと
  データセットが両方落ちる — 校正が Llama のトークナイザを引くので、**原因は
  1つでも2件に見える**
- `/data/kinoshita` に 100GB ほどの空き（モデル28GB + データセット + 一次データ）。
  **共有ホームとリポジトリには何も書かない**

## 手順1 イメージを作る

```
git clone <このリポジトリ> && cd cmoe-improve
docker build -t kinoshita/cmoe-improve .
```

イメージ名は `kinoshita/cmoe-improve`、コンテナ名は `kinoshita_cmoe-improve`。
**この2つが出てくるのはこの文書と `Dockerfile` の冒頭コメントだけで、コードは
docker の名前に一切依存していない**（`run.py` / `sweep.py` はコンテナの中でも
外でも同じように動く）。

CUDA はイメージに入れていない。torch のホイールが CUDA ランタイムを同梱して
いるので、要るのはホスト側のドライバだけである。依存は build 時に
`uv sync --frozen` で焼いてあり、実行時は解決し直さない。

## 手順2 呼び出し用の関数を読み込む

**ホスト側のシェルで実行する。**

```
cd /abs/path/to/cmoe-improve
source scripts/docker-env.sh
cmoe_env                       # 設定を確認する
```

以降の手順はすべて同じマウントで走らせるので、`docker run` を毎回書く代わりに
関数3本を定義してある。中身は `scripts/docker-env.sh` を見ること。

| 関数 | 何 |
|---|---|
| `cmoe1 <コマンド>` | **GPU 1枚だけ見せる。** 手順3 の smoke、手順4、手順5 はこれ |
| `cmoeall <コマンド>` | 全部見せる。**preflight だけ**（枚数を数えるのが仕事なので絞らない） |
| `cmoe_login` | HuggingFace にログインする（`HF_TOKEN` を使うなら不要） |
| `cmoe_sweep` | 手順6 の常駐コンテナを起こす |

### HuggingFace の認証

Llama-2 は gated なので通しておく。**共有アカウントなら環境変数を勧める。**

```
export HF_TOKEN=hf_...        # ディスクに残らない。シェルごとに設定する
```

`source` がこれを拾ってコンテナへ渡す（`cmoe_env` の `AUTH` 行で確認できる）。
**設定されているときだけ渡す** — 空のまま渡すとファイル側のトークンを上書きして、
かえって認証が通らなくなる。

ファイルに残したいなら `cmoe_login`（ホストには何も入れなくてよい。CLI は
コンテナの venv にある）。ただし**トークンは `/data/kinoshita/cmoe-cache/huggingface/token` に
平文で残り、そのディレクトリを読める人には見える**ので、アカウントを共有して
いるなら避けたほうがよい。

**シェルを開き直すたびに `source` し直すこと。** 28時間の実行中に再接続したら、
もう一度読んでから `docker logs` を見にいく。

### 共有マシンを汚さない作りにしてある

書き込むのは **`/data/kinoshita` の下だけ**で、共有のホームにもリポジトリにも
何も書かない。

| ホスト | コンテナ | 何が入るか |
|---|---|---|
| `/data/kinoshita/cmoe-cache` | `/workspace/.cache` | モデル28GB、データセット、uv と triton のキャッシュ。**ここだけ育つ** |
| `/data/kinoshita/cmoe-results` | `/workspace/result_logs` | 一次データ |

`/data/kinoshita` が無ければ先に作る。別のディスクにしたいときだけ
`CMOE_DATA=<場所> source scripts/docker-env.sh` で移せる。

**HuggingFace のキャッシュもここに入る**（`HF_HOME=/workspace/.cache/huggingface`）。
共有ホームの `~/.cache/huggingface` は触らないので、他の人が入れた新しい
`datasets` の索引と混ざる心配もない（固定してある 2.21.0 が知らない特徴量の型で
落ちる、という既知の問題を避けられる）。

**コンテナは呼んだ人の uid で走る**（`--user $(id -u):$(id -g)`）。既定の root で
走らせると、マウント先に root 所有のファイルが残って他の人が消せない。何か
壊れたら `CMOE_USER="" source scripts/docker-env.sh` で root に戻せるが、その
ときは出力の所有者に注意すること。

片付けは `/data/kinoshita` を消すだけでよい。イメージは
`docker rmi kinoshita/cmoe-improve`。

コンテナ名は3本とも `kinoshita_cmoe-improve` で共通なので、**同時に2本走らせない**
こと（名前が衝突する）。手順3〜5 は上から順に1本ずつなので問題にならない。

**`preflight` と `sweep.py` 以外は必ず1枚に絞る。** アダプタは
`device_map='auto'` で読むので、2枚以上見えていると7Bが分割され、その実行が
測る値だけが別のデバイス構成のものになる。`run.py` の `require_single_gpu()` が
止めるが、`cmoe1` を使えば引っかからない。

## 手順3 preflight

110時間を投げる前に、その環境で本当に走るかを確かめる。

```
cmoeall uv run python experiments/25_model_seeds/preflight.py
```

**初回は28GB のモデルとデータセットを引くので、回線次第で1時間ほどかかる。**
2回目からは数分。環境だけ先に見たいなら `--skip-models`。

見ているのは4つ。

1. **torch がその GPU でカーネルを持つか。** `is_available()` だけでは
   「カーネルが無い」を見逃すので、bf16 の行列積を実際に走らせる。ここが通れば
   ドライバは実質足りている。
2. **GPU が何枚見えているか。**
3. **モデル2本が読めるか。** 層構造の検査と、intermediate が N=8 で割り切れる
   ことまで見る。
4. **データセットが引けるか。** **測定の途中でダウンロードが起きると数時間の
   実行が止まる**ので先に温める。

環境が通ったら、変換と探索の経路を数分で通しておく。

```
cmoe1 uv run python experiments/25_model_seeds/run.py --model mistral-7b --smoke
```

3層・7本・8問だけ走る。経路のバグをここで見つければ、手順4 の35分を待たずに済む。

## 手順4 dense を測る（モデルごとに1回、各35分）

```
cmoe1 uv run python experiments/25_model_seeds/run.py --model llama2-7b  --check
cmoe1 uv run python experiments/25_model_seeds/run.py --model mistral-7b --check
```

`--check` は探索を回さず、**dense（5タスクとMMLU、全件）と固定対照 `uniform3`
だけ**を測る。目的が2つある。

**(a) 20ジョブを待ち無しにする。** dense は変換していないモデルなので A にも
seed にも配分にも依らず、モデルごとに1つでよい。A=4 のジョブもこれを取り込む。
**先に通していないと、A=6 / seed 0 だけが dense を持ち、4枚に投げた残りが
「dense の基準が要る」で即死する。**

**(b) 変換前後が妥当かを確かめる。** dense の `acc` を公称値と突き合わせる。

| | Llama-2-7b（旧マシン、参考） | Mistral-7B-v0.1 | Mistral の公称（0-shot） |
|---|--:|--:|--:|
| PIQA | 0.7802 | 0.8036 | ≈0.805 |
| WinoGrande | 0.6953 | 0.7506 | ≈0.740 |
| ARC-e | 0.7567 | 0.8022 | ≈0.809 |
| ARC-c | 0.4292 | 0.4966 | ≈0.499 |
| HellaSwag | 0.5708 | 0.6151 | `acc` の公称は少ない（`acc_norm` 0.810 に対し測定値 0.8107） |
| MMLU | 0.4078 | 0.6010 | ≈0.60 |

**判定は 0.02 を目安にする。** 公称値そのものが報告ごとに ±0.01 ほど動くので、
一致は小数第2位までは求められない（上の表でも WinoGrande が 0.751 対 0.740）。
0.02 を超えて外れたらアダプタかドライバのどちらかが違っているので、先へ進まない。
HellaSwag だけは公称が `acc_norm` で出回っているので、比べる列を間違えないこと。

## 手順5 dense の PPL を測る（モデルごとに1回、各2分）

```
cmoe1 uv run python experiments/25_model_seeds/dense_ppl.py --model meta-llama/Llama-2-7b-hf
cmoe1 uv run python experiments/25_model_seeds/dense_ppl.py --model mistralai/Mistral-7B-v0.1
```

`cmoe run` は変換後の構成しか PPL を測らないので、dense の行はこの段が作る。
主表の PPL を dense 比で読むのに要る。

## 手順6 20ジョブを配る

28時間かかるので、接続を保ったままにはできない。**切り離して走らせ、ログを残す。**

```
cmoe_sweep                             # -d で起こす
docker logs -f "$CMOE_NAME"            # 起動・完了・失敗の一覧はここにしか出ない
```

**このコンテナだけ `--rm` を付けていない。** 付けると終了後に `docker logs` で
追えなくなるためで、代わりに次に回す前へ `docker rm "$CMOE_NAME"` で片付ける
（`cmoe1` / `cmoeall` と同じ名前を使うので、残っていると衝突する）。

`sweep.py` は空いた GPU に次のジョブを渡すだけの配り役で、**1ジョブは1枚に
閉じる**（`CUDA_VISIBLE_DEVICES` を1枚だけ見せる）。長い A=6 から先に投げるので、
最後に1本だけ残って3枚が遊ぶ形になりにくい。

```
--dry-run              走らせるジョブの一覧だけ出す
--models mistral-7b    片方のモデルだけ
--seeds 3,4            特定の seed だけ
```

ジョブ個別のログは `result_logs/sweep_logs/<model>_a<A>_seed<N>.log`。

### 途中で落ちたとき

**同じコマンドをもう一度打つだけでよい。** 済んだ段は飛ばす。段の単位は
「探索1本」「一様の採点」「5タスクとPPL」「MMLU」で、**段の途中で落ちるとその段の
全構成をやり直す**（`cmoe run` が10構成を1回で回しているため）。失うのは最大で
2時間ほど（A=6 の段3が実測 1.6〜1.8時間）。

書きかけのディレクトリは消さずに `.partialN` へ退避される。落ちた跡は原因調べに
要るので、そのまま置いておくこと。

`sweep.py` を**2つ同時に立てない**こと。別々のジョブなら出力先が (モデル, A, seed)
で分かれるので衝突しないが、2つ立てると同じ未完ジョブ一覧を作って同じジョブを
同時に走らせる。

## 注意点

**`result_logs/` を旧マシンからコピーしない。** Llama-2-7b / A=6 の出力先は
report/06・07 と同じ名前なので（`slimpajama_w2_n16_seed0`、`bench_slimpajama_seed0`
など）、コピーすると `ensure()` が結果 JSON の中身を見て「済み」と判定し、
**旧マシンの数字が新しい表に混じる**。新しいマシンでは空から始めること。旧データを
参照したいなら別の場所（`result_logs_old/`）に置く。

**GPU をまたいで数字を混ぜない。** 効果量は `acc` で 0.005〜0.012 と小さく、
GPU が違えば浮動小数の積み方が変わって同じ桁の差が出かねない。5 seed を1台で
測り直すのはこのためで、旧マシンで測った dense を取り込まないようにコードからも
外してある（`run.py` の `dense_reference()`）。

**PPL はモデルをまたいで比較できない。** トークナイザが違うため、WikiText-2 の
トークン数からして違う（Llama 341,469 / Mistral 334,660）。表を分けるか dense 比で
出すこと。

**torch を差し替えるなら手順4 からやり直す。** cu128 が通らない場合の退避先は
cu126 系である（cu124 のホイールは torch 2.6 までで、いまの pin は
`torch>=2.7,<2.9`）。`pyproject.toml` の index を変えて `uv lock` → `docker build`
→ `--check` からやり直す。数値の環境が変わるということなので、途中からは繋げない。

## この手順に入っていないもの

**配分の対照6行**（EW-rule / OWL相当 / LExI相当 / 順序対照×3）は
`experiments/{21_ew_rule,22_alloc_baselines}` にあり、**まだモデル引数化されて
いない**。20ジョブが出すのは一様配分（x=0..A）と探索（幅2/3/4）だけである。
§4 の比較表を7行にするには、22 を 25 と同じ要領で `--model` を通す作業が要る。
Llama-2-7b のぶんは旧マシンで seed 0〜2 まで測ってあるが、上の「GPU をまたいで
混ぜない」に従うなら引き直すことになる。

**20ジョブの集計。** `experiments/25_model_seeds/` に集計スクリプトは**まだ無い**。
`experiments/06_slimpajama_seeds/summarize_seeds.py` は Llama・A=6・PPL と5タスク
しか読めない（札なしのパスに固定されている）ので、Mistral・A=4・MMLU を含む
主表を作るには新しく書く必要がある。
