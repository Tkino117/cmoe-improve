# 実験用コンテナを呼ぶための関数。**source して使う**（実行ではない）。
#
#   source scripts/docker-env.sh
#   cmoe1 uv run python experiments/25_model_seeds/preflight.py
#
# シェルを開き直すたびに source し直すこと。28時間の実行中に再接続したら、
# もう一度これを読んでから docker logs を見にいく。
#
# **共有マシンを汚さない作りにしてある。** ホームには何も書かず、書くのは
# $CMOE_CACHE と $CMOE_RESULTS の2つだけ。どちらも既定はリポジトリの中で、
# 環境変数で別の場所（scratch など）へ移せる。ファイルは呼んだ人の uid で
# 作られるので、root 所有の消せないゴミが残らない。
#
# 手順の全体は docs/05_running-the-sweep.md にある。

# このファイルの位置からリポジトリ直下を決める。**cwd に依存させない** —
# 相対パスでマウントすると、別のディレクトリから呼んだときに黙って別の場所を向く
CMOE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
CMOE_IMAGE=${CMOE_IMAGE:-kinoshita/cmoe-improve}
CMOE_NAME=${CMOE_NAME:-kinoshita_cmoe-improve}

# 書き込む先は2つだけ。**ホームは触らない。** モデル28GB とデータセットは
# $CMOE_CACHE に入るので、共有ホームの容量を食わない。ディスクを分けたいなら
# source する前に CMOE_CACHE=/scratch/$USER/cmoe-cache のように設定する
CMOE_CACHE=${CMOE_CACHE:-$CMOE_REPO/.cache}
CMOE_RESULTS=${CMOE_RESULTS:-$CMOE_REPO/result_logs}
mkdir -p "$CMOE_CACHE" "$CMOE_RESULTS"

CMOE_MOUNTS="-v $CMOE_CACHE:/workspace/.cache"
CMOE_MOUNTS="$CMOE_MOUNTS -v $CMOE_RESULTS:/workspace/result_logs"
# $CMOE_MOUNTS はクォートしない。空白で区切って複数の -v に割れてほしいところ

# **呼んだ人の uid で走らせる。** 既定（root）だと、マウントした先に root 所有の
# ファイルが残り、共有マシンでは他の人が消せない。$HOME をコンテナ内で
# /workspace/.cache に向けるのは、非 root だと /root が書けないためである。
# 何か壊れたら CMOE_USER="" にして root で走らせられる（そのときは出力の
# 所有者に注意すること）
CMOE_USER=${CMOE_USER-"--user $(id -u):$(id -g) -e HOME=/workspace/.cache"}

# HuggingFace の認証。**HF_TOKEN が設定されているときだけ**渡す — 空のまま
# 渡すと $CMOE_CACHE/huggingface/token を上書きして、かえって認証が通らなくなる。
# 共有アカウントならトークンをファイルに置かず、こちらを使うほうがよい
CMOE_AUTH=""
if [ -n "${HF_TOKEN:-}" ]; then
    CMOE_AUTH="-e HF_TOKEN=$HF_TOKEN"
fi

# 1枚だけ見せる。**sweep.py 以外はこちら。** アダプタは device_map='auto' で
# 読むので、2枚以上見えていると7Bが分割され、その実行が測る値だけが別の
# デバイス構成のものになる
cmoe1() {
    docker run --rm --name "$CMOE_NAME" --gpus all -e CUDA_VISIBLE_DEVICES=0 \
        $CMOE_USER $CMOE_MOUNTS $CMOE_AUTH "$CMOE_IMAGE" "$@"
}

# 全部見せる。**preflight だけ。** 枚数を数えるのが仕事なので絞らない
cmoeall() {
    docker run --rm --name "$CMOE_NAME" --gpus all \
        $CMOE_USER $CMOE_MOUNTS $CMOE_AUTH "$CMOE_IMAGE" "$@"
}

# HuggingFace にログインする。ホストには何も入れなくてよい（CLI はコンテナの
# venv にある）。対話的に訊かれるので -it が要り、GPU は要らない。
#
# **共有アカウントでは勧めない。** トークンが $CMOE_CACHE/huggingface/token に
# 平文で残り、そのディレクトリを読める人には見える。代わりに
# `export HF_TOKEN=hf_...` をシェルに置けば、ディスクに残らない
cmoe_login() {
    docker run --rm -it $CMOE_USER $CMOE_MOUNTS "$CMOE_IMAGE" \
        uv run huggingface-cli login "$@"
}

# 20ジョブを配る常駐コンテナ。--rm を付けない（終了後に docker logs で追う）
# ので、次に回す前に docker rm "$CMOE_NAME" で片付ける
cmoe_sweep() {
    docker run -d --name "$CMOE_NAME" --gpus all \
        $CMOE_USER $CMOE_MOUNTS $CMOE_AUTH "$CMOE_IMAGE" \
        uv run python experiments/25_model_seeds/sweep.py --gpus 0,1,2,3
}

cmoe_env() {
    echo "REPO    $CMOE_REPO"
    echo "IMAGE   $CMOE_IMAGE"
    echo "NAME    $CMOE_NAME"
    echo "CACHE   $CMOE_CACHE      （モデル28GB とデータセット。ここだけ育つ）"
    echo "RESULTS $CMOE_RESULTS"
    echo "USER    ${CMOE_USER:-（未指定 = root で走る。出力が root 所有になる）}"
    echo "AUTH    ${CMOE_AUTH:-（HF_TOKEN 未設定。$CMOE_CACHE/huggingface/token を使う）}"
    echo
    echo "ホームには何も書かない。消したいときは上の CACHE と RESULTS だけ消す"
}
