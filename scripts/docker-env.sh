# 実験用コンテナを呼ぶための関数。**source して使う**（実行ではない）。
#
#   source scripts/docker-env.sh
#   cmoe1 uv run python experiments/25_model_seeds/preflight.py
#
# シェルを開き直すたびに source し直すこと。28時間の実行中に再接続したら、
# もう一度これを読んでから docker logs を見にいく。
#
# 手順の全体は docs/05_running-the-sweep.md にある。

# このファイルの位置からリポジトリ直下を決める。**cwd に依存させない** —
# 相対パスでマウントすると、別のディレクトリから呼んだときに .cache と
# result_logs が黙って別の場所を向く
CMOE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
CMOE_IMAGE=kinoshita/cmoe-improve
CMOE_NAME=kinoshita_cmoe-improve
CMOE_MOUNTS="-v $HOME/.cache/huggingface:/root/.cache/huggingface"
CMOE_MOUNTS="$CMOE_MOUNTS -v $CMOE_REPO/.cache:/workspace/.cache"
CMOE_MOUNTS="$CMOE_MOUNTS -v $CMOE_REPO/result_logs:/workspace/result_logs"

# $CMOE_MOUNTS はクォートしない。空白で区切って複数の -v に割れてほしいところ

# HuggingFace の認証。ふつうはマウントした ~/.cache/huggingface/token が使われる
# ので何もしなくてよい。**HF_TOKEN が設定されているときだけ**渡す — 空のまま
# 渡すとファイル側のトークンを上書きして、かえって認証が通らなくなる
CMOE_AUTH=""
if [ -n "${HF_TOKEN:-}" ]; then
    CMOE_AUTH="-e HF_TOKEN=$HF_TOKEN"
fi

# 1枚だけ見せる。**sweep.py 以外はこちら。** アダプタは device_map='auto' で
# 読むので、2枚以上見えていると7Bが分割され、その実行が測る値だけが別の
# デバイス構成のものになる
cmoe1() {
    docker run --rm --name "$CMOE_NAME" --gpus all -e CUDA_VISIBLE_DEVICES=0 \
        $CMOE_MOUNTS $CMOE_AUTH "$CMOE_IMAGE" "$@"
}

# HuggingFace にログインする。**ホストに何も入れなくてよい** — CLI はコンテナの
# venv にあり、~/.cache/huggingface を read-write でマウントしているので、
# 中で書いたトークンはホスト側のファイルに残る（コンテナを捨てても消えない）。
# 対話的に訊かれるので -it が要る。GPU も要らない
cmoe_login() {
    docker run --rm -it $CMOE_MOUNTS "$CMOE_IMAGE" \
        uv run huggingface-cli login "$@"
}

# 全部見せる。**preflight だけ。** 枚数を数えるのが仕事なので絞らない
cmoeall() {
    docker run --rm --name "$CMOE_NAME" --gpus all $CMOE_MOUNTS $CMOE_AUTH "$CMOE_IMAGE" "$@"
}

# 20ジョブを配る常駐コンテナ。--rm を付けない（終了後に docker logs で追う）
# ので、次に回す前に docker rm "$CMOE_NAME" で片付ける
cmoe_sweep() {
    docker run -d --name "$CMOE_NAME" --gpus all \
        $CMOE_MOUNTS $CMOE_AUTH "$CMOE_IMAGE" \
        uv run python experiments/25_model_seeds/sweep.py --gpus 0,1,2,3
}

cmoe_env() {
    echo "REPO   $CMOE_REPO"
    echo "IMAGE  $CMOE_IMAGE"
    echo "NAME   $CMOE_NAME"
    echo "MOUNTS $CMOE_MOUNTS"
    echo "AUTH   ${CMOE_AUTH:-（HF_TOKEN 未設定。マウントした ~/.cache/huggingface/token を使う）}"
}
