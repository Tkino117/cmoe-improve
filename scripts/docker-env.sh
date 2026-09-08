# 実験用コンテナを呼ぶための関数。**source して使う**（実行ではない）。
#
#   source scripts/docker-env.sh
#   cmoe1 uv run python experiments/25_model_seeds/preflight.py
#
# シェルを開き直すたびに source し直すこと。28時間の実行中に再接続したら、
# もう一度これを読んでから docker logs を見にいく。
#
# **共有マシンを汚さない作りにしてある。** 書き込むのは /data/kinoshita の下
# （$CMOE_CACHE と $CMOE_RESULTS）だけで、共有のホームにもリポジトリにも
# 何も書かない。ファイルは呼んだ人の uid で作られるので、root 所有の消せない
# ゴミも残らない。片付けは /data/kinoshita を消すだけでよい。
#
# 手順の全体は docs/05_running-the-sweep.md にある。

# このファイルの位置からリポジトリ直下を決める。**cwd に依存させない** —
# 相対パスでマウントすると、別のディレクトリから呼んだときに黙って別の場所を向く
CMOE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
CMOE_IMAGE=${CMOE_IMAGE:-kinoshita/cmoe-improve}
CMOE_NAME=${CMOE_NAME:-kinoshita_cmoe-improve}

# 書き込む先は2つだけで、どちらも /data/kinoshita の下に固定してある。
# **ホームにもリポジトリにも書かない** — アカウントを共有しているので、
# 28GB のモデルや一次データを共有ホームや共有のリポジトリに置かない。
# 片付けたいときは、この2つを消せば全部消える。
#
# 別の場所にしたいときだけ、source する前に環境変数で上書きする
CMOE_DATA=${CMOE_DATA:-/data/kinoshita}
CMOE_CACHE=${CMOE_CACHE:-$CMOE_DATA/cmoe-cache}
CMOE_RESULTS=${CMOE_RESULTS:-$CMOE_DATA/cmoe-results}

# 置き場所を1つ確かめる。**作れない／書けないなら、その場で理由を言う。**
# 黙って先へ進むと、docker がホスト側に空のディレクトリを root 所有で作り、
# 共有マシンでは他の人が消せないゴミになる
_cmoe_check_dir() {
    _label=$1
    _path=$2
    _hint=$3
    if [ ! -d "$_path" ]; then
        if ! mkdir -p "$_path" 2>/dev/null; then
            echo "  [NG] $_label $_path を作れない" >&2
            echo "       親ディレクトリ $(dirname "$_path") が無いか書けない。" >&2
            echo "       用意するか、別の場所を指すこと:" >&2
            echo "         mkdir -p $(dirname "$_path")" >&2
            echo "         CMOE_DATA=<書ける場所> source scripts/docker-env.sh" >&2
            return 1
        fi
    fi
    if [ ! -w "$_path" ]; then
        echo "  [NG] $_label $_path に書けない（所有者 $(stat -c '%U:%G' "$_path" 2>/dev/null)）" >&2
        if [ "$(stat -c '%U' "$_path" 2>/dev/null)" = root ]; then
            echo "       root で走らせた実行が作ったものらしい。取り戻すか:" >&2
            echo "         sudo chown -R \$(id -u):\$(id -g) $_path" >&2
            echo "       別の場所を指す:" >&2
            echo "         CMOE_DATA=<書ける場所> source scripts/docker-env.sh" >&2
        else
            echo "       CMOE_DATA=<書ける場所> で別の場所を指すこと" >&2
        fi
        return 1
    fi
    return 0
}

CMOE_READY=1
_cmoe_check_dir CMOE_CACHE "$CMOE_CACHE" cache || CMOE_READY=0
_cmoe_check_dir CMOE_RESULTS "$CMOE_RESULTS" results || CMOE_READY=0
if [ "$CMOE_READY" = 0 ]; then
    echo "  置き場所が用意できていない。cmoe_env で確認すること" >&2
fi
unset _label _path _hint

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

# 20ジョブの進み具合。**ホスト側のファイルを読むだけ**なので、sweep が回って
# いる最中に何度呼んでもよいし、docker が要らない。
#
#   済   段の一覧（exp25_stages_*.json）が書かれた
#   走行 ログはあるが一覧がまだ無い
#   未   ログも無い
cmoe_progress() {
    _done=0
    _running=0
    for _model in llama2-7b mistral-7b; do
        for _na in 6 4; do
            for _seed in 0 1 2 3 4; do
                _job="${_model}_a${_na}_seed${_seed}"
                _stages="$CMOE_RESULTS/exp25_stages_${_job}.json"
                _joblog="$CMOE_RESULTS/sweep_logs/${_job}.log"
                if [ -f "$_stages" ]; then
                    _done=$((_done + 1))
                    printf '  済   %-26s %s\n' "$_job" \
                        "$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('%.1f時間'%(d.get('seconds',0)/3600))" "$_stages" 2>/dev/null)"
                elif [ -f "$_joblog" ]; then
                    _running=$((_running + 1))
                    printf '  走行 %-26s %s\n' "$_job" \
                        "$(grep -E '^=== 段' "$_joblog" | tail -1)"
                fi
            done
        done
    done
    echo
    echo "  済 $_done / 20   走行 $_running   未 $((20 - _done - _running))"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
    fi
    echo "  配り役のログ: docker logs --tail 20 $CMOE_NAME"
    echo "  ジョブ個別  : tail -f $CMOE_RESULTS/sweep_logs/<job>.log"
    unset _done _running _model _na _seed _job _stages _joblog
}

cmoe_env() {
    echo "REPO    $CMOE_REPO"
    echo "DATA    $CMOE_DATA      （書き込むのはこの下だけ）"
    echo "IMAGE   $CMOE_IMAGE"
    echo "NAME    $CMOE_NAME"
    echo "CACHE   $CMOE_CACHE $(_cmoe_state "$CMOE_CACHE")"
    echo "        （モデル28GB とデータセット。ここだけ育つ）"
    echo "RESULTS $CMOE_RESULTS $(_cmoe_state "$CMOE_RESULTS")"
    echo "USER    ${CMOE_USER:-（未指定 = root で走る。出力が root 所有になる）}"
    echo "AUTH    ${CMOE_AUTH:-（HF_TOKEN 未設定。$CMOE_CACHE/huggingface/token を使う）}"
    echo
    if [ "${CMOE_READY:-1}" = 0 ]; then
        echo "**置き場所が用意できていない。** 上の [NG] を直してから source し直す"
    else
        echo "ホームには何も書かない。消したいときは上の CACHE と RESULTS だけ消す"
    fi
}

_cmoe_state() {
    if [ ! -d "$1" ]; then
        echo "[NG 無い]"
    elif [ ! -w "$1" ]; then
        echo "[NG 書けない / 所有者 $(stat -c '%U:%G' "$1" 2>/dev/null)]"
    else
        echo "[OK $(stat -c '%U:%G' "$1" 2>/dev/null)]"
    fi
}
