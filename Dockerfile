# 実験を回すための像。**1コンテナに GPU を全部見せ、その中の
# experiments/25_model_seeds/sweep.py が1ジョブ1枚で配る。** 1枚1コンテナに
# すると、各コンテナの sweep.py が同じ未完ジョブ一覧を作って同じジョブを
# 同時に走らせる。手順は docs/05_running-the-sweep.md にある。
#
# **sweep.py 以外（--check / --smoke / dense_ppl.py）は必ず1枚に絞ること。**
# アダプタは device_map='auto' で読むので、2枚以上見えていると7Bが分割され、
# その実行が測る値だけが別のデバイス構成のものになる。run.py の
# require_single_gpu() が止めるが、-e CUDA_VISIBLE_DEVICES=0 を付けて呼ぶ。
#
# **CUDA を像に入れない。** torch のホイールが CUDA ランタイムを同梱している
# ので、要るのはホスト側のドライバだけである。nvidia/cuda ベースを敷くと、
# 像の CUDA とホイールの CUDA という揃えるものが2つになる。
#
# ドライバの要件: torch 2.8 (cu128) は CUDA の minor version compatibility で
# ドライバ 525.60.13 以上なら動く。測定先は 535.309.01（CUDA 12.2）なので
# 満たしているが、**最初に preflight.py を通して確かめること**（sm_89 の
# カーネルが cu128 ホイールに入っているかまで見る）。
#
#   docker build -t kinoshita/cmoe-improve .
#   docker run --rm --name kinoshita_cmoe-improve --gpus all \
#       kinoshita/cmoe-improve uv run python experiments/25_model_seeds/preflight.py
#
FROM python:3.11-slim

# git は summarize() が commit を残すのに要る。build-essential は lap のビルド用
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /workspace

# 依存だけを先に解決して層に焼く。ソースを触っても再解決が走らないように、
# pyproject と lock だけを先に入れる
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p src/cmoe && touch src/cmoe/__init__.py \
    && uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# 実行時に uv が解決し直さない。4本が同時に venv を触るのを避ける。
#
# **書き込む先を /workspace/.cache に集める。** /root 以下に散らすと、
# --user で非 root として走らせたときに書けない。共有マシンでは root 所有の
# ファイルを残さないほうが大事なので、そちらに寄せてある
ENV UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv \
    TRITON_CACHE_DIR=/workspace/.cache/triton

# 非 root でも venv を読めるようにする（実行するだけで書きはしない）
RUN chmod -R a+rX /workspace/.venv

# マウントするのは2つだけ。**ホームには何も書かない**:
#   /workspace/.cache        モデル28GB・データセット・uv/triton のキャッシュ
#   /workspace/result_logs   一次データ。ジョブごとにディレクトリが分かれる
#
# VOLUME は宣言しない。マウントを忘れたときに匿名ボリュームが作られて、
# それ自体が消し忘れのゴミになる

CMD ["bash"]
