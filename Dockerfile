# 実験を回すための像。GPU 1枚につき1コンテナを立て、experiments/25 のジョブを
# 配る（experiments/25_model_seeds/sweep.py）。
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
#   docker build -t cmoe .
#   docker run --rm --gpus all cmoe uv run python experiments/25_model_seeds/preflight.py
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

# 実行時に uv が解決し直さない。4本が同時に venv を触るのを避ける
ENV UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# マウントするもの（いずれもホストと共有する）:
#   /root/.cache/huggingface  モデル14GB とデータセット。読むだけなので4本で共有可
#   /workspace/.cache         lm-eval のデータセットキャッシュ
#   /workspace/result_logs    一次データ。ジョブごとにディレクトリが分かれる
VOLUME ["/root/.cache/huggingface", "/workspace/.cache", "/workspace/result_logs"]

CMD ["bash"]
