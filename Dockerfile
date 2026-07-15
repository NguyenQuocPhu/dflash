FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv

ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    gcc-11 \
    g++-11 \
    ninja-build \
    ccache \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-11 \
    CXX=/usr/bin/g++-11

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv python install "${PYTHON_VERSION}" && \
    uv venv "${VIRTUAL_ENV}" \
        --python "${PYTHON_VERSION}" \
        --seed

WORKDIR /workspace/dflash
COPY . .

# Cài DFlash base.
RUN uv pip install -e .

# Cài chính xác phiên bản vLLM mà bài chấm yêu cầu.
RUN uv pip install "vllm==0.22.1"

