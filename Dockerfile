FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv

ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Công cụ cần để uv build vLLM từ source sau này.
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

# Cài uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo Python và virtual environment riêng.
RUN uv python install "${PYTHON_VERSION}" && \
    uv venv "${VIRTUAL_ENV}" \
        --python "${PYTHON_VERSION}" \
        --seed

# Copy source DFlash vào image.
WORKDIR /workspace/dflash
COPY . .

# Chỉ cài DFlash base, KHÔNG cài extra [vllm].
RUN uv pip install -e .

# Giữ container chạy để có thể docker exec vào.
CMD ["sleep", "infinity"]