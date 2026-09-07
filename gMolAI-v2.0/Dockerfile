# syntax=docker/dockerfile:1

# NVIDIA NGC PyG 25.09, pinned to its linux/amd64 manifest. The base provides
# Python 3.12, CUDA 13, PyTorch, PyG 2.7, and GPU code for compute capability
# 7.5 and newer (including Ampere, Hopper, and Blackwell).
FROM nvcr.io/nvidia/pyg@sha256:16388647ef8c7384d3f9394f742103b58854c41468e8ae1972162cdfcee9ffa3 AS runtime-amd64

ARG TARGETARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [[ "${TARGETARCH}" != "amd64" ]]; then \
        echo "gMolAI Docker images must be built with --platform linux/amd64" >&2; \
        exit 1; \
    fi

COPY requirements.lock /opt/gmolai/requirements.lock

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        jq \
        libgomp1 \
        procps \
        time \
    && python -m pip install --no-cache-dir -r /opt/gmolai/requirements.lock \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

COPY pyproject.toml README.md /opt/gmolai/
COPY src /opt/gmolai/src
COPY configs /opt/gmolai/configs
COPY inference /opt/gmolai/inference
COPY tests /opt/gmolai/tests

RUN python -m pip install --no-cache-dir --no-deps /opt/gmolai \
    && chmod 0555 /opt/gmolai/inference/gmolai.py \
    && ln -s /opt/gmolai/inference/gmolai.py /usr/local/bin/gmolai \
    && rm -rf /root/.cache/pip

ENV PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    TOKENIZERS_PARALLELISM="false" \
    OMP_NUM_THREADS="8" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor"

WORKDIR /work

RUN test "$(uname -m)" = "x86_64" \
    && gmolai --version \
    && gmolai-retrain --help >/dev/null \
    && gmolai validate --device cpu >/dev/null \
    && python -c "import duckdb, pandas, pyarrow, rdkit, scipy, sklearn, torch, torch_geometric, yaml; assert torch_geometric.__version__.split('.')[0] == '2'; required = {'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120'}; actual = set(torch._C._cuda_getArchFlags().split()); assert required <= actual, sorted(required - actual)"

LABEL org.opencontainers.image.title="gMolAI" \
    org.opencontainers.image.version="0.4.1" \
    org.opencontainers.image.source="https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0" \
    org.opencontainers.image.description="AMD64 CUDA gMolAI release inference and retraining runtime" \
    org.opencontainers.image.architecture="amd64" \
    org.opencontainers.image.base.name="nvcr.io/nvidia/pyg:25.09-py3" \
    org.opencontainers.image.base.digest="sha256:16388647ef8c7384d3f9394f742103b58854c41468e8ae1972162cdfcee9ffa3" \
    gmolai.cuda.version="13.0.1" \
    gmolai.pytorch.version="2.9.0a0+50eac811a6" \
    gmolai.pyg.version="2.7.0" \
    gmolai.gpu.compute-capability="7.5+"

ENTRYPOINT ["gmolai"]
CMD ["--help"]
