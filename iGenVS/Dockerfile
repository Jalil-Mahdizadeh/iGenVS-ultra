# syntax=docker/dockerfile:1

ARG CUDA_VERSION=12.8.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS engines-amd64

ARG TARGETARCH
ARG CUDA_ARCHITECTURES="80;86;89;90;100;120"
ARG ADGPU_TARGETS="80 86 89 90 100 120"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [[ "${TARGETARCH}" != "amd64" ]]; then \
        echo "iGenVS Docker images must be built with --platform linux/amd64" >&2; \
        exit 1; \
    fi

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
        meson \
        ninja-build \
        patch \
        tcsh \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/Uni-Dock/unidock /src/unidock
COPY containers/patches/unidock-1.2.0-respect-max-gpu-memory.patch /src/unidock.patch
COPY containers/patches/unidock-1.2.0-scores-only-output.patch /src/unidock-scores.patch

RUN sed -i 's/\r$//' \
        /src/unidock.patch \
        /src/unidock-scores.patch \
        /src/unidock/src/lib/vina.cpp \
        /src/unidock/src/lib/vina.h \
        /src/unidock/src/main/main.cpp \
    && patch -d /src/unidock -p1 < /src/unidock.patch \
    && patch -l -d /src/unidock -p1 < /src/unidock-scores.patch \
    && cmake \
        -S /src/unidock \
        -B /tmp/unidock-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/opt/unidock \
    && cmake --build /tmp/unidock-build --target unidock --parallel "$(nproc)" \
    && cmake --install /tmp/unidock-build --component Unspecified \
    && strip /opt/unidock/bin/unidock

COPY third_party/AutoGrid /src/autogrid

RUN sed -i 's/\r$//' \
        /src/autogrid/ad4_shared/paramdat2h.csh \
        /src/autogrid/ad4_shared/AD4_parameters.dat \
        /src/autogrid/ad4_shared/AD4.1_bound.dat \
    && meson setup \
        /src/autogrid/builddir \
        /src/autogrid \
        --prefix=/opt/autogrid \
        --buildtype=release \
    && meson compile -C /src/autogrid/builddir \
    && meson install -C /src/autogrid/builddir

COPY third_party/AutoDock-GPU /src/autodock-gpu

# Docker builds do not receive the host driver. AutoDock-GPU expects libcuda,
# libcudart, and libcurand in one link directory, so expose the devel image's
# libcuda stub beside the toolkit libraries only for this build step. The
# NVIDIA Container Toolkit injects the actual driver at runtime.
RUN test ! -e /usr/local/cuda/lib64/libcuda.so \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
    && export GPU_INCLUDE_PATH=/usr/local/cuda/include \
    && export GPU_LIBRARY_PATH=/usr/local/cuda/lib64 \
    && cd /src/autodock-gpu \
    && find . -type f -name '*.sh' -exec sed -i 's/\r$//' {} + \
    && mkdir -p bin /opt/autodock-gpu/bin \
    && make DEVICE=CUDA NUMWI=64 TARGETS="${ADGPU_TARGETS}" OVERLAP=ON odock \
    && cp bin/autodock_gpu_64wi /opt/autodock-gpu/bin/ \
    && make DEVICE=CUDA NUMWI=128 TARGETS="${ADGPU_TARGETS}" OVERLAP=ON odock \
    && cp bin/autodock_gpu_128wi /opt/autodock-gpu/bin/ \
    && make DEVICE=CUDA NUMWI=256 TARGETS="${ADGPU_TARGETS}" OVERLAP=ON odock \
    && cp bin/autodock_gpu_256wi /opt/autodock-gpu/bin/ \
    && strip /opt/autodock-gpu/bin/autodock_gpu_*wi \
    && ln -s autodock_gpu_64wi /opt/autodock-gpu/bin/autodock_gpu \
    && rm /usr/local/cuda/lib64/libcuda.so


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime-amd64

ARG TARGETARCH
ARG CUDA_ARCHITECTURES="80;86;89;90;100;120"
ARG ADGPU_TARGETS="80 86 89 90 100 120"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [[ "${TARGETARCH}" != "amd64" ]]; then \
        echo "iGenVS Docker images must be built with --platform linux/amd64" >&2; \
        exit 1; \
    fi

COPY --from=engines-amd64 /opt/unidock /opt/unidock
COPY --from=engines-amd64 /opt/autodock-gpu /opt/autodock-gpu
COPY --from=engines-amd64 /opt/autogrid /opt/autogrid

COPY pyproject.toml README.md /opt/igenvs/
COPY src /opt/igenvs/src
COPY tests /opt/igenvs/tests
COPY examples /opt/igenvs/examples
COPY iGen3/pyproject.toml iGen3/README.md iGen3/requirements.txt /opt/iGen3/
COPY iGen3/src /opt/iGen3/src
COPY iGen3/models /opt/iGen3/models
COPY iGen3/examples /opt/iGen3/examples
COPY third_party/Uni-Dock/LICENSE /opt/licenses/Uni-Dock-LICENSE
COPY third_party/AutoDock-GPU/LICENSE /opt/licenses/AutoDock-GPU-GPL-2.0
COPY third_party/AutoDock-GPU/LICENSE_LGPL /opt/licenses/AutoDock-GPU-LGPL-2.1
COPY third_party/AutoGrid/COPYING /opt/licenses/AutoGrid-GPL-2.0
COPY third_party/Uni-Dock/unidock/test /opt/unidock-tests
COPY third_party/AutoDock-GPU/input /opt/autodock-gpu-tests

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libboost-serialization1.74.0 \
        libboost-system1.74.0 \
        libboost-thread1.74.0 \
        libboost-timer1.74.0 \
        libgomp1 \
        openbabel \
        python3 \
        python3-dev \
        python3-pip \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
        torch==2.11.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
    && python3 -m pip install --no-cache-dir \
        gemmi==0.7.3 \
        meeko==0.8.0 \
        numpy==2.2.6 \
        pandas==2.3.2 \
        rdkit==2025.9.2 \
        scipy==1.15.3 \
        tqdm==4.67.1 \
        matplotlib==3.10.5 \
        pytest==8.4.2 \
    && python3 -m pip install --no-cache-dir --no-deps /opt/iGen3 \
    && python3 -m pip install --no-cache-dir --no-deps /opt/igenvs \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

ENV PATH="/opt/unidock/bin:/opt/autodock-gpu/bin:/opt/autogrid/bin:${PATH}" \
    IGEN3_MODEL_DIR="/opt/iGen3/models" \
    PYTHONUNBUFFERED="1" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor" \
    MPLCONFIGDIR="/tmp/matplotlib"

WORKDIR /work

RUN test "$(uname -m)" = "x86_64" \
    && igenvs --version \
    && igenvs prepare-target --help >/dev/null \
    && igenvs screen --help | grep -q -- --engine \
    && igen3 list-models \
    && python3 -c "import igenvs, igen3, torch, rdkit, meeko; assert len(igen3.MODEL_SPECS) == 4" \
    && smoke_dir="$(mktemp -d)" \
    && igenvs prepare-target \
        --complex /opt/autodock-gpu-tests/3ce3/3CE3.pdb \
        --ligand-id A:1FN:1401 \
        --output-dir "${smoke_dir}/target" >/dev/null \
    && test -s "${smoke_dir}/target/manifest.json" \
    && test "$(grep -c '^atom_par ' "${smoke_dir}/target/boron-silicon-atom_par.dat")" -eq 2 \
    && grep -q '^atom_par Si ' "${smoke_dir}/target/boron-silicon-atom_par.dat" \
    && grep -q '^atom_par B ' "${smoke_dir}/target/boron-silicon-atom_par.dat" \
    && grep -q 'Successful Completion' "${smoke_dir}/target/autogrid.log" \
    && rm -rf "${smoke_dir}"

LABEL org.opencontainers.image.title="iGenVS" \
    org.opencontainers.image.version="0.2.0" \
    org.opencontainers.image.source="https://github.com/Jalil-Mahdizadeh/iGenVS" \
    org.opencontainers.image.description="AMD64 iGen3 + RDKit/Meeko + Uni-Dock 1.2.0 + AutoDock-GPU 1.6 virtual screening" \
    org.opencontainers.image.architecture="amd64" \
    igenvs.unidock.version="1.2.0" \
    igenvs.unidock.commit="95e409172b15dec0989aea70b0f2328e8ca52025" \
    igenvs.unidock.patch="respect-max-gpu-memory,scores-only-output" \
    igenvs.autodock-gpu.version="1.6" \
    igenvs.autodock-gpu.commit="e63e6f6280ebfad18caa3e8f48afdc269e79e063" \
    igenvs.autodock-gpu.work-items="64,128,256" \
    igenvs.autodock-gpu.default-work-items="64" \
    igenvs.autogrid.version="4.2.9" \
    igenvs.autogrid.commit="6d2847beaeac8ff43ca99094707fd74e3ca1ff37" \
    igenvs.cuda.architectures="${CUDA_ARCHITECTURES}" \
    igenvs.autodock-gpu.cuda-targets="${ADGPU_TARGETS}" \
    igenvs.igen3.models="base-isomeric,base-nonisomeric,rl-isomeric,rl-nonisomeric"

ENTRYPOINT ["igenvs"]
CMD ["--help"]
