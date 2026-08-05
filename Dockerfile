# ===========================================================================
# PyTorch Benchmark Suite - Docker Environment
#
# Single Python version (3.12, official slim base) across CPU and GPU
# targets — matches the native install and the Spark worker images, so
# results are comparable and airgap wheel bundles work everywhere.
#
# LAYER CACHE STRATEGY (rebuild only what changed):
#   1. FROM python:3.12-slim        - base OS + interpreter (bump only to change Python version)
#   2. apt-get (curl/wget/procps)   - system deps, effectively never changes
#   3. Java 17 (pinned URL)         - rarely changes
#   4. pip install torch/torchvision - large, isolated in its own layer
#   5. pip install requirements-base - the rest of the libs, changes most often
#   6. COPY pytorch_benchmark/       - app code, changes every dev iteration
# Each pip layer only invalidates when the specific requirements file it
# installs from changes, because that file is COPYed immediately before it.
#
# Build targets:
#   docker build --target cpu -t pytorch-benchmark:cpu .
#   docker build --target gpu -t pytorch-benchmark:gpu .
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: CPU Base
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS cpu-base

# System dependencies (no compiler needed — every pip package below ships a
# manylinux/win wheel for cp312, nothing builds from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Java 17 (Temurin JRE - direct download, pinned version)
RUN mkdir -p /opt/java && \
    wget -q https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jre_x64_linux_hotspot_17.0.11_9.tar.gz -O /tmp/java.tar.gz && \
    tar -xzf /tmp/java.tar.gz -C /opt/java --strip-components=1 && \
    rm /tmp/java.tar.gz

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# Torch layer — isolated so it stays cached when only requirements-base.txt changes
COPY pytorch_benchmark/requirements-torch-cpu.txt /app/requirements-torch-cpu.txt
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements-torch-cpu.txt

# Everything else — isolated so it stays cached when only torch's pin changes
COPY pytorch_benchmark/requirements-base.txt /app/requirements-base.txt
RUN pip install --no-cache-dir -r requirements-base.txt

# ---------------------------------------------------------------------------
# Stage 2: CPU Final
# ---------------------------------------------------------------------------
FROM cpu-base AS cpu

# Copy benchmark code (changes every dev iteration — kept as the last layer)
COPY pytorch_benchmark/ /app/pytorch_benchmark/

# Environment configuration
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV BENCHMARK_OUTPUT_DIR=/app/benchmark_results
ENV SPARK_MASTER=local[*]
ENV SPARK_DRIVER_MEMORY=2g
ENV SPARK_EXECUTOR_MEMORY=2g

# Create output directory
RUN mkdir -p /app/benchmark_results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; import pyspark; print('OK')"

ENTRYPOINT ["python", "-m", "pytorch_benchmark.main"]
CMD ["--cpu-only"]

# ---------------------------------------------------------------------------
# Stage 3: GPU Base (python:3.12-slim + CUDA 12.8 torch wheel)
#
# No CUDA base image needed: the torch cu128 wheel bundles its own CUDA/cuDNN
# runtime libs as pip dependencies. GPU access at `docker run --gpus all`
# time comes from the NVIDIA Container Toolkit / Docker Desktop mounting the
# host driver (libcuda.so, nvidia-smi) into the container regardless of base
# image — that's what NVIDIA_DRIVER_CAPABILITIES=compute,utility below is for.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS gpu-base

# System dependencies — same minimal set as CPU; no source build needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Java 17 (Temurin JRE - direct download, pinned version)
RUN mkdir -p /opt/java && \
    wget -q https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jre_x64_linux_hotspot_17.0.11_9.tar.gz -O /tmp/java.tar.gz && \
    tar -xzf /tmp/java.tar.gz -C /opt/java --strip-components=1 && \
    rm /tmp/java.tar.gz

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# Torch layer — isolated so it stays cached when only requirements-base.txt changes
COPY pytorch_benchmark/requirements-torch-gpu.txt /app/requirements-torch-gpu.txt
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    -r requirements-torch-gpu.txt

# Everything else — isolated so it stays cached when only torch's pin changes
COPY pytorch_benchmark/requirements-base.txt /app/requirements-base.txt
RUN pip install --no-cache-dir -r requirements-base.txt

# ---------------------------------------------------------------------------
# Stage 4: GPU Final
# ---------------------------------------------------------------------------
FROM gpu-base AS gpu

# Copy benchmark code (changes every dev iteration — kept as the last layer)
COPY pytorch_benchmark/ /app/pytorch_benchmark/

# Environment configuration
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV BENCHMARK_OUTPUT_DIR=/app/benchmark_results
ENV SPARK_MASTER=local[*]
ENV SPARK_DRIVER_MEMORY=4g
ENV SPARK_EXECUTOR_MEMORY=4g

# Create output directory
RUN mkdir -p /app/benchmark_results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

ENTRYPOINT ["python", "-m", "pytorch_benchmark.main"]
CMD []
