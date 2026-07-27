# ===========================================================================
# PyTorch Benchmark Suite - Docker Environment
#
# Multi-stage build:
#   - Base: Python + system deps + Java (for Spark)
#   - Final: All Python packages + benchmark code
#
# Build targets:
#   docker build --target cpu -t pytorch-benchmark:cpu .
#   docker build --target gpu -t pytorch-benchmark:gpu .
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: CPU Base
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS cpu-base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    procps \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Java 17 (Temurin JRE - direct download)
RUN mkdir -p /opt/java && \
    wget -q https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jre_x64_linux_hotspot_17.0.11_9.tar.gz -O /tmp/java.tar.gz && \
    tar -xzf /tmp/java.tar.gz -C /opt/java --strip-components=1 && \
    rm /tmp/java.tar.gz

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Python dependencies
WORKDIR /app

COPY pytorch_benchmark/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    pyspark==3.5.1 \
    psutil==5.9.8 \
    GPUtil==1.4.0 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.2 \
    tabulate==0.9.0 \
    matplotlib==3.8.4

# ---------------------------------------------------------------------------
# Stage 2: CPU Final
# ---------------------------------------------------------------------------
FROM cpu-base AS cpu

# Copy benchmark code
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
# Stage 3: GPU Base (NVIDIA CUDA)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS gpu-base

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies + Java + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    procps \
    gcc \
    g++ \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Java 17 (direct download - avoids apt repo issues)
RUN mkdir -p /opt/java && \
    wget -q https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jre_x64_linux_hotspot_17.0.11_9.tar.gz -O /tmp/java.tar.gz && \
    tar -xzf /tmp/java.tar.gz -C /opt/java --strip-components=1 && \
    rm /tmp/java.tar.gz

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

COPY pytorch_benchmark/requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.8 (required for RTX 5060 Blackwell sm_120)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --pre \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir \
    pyspark==3.5.1 \
    psutil==5.9.8 \
    GPUtil==1.4.0 \
    pandas==2.2.1 \
    scikit-learn==1.4.2 \
    tabulate==0.9.0 \
    matplotlib==3.8.4

# ---------------------------------------------------------------------------
# Stage 4: GPU Final
# ---------------------------------------------------------------------------
FROM gpu-base AS gpu

# Copy benchmark code
COPY pytorch_benchmark/ /app/pytorch_benchmark/

# Environment configuration
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV BENCHMARK_OUTPUT_DIR=/app/benchmark_results
ENV SPARK_MASTER=local[*]
ENV SPARK_DRIVER_MEMORY=4g
ENV SPARK_EXECUTOR_MEMORY=4g
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create output directory
RUN mkdir -p /app/benchmark_results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

ENTRYPOINT ["python", "-m", "pytorch_benchmark.main"]
CMD []
