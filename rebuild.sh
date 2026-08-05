#!/usr/bin/env bash
# ============================================================
# ONE-COMMAND REBUILD — Linux / macOS
#
# Rebuilds all 4 Docker images (pytorch-benchmark + pytorch-spark-worker,
# CPU + GPU). There is no native (non-Docker) install path on Linux — see
# airgap/ARCHITECTURE.md — so this only builds images.
#
# Uses normal `docker build` (no --no-cache) on purpose: Dockerfile and
# Dockerfile.worker isolate the apt/Java layers from the pip layers, so
# bumping a library version in pytorch_benchmark/requirements-*.txt only
# invalidates that library's layer.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker is not running. Start it and retry."
    exit 1
fi

echo "[1/4] Building pytorch-benchmark:cpu ..."
docker build --target cpu -t pytorch-benchmark:cpu "$ROOT"

echo "[2/4] Building pytorch-benchmark:gpu ..."
docker build --target gpu -t pytorch-benchmark:gpu "$ROOT"

echo "[3/4] Building pytorch-spark-worker:cpu ..."
docker build --file "$ROOT/Dockerfile.worker" --target cpu -t pytorch-spark-worker:cpu "$ROOT"

echo "[4/4] Building pytorch-spark-worker:gpu ..."
docker build --file "$ROOT/Dockerfile.worker" --target gpu -t pytorch-spark-worker:gpu "$ROOT"

echo "[OK] All 4 Docker images built."
