#!/usr/bin/env bash
# ============================================================
# Install GPU PyTorch on a Linux worker/dev machine
# (For RTX 5000-series / Blackwell sm_120)
#
# Bash equivalent of install_gpu_worker.bat — installs the exact same
# pinned versions as the GPU Docker images and the Windows native install
# (pytorch_benchmark/requirements-*.txt), so results stay comparable.
#
# This is an ONLINE install (needs internet access). It does not produce
# an airgap-transferable bundle — for an airgapped Linux target, the
# shipped path is the Docker images (see airgap/ARCHITECTURE.md); this
# script is for a Linux dev/test box with internet access.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "Installing PyTorch 2.11.0 with CUDA 12.8 support (for RTX 50-series)..."
echo "This requires NVIDIA driver 570.26+ installed (see airgap/TESTING.md §0)."
echo

"$PYTHON" -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    -r "$ROOT/pytorch_benchmark/requirements-torch-gpu.txt"

echo
echo "Installing other dependencies..."
"$PYTHON" -m pip install --no-cache-dir -r "$ROOT/pytorch_benchmark/requirements-base.txt"

echo
echo "Verifying GPU..."
"$PYTHON" -c "import torch; t=torch.tensor([2.0]).cuda(); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Test: {t*t}'); print('SUCCESS')"
