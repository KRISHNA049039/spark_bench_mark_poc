#!/usr/bin/env bash
# ============================================================
# AIRGAP SIMULATION TEST — Linux/Docker
#
# Bash port of simulate_airgap_test.bat's [B] Docker section. There is no
# Linux native install path (see airgap/ARCHITECTURE.md) — Linux only runs
# the Docker images, so this only covers the Docker checks (B1-B5).
#
# HOW THE SIMULATION WORKS:
#   - Docker is run with --network none (no internet)
#   - Spark runs local[*] so no real cluster is needed
#
# This tells you: "will it work on the real airgapped Linux machine?"
# ============================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/benchmark_results/airgap_test"
mkdir -p "$RESULTS"

LOG="$RESULTS/airgap_sim_linux_$(date +%Y%m%d_%H%M%S).log"
PASS=0
FAIL=0

echo "============================================================"
echo " AIRGAP SIMULATION TEST (Linux/Docker)"
echo " Log: $LOG"
echo "============================================================"
{
  echo "AIRGAP SIMULATION TEST (Linux/Docker)  $(date)"
  echo "============================================================"
} > "$LOG"

check() {
    local id="$1" desc="$2"
    shift 2
    echo "  [$id] $desc..."
    if "$@" >> "$LOG" 2>&1; then
        echo "  [$id] PASS -- $desc"
        echo "  [$id] PASS -- $desc" >> "$LOG"
        PASS=$((PASS+1))
    else
        echo "  [$id] FAIL -- $desc"
        echo "  [$id] FAIL -- $desc" >> "$LOG"
        FAIL=$((FAIL+1))
    fi
}

if ! docker info >/dev/null 2>&1; then
    echo "[SKIP] Docker is not running."
    exit 1
fi

echo
echo "[B] DOCKER ENVIRONMENT TESTS"
echo "-------------------------------------------------------"

check B1 "pytorch-benchmark:cpu image present" docker image inspect pytorch-benchmark:cpu
check B2 "pytorch-spark-worker:gpu image present" docker image inspect pytorch-spark-worker:gpu

check B3 "Docker CPU container (--network none) import check" \
    docker run --rm --network none --entrypoint python pytorch-benchmark:cpu \
    -c "import torch, pyspark; print('torch:', torch.__version__, 'pyspark:', pyspark.__version__)"

if command -v nvidia-smi >/dev/null 2>&1; then
    check B4 "Docker GPU worker (--network none) CUDA check" \
        docker run --rm --network none --gpus all pytorch-spark-worker:gpu \
        python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
else
    echo "  [B4] SKIP -- no nvidia-smi on this host (CPU-only machine, or NVIDIA Container Toolkit not installed yet)"
    echo "  [B4] SKIP -- no nvidia-smi on this host" >> "$LOG"
fi

# --network none blocks DNS so the container can't resolve its own hostname;
# pin SPARK_LOCAL_HOSTNAME/driver.host to loopback the same way the .bat does.
check B5 "Docker CPU container -- Spark local session" \
    docker run --rm --network none --entrypoint python \
    -e SPARK_LOCAL_HOSTNAME=127.0.0.1 \
    pytorch-benchmark:cpu \
    -c "from pyspark.sql import SparkSession; spark=SparkSession.builder.master('local[2]').appName('ag').config('spark.ui.enabled','false').config('spark.driver.host','127.0.0.1').config('spark.driver.bindAddress','127.0.0.1').getOrCreate(); t=spark.sparkContext.parallelize(range(100)).sum(); spark.stop(); assert t==4950; print('Spark sum OK:', t)"

TOTAL=$((PASS+FAIL))
echo
echo "============================================================"
echo " AIRGAP SIMULATION RESULTS (Linux/Docker)"
echo " Passed : $PASS / $TOTAL"
echo " Failed : $FAIL / $TOTAL"
echo " Log    : $LOG"
echo "============================================================"
{
  echo "PASSED: $PASS  FAILED: $FAIL"
} >> "$LOG"

if [ "$FAIL" -eq 0 ]; then
    echo " All tests passed — safe to run the real benchmark on this machine."
    echo "STATUS: ALL PASS" >> "$LOG"
else
    echo " Some tests failed — review the log before relying on this machine."
    echo "STATUS: SOME FAILURES" >> "$LOG"
    exit 1
fi
