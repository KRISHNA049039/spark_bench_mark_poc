@echo off
REM ============================================
REM Run 3-Phase Benchmark on SINGLE MACHINE
REM Uses: All CPU cores + RTX 5060 GPU
REM No networking, no Docker, no remote workers
REM ============================================

cd /d d:\spark_pytorch_poc

REM Spark runs locally using all cores
set SPARK_MASTER=local[*]
set SPARK_DRIVER_MEMORY=4g
set SPARK_EXECUTOR_MEMORY=4g
set SPARK_EXECUTOR_CORES=4

REM Benchmark config
set BENCHMARK_OUTPUT_DIR=benchmark_results
set BENCHMARK_SAMPLES=200
set BENCHMARK_BATCH_SIZE=64
set BENCHMARK_PARTITIONS=4
set FORCE_GPU_PHASES=true

REM Run single model (change this to test different models)
set BENCHMARK_MODELS=efficientnet_b0

echo ==========================================
echo SINGLE-MACHINE CLUSTER BENCHMARK
echo Mode: local[*] (all CPU cores + GPU)
echo Model: %BENCHMARK_MODELS%
echo Samples: %BENCHMARK_SAMPLES%
echo ==========================================
echo.

python -m pytorch_benchmark.cluster_benchmark_low_rpc

echo.
echo Done! Results in benchmark_results\
pause
