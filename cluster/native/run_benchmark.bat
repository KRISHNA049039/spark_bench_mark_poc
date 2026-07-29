@echo off
REM ============================================
REM Run the 3-Phase Cluster Benchmark (Node 1)
REM ============================================

set SPARK_HOME=%~dp0spark\spark-3.5.1-bin-hadoop3
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%

REM Spark configuration
set SPARK_MASTER=spark://192.168.4.100:7077
set SPARK_LOCAL_HOSTNAME=192.168.4.100
set SPARK_DRIVER_HOST=192.168.4.100
set SPARK_DRIVER_PORT=33000
set SPARK_DRIVER_BLOCKMANAGER_PORT=33005
set SPARK_PUBLIC_DNS=192.168.4.100

REM Executor settings (optimized for 32 GB RAM workers)
set SPARK_DRIVER_MEMORY=4g
set SPARK_EXECUTOR_MEMORY=12g
set SPARK_EXECUTOR_MEMORY_OVERHEAD=2g
set SPARK_EXECUTOR_CORES=4
set SPARK_NUM_EXECUTORS=4

REM Benchmark settings
set BENCHMARK_OUTPUT_DIR=..\..\benchmark_results
set BENCHMARK_SAMPLES=1000
set BENCHMARK_BATCH_SIZE=64
set BENCHMARK_PARTITIONS=8
set FORCE_GPU_PHASES=true

REM Python path (so imports work)
set PYTHONPATH=%~dp0..\..

REM === CHANGE THIS to the exact python.exe that has GPU torch installed ===
REM Must match what you set PYSPARK_PYTHON to on every worker node, or the
REM driver and executors can end up on different torch builds.
set PYSPARK_PYTHON=C:\Path\To\Your\GPU\python.exe
set PYSPARK_DRIVER_PYTHON=%PYSPARK_PYTHON%

echo ==========================================
echo GPU preflight check on driver node...
echo ==========================================
"%PYSPARK_PYTHON%" "%~dp0check_gpu.py"

echo ==========================================
echo 3-Phase Cluster Benchmark
echo Master: %SPARK_MASTER%
echo Driver: %SPARK_DRIVER_HOST%:%SPARK_DRIVER_PORT%
echo Samples: %BENCHMARK_SAMPLES%, Batch: %BENCHMARK_BATCH_SIZE%
echo ==========================================
echo.

python -m pytorch_benchmark.cluster_benchmark

echo.
echo Benchmark complete. Results in: benchmark_results\
echo.

REM Generate report
python -m pytorch_benchmark.generate_cluster_report

pause
