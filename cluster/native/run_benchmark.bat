@echo off
REM ============================================
REM Run the 3-Phase Cluster Benchmark (Node 1)
REM ============================================

REM Use Spark bundled inside pyspark package (no manual download needed)
for /f "delims=" %%i in ('C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))"') do set SPARK_HOME=%%i
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot
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

REM Python with GPU torch (cu128) installed
set PYSPARK_PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe
set PYSPARK_DRIVER_PYTHON=%PYSPARK_PYTHON%

REM Suppress Hadoop winutils warnings on Windows (no HDFS needed for standalone cluster)
set HADOOP_HOME=%SPARK_HOME%
set PYSPARK_HADOOP_VERSION=without

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

"%PYSPARK_PYTHON%" -m pytorch_benchmark.cluster_benchmark

echo.
echo Benchmark complete. Results in: benchmark_results\
echo.

REM Generate report
"%PYSPARK_PYTHON%" -m pytorch_benchmark.generate_cluster_report

pause
