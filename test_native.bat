@echo off
REM ============================================================
REM Quick native test — runs benchmark in Spark local[*] mode
REM No master/worker needed. Tests GPU + CPU on this machine.
REM ============================================================

set PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe

for /f "delims=" %%i in ('"%PYTHON%" -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))"') do set SPARK_HOME=%%i
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%

set PYTHONPATH=%~dp0
set PYSPARK_PYTHON=%PYTHON%
set PYSPARK_DRIVER_PYTHON=%PYTHON%
set HADOOP_HOME=%SPARK_HOME%
set PYSPARK_HADOOP_VERSION=without
set SPARK_LOCAL_DIRS=C:\spark_tmp

REM Run in local mode — no cluster needed
set SPARK_MASTER=local[*]
set SPARK_DRIVER_HOST=127.0.0.1

REM Small run for quick verification
set BENCHMARK_SAMPLES=1000
set BENCHMARK_BATCH_SIZE=64
set BENCHMARK_PARTITIONS=4
set BENCHMARK_OUTPUT_DIR=%~dp0benchmark_results
set FORCE_GPU_PHASES=true

echo.
echo ============================================================
echo  NATIVE TEST — Spark local[*] mode
echo  Samples: 100  Batch: 32  (quick run)
echo ============================================================
echo.

"%PYTHON%" cluster\native\check_gpu.py
echo.

"%PYTHON%" -m pytorch_benchmark.cluster_benchmark

echo.
echo Done. Results in: benchmark_results\
echo.
pause
