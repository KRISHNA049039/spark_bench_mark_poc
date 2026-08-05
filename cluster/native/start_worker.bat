@echo off
REM ============================================
REM Start Spark Worker (run on Node 2 / Node 3)
REM ============================================

REM Use Spark bundled inside pyspark package
for /f "delims=" %%i in ('C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))"') do set SPARK_HOME=%%i
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%

REM === CHANGE THIS to the worker machine's own LAN IP ===
set SPARK_LOCAL_HOSTNAME=192.168.4.101
set SPARK_PUBLIC_DNS=192.168.4.101

REM === Master IP (Node 1) ===
set MASTER_IP=192.168.4.100

REM === CHANGE THIS to the exact python.exe that has GPU torch installed ===
REM (the same one you tested with check_gpu.py). If this is wrong or unset,
REM Spark falls back to whatever "python" is first on PATH, which is often
REM a CPU-only install -- tasks then silently run on CPU with no error.
set PYSPARK_PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe

echo ==========================================
echo GPU preflight check for this worker...
echo ==========================================
"%PYSPARK_PYTHON%" "%~dp0check_gpu.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: GPU preflight check FAILED on this node.
    echo This worker will run Spark GPU tasks on CPU instead, silently.
    echo Fix PYSPARK_PYTHON above or the GPU torch install, then re-run.
    echo.
    pause
)

echo ==========================================
echo Starting Spark Worker
echo Worker IP: %SPARK_LOCAL_HOSTNAME%
echo Master: spark://%MASTER_IP%:7077
echo Python: %PYSPARK_PYTHON%
echo Memory: 28g, Cores: all
echo ==========================================

call "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.worker.Worker ^
  spark://%MASTER_IP%:7077 ^
  --host %SPARK_LOCAL_HOSTNAME% ^
  --memory 28g ^
  --webui-port 8081

pause
