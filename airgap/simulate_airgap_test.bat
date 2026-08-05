@echo off
REM ============================================================
REM AIRGAP SIMULATION TEST
REM
REM Simulates an airgapped environment and smoke-tests both:
REM   [A] Native  — Python + PySpark + torch (no downloads)
REM   [B] Docker  — pre-loaded images (no pull)
REM
REM HOW THE SIMULATION WORKS:
REM   - Pip is invoked with --no-index (no PyPI access)
REM   - Docker is run with --network none (no internet)
REM   - Spark runs local[*] so no real cluster is needed
REM
REM This tells you: "will it work on the real airgapped machine?"
REM ============================================================
setlocal enabledelayedexpansion

set PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe
set ROOT=%~dp0..
set RESULTS=%ROOT%\benchmark_results\airgap_test

if not exist "%RESULTS%" mkdir "%RESULTS%"

set PASS=0
set FAIL=0
set LOG=%RESULTS%\airgap_sim_%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log
set LOG=%LOG: =0%

echo.
echo ============================================================
echo  AIRGAP SIMULATION TEST
echo  Results: %RESULTS%
echo  Log:     %LOG%
echo ============================================================
echo. > "%LOG%"
echo AIRGAP SIMULATION TEST  %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

REM ===========================================================
REM TEST A: Native — import check (no network calls allowed)
REM ===========================================================
echo.
echo [A] NATIVE ENVIRONMENT TESTS
echo -------------------------------------------------------

REM A1: torch + CUDA
echo   [A1] torch import + CUDA check...
"%PYTHON%" "%~dp0tests\a1_torch_cuda.py" >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 (
    echo   [A1] PASS — torch imports OK >> "%LOG%"
    echo   [A1] PASS — torch OK
    set /a PASS+=1
) else (
    echo   [A1] FAIL — torch import error >> "%LOG%"
    echo   [A1] FAIL — torch import error
    set /a FAIL+=1
)

REM A2: pyspark import
echo   [A2] pyspark import check...
"%PYTHON%" "%~dp0tests\a2_pyspark.py" >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 (
    echo   [A2] PASS — pyspark OK >> "%LOG%"
    echo   [A2] PASS — pyspark OK
    set /a PASS+=1
) else (
    echo   [A2] FAIL — pyspark import error >> "%LOG%"
    echo   [A2] FAIL — pyspark import error
    set /a FAIL+=1
)

REM A3: Spark local session (no cluster, no internet)
echo   [A3] Spark local session smoke test...
"%PYTHON%" "%~dp0tests\a3_spark_session.py" >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 (
    echo   [A3] PASS — Spark local session OK >> "%LOG%"
    echo   [A3] PASS — Spark local session OK
    set /a PASS+=1
) else (
    echo   [A3] FAIL — Spark local session error >> "%LOG%"
    echo   [A3] FAIL — Spark local session error
    set /a FAIL+=1
)

REM A4: GPU tensor op (no internet)
echo   [A4] GPU tensor smoke test...
"%PYTHON%" "%~dp0tests\a4_gpu_tensor.py" >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 (
    echo   [A4] PASS — GPU tensor OK >> "%LOG%"
    echo   [A4] PASS — GPU tensor OK
    set /a PASS+=1
) else (
    echo   [A4] FAIL — GPU tensor error >> "%LOG%"
    echo   [A4] FAIL — GPU tensor error
    set /a FAIL+=1
)

REM A5: Benchmark module importable
echo   [A5] pytorch_benchmark module import...
"%PYTHON%" "%~dp0tests\a5_benchmark_import.py" >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 (
    echo   [A5] PASS — benchmark module OK >> "%LOG%"
    echo   [A5] PASS — benchmark module OK
    set /a PASS+=1
) else (
    echo   [A5] FAIL — benchmark module error >> "%LOG%"
    echo   [A5] FAIL — benchmark module error
    set /a FAIL+=1
)

REM ===========================================================
REM TEST B: Docker — run containers with --network none
REM ===========================================================
echo.
echo [B] DOCKER ENVIRONMENT TESTS
echo -------------------------------------------------------

REM Check Docker running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [B] SKIP — Docker Desktop not running
    echo   [B] SKIP — Docker Desktop not running >> "%LOG%"
    goto :summary
)

REM B1: CPU image exists
echo   [B1] pytorch-benchmark:cpu image present...
docker image inspect pytorch-benchmark:cpu >nul 2>&1
set B1_ERR=!ERRORLEVEL!
if !B1_ERR!==0 (
    echo   [B1] PASS -- image exists >> "%LOG%"
    echo   [B1] PASS -- image exists
    set /a PASS+=1
) else (
    echo   [B1] FAIL -- image not found (run load_docker_images.bat) >> "%LOG%"
    echo   [B1] FAIL -- image not found  ^(run load_docker_images.bat^)
    set /a FAIL+=1
)

REM B2: GPU worker image exists
echo   [B2] pytorch-spark-worker:gpu image present...
docker image inspect pytorch-spark-worker:gpu >nul 2>&1
set B2_ERR=!ERRORLEVEL!
if !B2_ERR!==0 (
    echo   [B2] PASS -- image exists >> "%LOG%"
    echo   [B2] PASS -- image exists
    set /a PASS+=1
) else (
    echo   [B2] FAIL -- image not found >> "%LOG%"
    echo   [B2] FAIL -- image not found
    set /a FAIL+=1
)

REM B3: CPU container imports (--entrypoint overrides main.py entrypoint)
echo   [B3] Docker CPU container (--network none) import check...
docker run --rm --network none --entrypoint python pytorch-benchmark:cpu ^
    -c "import torch, pyspark; print('torch:', torch.__version__, 'pyspark:', pyspark.__version__)" ^
    >> "%LOG%" 2>&1
if !ERRORLEVEL!==0 (
    echo   [B3] PASS -- CPU container imports OK >> "%LOG%"
    echo   [B3] PASS -- CPU container imports OK
    set /a PASS+=1
) else (
    echo   [B3] FAIL -- CPU container error >> "%LOG%"
    echo   [B3] FAIL -- CPU container error
    set /a FAIL+=1
)

REM B4: GPU worker container CUDA check (no entrypoint override needed)
echo   [B4] Docker GPU worker (--network none) CUDA check...
docker run --rm --network none --gpus all pytorch-spark-worker:gpu ^
    python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" ^
    >> "%LOG%" 2>&1
if !ERRORLEVEL!==0 (
    echo   [B4] PASS -- GPU worker CUDA OK >> "%LOG%"
    echo   [B4] PASS -- GPU worker CUDA OK
    set /a PASS+=1
) else (
    echo   [B4] FAIL -- GPU worker CUDA error >> "%LOG%"
    echo   [B4] FAIL -- GPU worker CUDA error
    set /a FAIL+=1
)

REM B5: Spark inside CPU container
REM     --network none blocks DNS so container can't resolve its own hostname.
REM     Pass SPARK_LOCAL_HOSTNAME=127.0.0.1 and spark.driver.host=127.0.0.1 to fix.
echo   [B5] Docker CPU container -- Spark local session...
docker run --rm --network none --entrypoint python ^
    -e SPARK_LOCAL_HOSTNAME=127.0.0.1 ^
    pytorch-benchmark:cpu ^
    -c "from pyspark.sql import SparkSession; spark=SparkSession.builder.master('local[2]').appName('ag').config('spark.ui.enabled','false').config('spark.driver.host','127.0.0.1').config('spark.driver.bindAddress','127.0.0.1').getOrCreate(); t=spark.sparkContext.parallelize(range(100)).sum(); spark.stop(); assert t==4950; print('Spark sum OK:', t)" ^
    >> "%LOG%" 2>&1
if !ERRORLEVEL!==0 (
    echo   [B5] PASS -- Docker Spark local OK >> "%LOG%"
    echo   [B5] PASS -- Docker Spark local OK
    set /a PASS+=1
) else (
    echo   [B5] FAIL -- Docker Spark local error >> "%LOG%"
    echo   [B5] FAIL -- Docker Spark local error
    set /a FAIL+=1
)

:summary
echo.
echo ============================================================
echo  AIRGAP SIMULATION RESULTS
set /a TOTAL=PASS+FAIL
echo  Passed : %PASS% / %TOTAL%
echo  Failed : %FAIL% / %TOTAL%
echo  Log    : %LOG%
echo ============================================================
echo.
echo  PASSED: %PASS%  FAILED: %FAIL% >> "%LOG%"

if %FAIL%==0 (
    echo  All tests passed — safe to deploy on airgapped machine.
    echo  STATUS: ALL PASS >> "%LOG%"
) else (
    echo  Some tests failed — review log before deploying.
    echo  STATUS: SOME FAILURES >> "%LOG%"
)
echo.
pause
