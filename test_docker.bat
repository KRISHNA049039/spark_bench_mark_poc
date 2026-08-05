@echo off
REM ============================================================
REM Quick Docker test — spins up spark-master + 2 workers + driver
REM Uses docker-compose.yml services already defined.
REM ============================================================

echo.
echo ============================================================
echo  DOCKER TEST
echo  Starts: spark-master, spark-worker-1, spark-worker-2
echo  Then runs: benchmark-cluster (driver)
echo ============================================================
echo.

REM Check Docker running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running. Start it and retry.
    pause & exit /b 1
)

REM Start cluster in background
echo [1/3] Starting Spark master + workers...
docker compose up -d spark-master spark-worker-1 spark-worker-2
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed to start cluster. & pause & exit /b 1)

REM Wait for workers to register
echo [2/3] Waiting 20s for workers to connect to master...
timeout /t 20 /nobreak >nul

echo       Spark UI: http://localhost:8080
echo       Check workers are listed there before proceeding.
echo.

REM Run benchmark driver
echo [3/3] Running benchmark driver...
docker compose run --rm benchmark-cluster

echo.
echo ============================================================
echo  Docker test complete.
echo  Results in: benchmark_results\
echo.
echo  Stopping cluster...
docker compose stop spark-master spark-worker-1 spark-worker-2
echo  Done.
echo ============================================================
echo.
pause
