@echo off
REM ============================================================
REM AIRGAP STEP 4 — Run on AIRGAPPED machine
REM
REM Loads all Docker image tars into local Docker Desktop.
REM No internet connection required.
REM ============================================================
setlocal enabledelayedexpansion

set DOCKER_PKG=%~dp0packages\docker

echo.
echo ============================================================
echo  DOCKER IMAGE LOADER
echo  Source: %DOCKER_PKG%
echo ============================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running. Start it and retry.
    pause & exit /b 1
)

if not exist "%DOCKER_PKG%" (
    echo [ERROR] Docker packages folder not found: %DOCKER_PKG%
    echo         Run save_docker_images.bat on an internet-connected machine first.
    pause & exit /b 1
)

REM ---- Load .tar files ----
set COUNT=0
set FAILED=0

for %%F in ("%DOCKER_PKG%\*.tar") do (
    echo [LOAD] %%~nxF ...
    docker load -i "%%F"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to load %%~nxF
        set /a FAILED+=1
    ) else (
        echo [OK]   %%~nxF loaded.
        set /a COUNT+=1
    )
    echo.
)

REM ---- Load .tar.gz files (compressed) ----
for %%F in ("%DOCKER_PKG%\*.tar.gz") do (
    echo [LOAD] %%~nxF (compressed)...
    docker load -i "%%F"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to load %%~nxF
        set /a FAILED+=1
    ) else (
        echo [OK]   %%~nxF loaded.
        set /a COUNT+=1
    )
    echo.
)
REM ---- Summary ----
echo ============================================================
echo  Loaded: %COUNT% image(s)
if %FAILED% GTR 0 (
    echo  FAILED: %FAILED% image(s^) — check errors above
) else (
    echo  All images loaded successfully.
)
echo.
echo  Verify:
echo    docker images
echo.
echo  Test Docker environment (no internet):
echo    airgap\simulate_airgap_test.bat
echo ============================================================
echo.
pause
