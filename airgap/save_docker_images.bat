@echo off
REM ============================================================
REM AIRGAP STEP 3 - Run on INTERNET-CONNECTED machine
REM
REM Builds all Docker images and saves them as compressed .tar.gz files.
REM Compression typically reduces size by 30-45% vs raw .tar
REM
REM Output folder: airgap\packages\docker\
REM ============================================================
setlocal enabledelayedexpansion

set ROOT=%~dp0..
set DOCKER_PKG=%~dp0packages\docker
set PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe

if not exist "%DOCKER_PKG%" mkdir "%DOCKER_PKG%"

echo.
echo ============================================================
echo  DOCKER IMAGE SAVER (with gzip compression)
echo  Output: %DOCKER_PKG%
echo ============================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running. Start it and retry.
    pause & exit /b 1
)

REM ---- Find gzip ----
REM Plain cmd.exe/PowerShell PATH usually doesn't include it even when
REM installed — Git for Windows bundles a copy that's the most common
REM source on a dev machine, so check its usual fixed location too before
REM giving up.
set GZIP=
where gzip >nul 2>&1
if !ERRORLEVEL!==0 set GZIP=gzip
if not defined GZIP if exist "C:\Program Files\Git\usr\bin\gzip.exe" set GZIP=C:\Program Files\Git\usr\bin\gzip.exe
if not defined GZIP if exist "%ProgramFiles%\Git\usr\bin\gzip.exe" set GZIP=%ProgramFiles%\Git\usr\bin\gzip.exe
if not defined GZIP (
    echo [ERROR] gzip not found on PATH or at Git for Windows' usual location.
    echo         Install Git for Windows (https://git-scm.com/download/win^) or
    echo         add a gzip.exe to PATH, then retry.
    pause & exit /b 1
)

REM ---- Build all images ----

echo [BUILD] pytorch-spark-worker:gpu
docker build --file "%ROOT%\Dockerfile.worker" --target gpu --tag pytorch-spark-worker:gpu "%ROOT%"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed & pause & exit /b 1)

echo [BUILD] pytorch-benchmark:gpu
docker build --file "%ROOT%\Dockerfile" --target gpu --tag pytorch-benchmark:gpu "%ROOT%"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed & pause & exit /b 1)

echo [BUILD] pytorch-benchmark:cpu
docker build --file "%ROOT%\Dockerfile" --target cpu --tag pytorch-benchmark:cpu "%ROOT%"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed & pause & exit /b 1)

echo.

REM ---- Save + compress GPU images in ONE tar.gz ----
REM Shared CUDA base written once + gzip compression = smallest possible file

set GPU_TGZ=%DOCKER_PKG%\gpu-images-combined.tar.gz
if exist "%GPU_TGZ%" (
    echo [SKIP] gpu-images-combined.tar.gz already exists. Delete it to re-export.
) else (
    echo [SAVE] Saving + compressing GPU images...
    echo        docker save pipes directly into gzip - no temp file needed
    docker save pytorch-spark-worker:gpu pytorch-benchmark:gpu | "%GZIP%" > "%GPU_TGZ%"
    if !ERRORLEVEL! NEQ 0 (echo [ERROR] docker save/compress failed & pause & exit /b 1)
    echo [OK] Saved: gpu-images-combined.tar.gz
)

REM ---- Save + compress CPU image ----

set CPU_TGZ=%DOCKER_PKG%\pytorch-benchmark-cpu.tar.gz
if exist "%CPU_TGZ%" (
    echo [SKIP] pytorch-benchmark-cpu.tar.gz already exists.
) else (
    echo [SAVE] Saving + compressing CPU image...
    docker save pytorch-benchmark:cpu | "%GZIP%" > "%CPU_TGZ%"
    if !ERRORLEVEL! NEQ 0 (echo [ERROR] docker save/compress failed & pause & exit /b 1)
    echo [OK] Saved: pytorch-benchmark-cpu.tar.gz
)

echo.
echo ============================================================
echo  Compression complete. File sizes:
echo.
"%PYTHON%" -c "
import os
pkg = r'%DOCKER_PKG%'
for f in ['gpu-images-combined.tar.gz', 'pytorch-benchmark-cpu.tar.gz']:
    p = os.path.join(pkg, f)
    if os.path.exists(p):
        gb = os.path.getsize(p) / 1024**3
        print(f'  {f:<35} {gb:.2f} GB')
"
echo.
echo  Load on airgapped machine with:
echo    airgap\load_docker_images.bat
echo ============================================================
echo.
pause
