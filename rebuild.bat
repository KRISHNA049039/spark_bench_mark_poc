@echo off
REM ============================================================
REM ONE-COMMAND REBUILD — Windows
REM
REM Rebuilds all 4 Docker images (pytorch-benchmark + pytorch-spark-worker,
REM CPU + GPU) and, unless -docker-only is passed, refreshes the native
REM Windows Python environment to the same pinned versions.
REM
REM Uses normal `docker build` (no --no-cache) on purpose: Dockerfile and
REM Dockerfile.worker isolate the apt/Java layers from the pip layers, so
REM bumping a library version in pytorch_benchmark\requirements-*.txt only
REM invalidates that library's layer — Docker's build cache does the
REM "just rebuild what changed" work for you.
REM
REM Usage:
REM   rebuild.bat              Rebuild Docker images + native env
REM   rebuild.bat -docker-only Rebuild Docker images only
REM   rebuild.bat -native-only Refresh native env only (skips Docker/docker info entirely)
REM ============================================================
setlocal enabledelayedexpansion

set ROOT=%~dp0
set DOCKER_ONLY=0
set NATIVE_ONLY=0
if /I "%~1"=="-docker-only" set DOCKER_ONLY=1
if /I "%~1"=="-native-only" set NATIVE_ONLY=1

echo.
echo ============================================================
echo  REBUILD — pytorch-benchmark suite
echo ============================================================
echo.

if %NATIVE_ONLY%==1 goto :native

docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running. Start it and retry.
    pause & exit /b 1
)

echo [1/4] Building pytorch-benchmark:cpu ...
docker build --target cpu -t pytorch-benchmark:cpu "%ROOT%."
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed. & pause & exit /b 1)

echo.
echo [2/4] Building pytorch-benchmark:gpu ...
docker build --target gpu -t pytorch-benchmark:gpu "%ROOT%."
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed. & pause & exit /b 1)

echo.
echo [3/4] Building pytorch-spark-worker:cpu ...
docker build --file "%ROOT%Dockerfile.worker" --target cpu -t pytorch-spark-worker:cpu "%ROOT%."
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed. & pause & exit /b 1)

echo.
echo [4/4] Building pytorch-spark-worker:gpu ...
docker build --file "%ROOT%Dockerfile.worker" --target gpu -t pytorch-spark-worker:gpu "%ROOT%."
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Build failed. & pause & exit /b 1)

echo.
echo [OK] All 4 Docker images built.

if %DOCKER_ONLY%==1 goto :done

:native
echo.
echo ============================================================
echo  Refreshing native Windows Python environment
echo ============================================================
REM Prefer the `py` launcher's own -3.12 selector — it resolves the right
REM interpreter from its own runtime registry, so it works regardless of
REM which account's profile Python 3.12 actually got installed under
REM (a guessed "C:\Users\%USERNAME%\..." path can silently miss it).
set PYTHON=
where py >nul 2>&1
if !ERRORLEVEL!==0 (
    for /f "delims=" %%i in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set PYTHON=%%i
)
if not defined PYTHON (
    for %%P in (
        "C:\Users\%USERNAME%\AppData\Local\Python\pythoncore-3.12-64\python.exe"
        "C:\Python312\python.exe"
    ) do (
        if not defined PYTHON if exist %%P set PYTHON=%%~P
    )
)
if not defined PYTHON (
    where python >nul 2>&1
    if !ERRORLEVEL!==0 set PYTHON=python
)
if not defined PYTHON (
    echo [SKIP] No Python found at all — install Python 3.12 first, or run with -docker-only.
    goto :done
)

REM A "python" found on PATH is not necessarily 3.12 (e.g. the Windows
REM Store alias often resolves to whatever's newest). Our pinned versions
REM (numpy==1.26.4 etc.) have no prebuilt wheel for other interpreters, so
REM pip falls back to building from source and fails without a compiler —
REM verify the version before touching anything.
"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [SKIP] %PYTHON% is not Python 3.12 — install Python 3.12 from python.org
    echo        and re-run, or run with -docker-only to skip the native refresh.
    goto :done
)
echo Using: %PYTHON%

REM Detect via nvidia-smi (provided by the NVIDIA driver, independent of
REM whatever Python packages happen to be installed already) rather than
REM `import torch; torch.cuda.is_available()` — that check requires torch
REM to already be installed to answer, so on a brand-new interpreter with
REM nothing installed yet it always says "no GPU" and silently installs
REM the CPU build even on a real GPU machine.
echo Detecting GPU install target (checking for nvidia-smi)...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    nvidia-smi >nul 2>&1
    if !ERRORLEVEL!==0 (set GPU_MACHINE=1) else (set GPU_MACHINE=0)
) else (
    set GPU_MACHINE=0
)

if "%GPU_MACHINE%"=="1" (
    echo GPU detected — reinstalling pinned CUDA 12.8 torch build.
    "%PYTHON%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 -r "%ROOT%pytorch_benchmark\requirements-torch-gpu.txt"
) else (
    echo No GPU torch detected — installing CPU build. Run cluster\native\install_gpu_worker.bat on GPU machines instead.
    "%PYTHON%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r "%ROOT%pytorch_benchmark\requirements-torch-cpu.txt"
)
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] torch install failed — native environment NOT refreshed.
    pause & exit /b 1
)

"%PYTHON%" -m pip install --no-cache-dir -r "%ROOT%pytorch_benchmark\requirements-base.txt"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] base dependency install failed — native environment NOT fully refreshed.
    pause & exit /b 1
)

echo [OK] Native environment refreshed.

:done
echo.
echo ============================================================
echo  Rebuild complete.
echo ============================================================
echo.
pause
