@echo off
REM ============================================================
REM AIRGAP STEP 1 - Run on INTERNET-CONNECTED machine
REM
REM Downloads everything needed for the NATIVE (Windows) environment:
REM   - Python wheels (torch cu128 2.11.0, pyspark 4.2.0, all deps —
REM     pinned in pytorch_benchmark/requirements-*.txt, the same files
REM     the Docker images build from)
REM   - Spark 4.2.0 tgz
REM   - Java 17 JRE portable zip (Windows)
REM   - NVIDIA Container Toolkit RPMs (for Docker GPU passthrough on
REM     an airgapped Linux/RHEL target — Linux itself only runs the
REM     Docker images, there is no native Linux install path)
REM
REM Output: airgap\packages\
REM   native\wheels\   - pip wheels
REM   native\spark\    - spark-4.2.0-bin-hadoop3.tgz
REM   native\java\     - OpenJDK17 portable zip (Windows only)
REM ============================================================
setlocal enabledelayedexpansion

REM ---- Detect Python 3.12 (same approach as install_native.bat) ----
REM Tries, in order: `py` launcher's -3.12 selector (python.org/Store
REM installs), `uv python find` (uv-managed installs, tagged under a
REM different "company" so the py launcher's -3.12 shorthand can't see
REM them), a couple of common fixed paths, then bare `python` on PATH.
set PYTHON=
where py >nul 2>&1
if !ERRORLEVEL!==0 (
    for /f "delims=" %%i in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set PYTHON=%%i
)
if not defined PYTHON (
    where uv >nul 2>&1
    if !ERRORLEVEL!==0 (
        for /f "delims=" %%i in ('uv python find 3.12 2^>nul') do set PYTHON=%%i
    )
)
if not defined PYTHON (
    for %%P in (
        "C:\Users\%USERNAME%\AppData\Local\Python\pythoncore-3.12-64\python.exe"
        "C:\Python312\python.exe"
    ) do (
        if not defined PYTHON (
            if exist %%P set PYTHON=%%~P
        )
    )
)
if not defined PYTHON (
    where python >nul 2>&1
    if !ERRORLEVEL!==0 set PYTHON=python
)
if not defined PYTHON (
    echo [ERROR] Python 3.12 not found. Install it, or if using uv: uv python install 3.12
    pause & exit /b 1
)
echo [OK] Python: %PYTHON%

set ROOT=%~dp0..
set TESTS=%~dp0tests
set PKG=%~dp0packages
set WHEELS=%PKG%\native\wheels
set SPARKS=%PKG%\native\spark
set JAVAS=%PKG%\native\java
set SPARK_TGZ=%SPARKS%\spark-4.2.0-bin-hadoop3.tgz
set JAVA_ZIP=%JAVAS%\OpenJDK17U-jre_x64_windows_hotspot_17.0.11_9.zip

echo.
echo ============================================================
echo  AIRGAP DOWNLOADER - Native + Docker
echo  Output: %PKG%
echo ============================================================
echo.

REM ---- Create directories ----
if not exist "%WHEELS%" mkdir "%WHEELS%"
if not exist "%SPARKS%" mkdir "%SPARKS%"
if not exist "%JAVAS%"  mkdir "%JAVAS%"

REM ---- 1. Python wheels (pinned — same versions as the GPU Docker images) ----
echo [1/4] Downloading Python wheels...
echo       torch 2.11.0+cu128, pyspark 4.2.0, and all benchmark deps
echo.

"%PYTHON%" -m pip download ^
    --dest "%WHEELS%" ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    -r "%ROOT%\pytorch_benchmark\requirements-torch-gpu.txt"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] pip download torch wheels failed. & pause & exit /b 1)

"%PYTHON%" -m pip download ^
    --dest "%WHEELS%" ^
    -r "%ROOT%\pytorch_benchmark\requirements-base.txt"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] pip download base deps failed. & pause & exit /b 1)

echo.
echo [OK] Wheels saved to: %WHEELS%
echo.

REM ---- 2. Spark 4.2.0 ----
echo [2/4] Downloading Spark 4.2.0...
if exist "%SPARK_TGZ%" (
    echo       Already exists, skipping.
) else (
    "%PYTHON%" "%TESTS%\download_spark.py" "%SPARK_TGZ%"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Spark download failed.
        if exist "%SPARK_TGZ%" del "%SPARK_TGZ%"
        pause & exit /b 1
    )
)
echo [OK] Spark saved to: %SPARK_TGZ%
echo.

REM ---- 3. Java 17 JRE (Windows) ----
echo [3/4] Downloading Java 17 JRE (Windows x64)...
if exist "%JAVA_ZIP%" (
    echo       Already exists, skipping.
) else (
    "%PYTHON%" "%TESTS%\download_java.py" "%JAVA_ZIP%"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Java Windows download failed.
        if exist "%JAVA_ZIP%" del "%JAVA_ZIP%"
        pause & exit /b 1
    )
)
echo [OK] Java (Windows) saved to: %JAVA_ZIP%
echo.

REM ---- 4. NVIDIA Container Toolkit RPMs (RHEL 8/9 x86_64) ----
REM Only needed for GPU passthrough into the Docker images on a Linux
REM airgapped target — there is no native (non-Docker) Linux install path.
echo [4/4] Downloading NVIDIA Container Toolkit RPMs for RHEL...
set CTK_DIR=%PKG%\container_toolkit
if not exist "%CTK_DIR%" mkdir "%CTK_DIR%"
"%PYTHON%" "%TESTS%\download_container_toolkit.py" "%CTK_DIR%"
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Container Toolkit download failed.
    pause & exit /b 1
)
echo [OK] Container Toolkit RPMs saved to: %CTK_DIR%
echo.

REM ---- Summary ----
echo ============================================================
echo  Download complete!
echo.
echo  Packages:
echo    native\wheels\        Python wheels (torch 2.11.0+cu128 + all deps)
echo    native\spark\         Spark 4.2.0 tgz
echo    native\java\          Java 17 JRE zip (Windows)
echo    container_toolkit\    NVIDIA Container Toolkit RPMs (Linux Docker GPU only)
echo    docker\               Docker image tars (run save_docker_images.bat)
echo.
echo  Next steps:
echo    1. Run  airgap\save_docker_images.bat  (if not done)
echo    2. Run  airgap\split_for_dvd.bat       (split for DVD burning)
echo    3. Burn disc1\ disc2\ disc3\ to DVDs
echo    4. On the airgapped machine: run reassemble_dvd.bat
echo       (Windows: native install + Docker. Linux: Docker only.)
echo ============================================================
echo.
pause
