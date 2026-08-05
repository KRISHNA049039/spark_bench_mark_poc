@echo off
REM ============================================================
REM DVD REASSEMBLE - Run on AIRGAPPED machine
REM
REM Copy all 3 DVDs to a temp folder first:
REM   D:\airgap_restore\disc1\
REM   D:\airgap_restore\disc2\
REM   D:\airgap_restore\disc3\
REM
REM Then run this script from disc3\project\airgap\
REM ============================================================
setlocal enabledelayedexpansion

REM ---- Detect Python ----
REM Prefer the `py` launcher's own -3.12 selector — it resolves the right
REM interpreter from its own runtime registry, so it works regardless of
REM which account's profile Python 3.12 actually got installed under
REM (a guessed "C:\Users\%USERNAME%\..." path can silently miss it). Any
REM Python works for the reassembly step itself (plain file I/O); the
REM install_native.bat this script calls next enforces 3.12 strictly.
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
    echo [ERROR] Python not found. Install Python first (it is on disc3).
    pause & exit /b 1
)
echo [OK] Python: %PYTHON%

REM ---- Ask for disc root ----
set /p DISC_ROOT=Enter path where you copied all 3 DVDs (e.g. D:\airgap_restore): 
if not exist "%DISC_ROOT%\disc1" (
    echo [ERROR] disc1 folder not found under: %DISC_ROOT%
    pause & exit /b 1
)

set OUT=%DISC_ROOT%\packages
if not exist "%OUT%\docker" mkdir "%OUT%\docker"
if not exist "%OUT%\native\wheels" mkdir "%OUT%\native\wheels"
if not exist "%OUT%\native\spark"  mkdir "%OUT%\native\spark"
if not exist "%OUT%\native\java"   mkdir "%OUT%\native\java"

REM ---- Reassemble GPU tar from chunks ----
echo.
echo [1/3] Reassembling gpu-images-combined.tar.gz from DVD chunks...
set GPU_TAR=%OUT%\docker\gpu-images-combined.tar.gz
if exist "%GPU_TAR%" (
    echo       Already reassembled, skipping.
) else (
    "%PYTHON%" "%~dp0tests\reassemble_file.py" "%DISC_ROOT%" "gpu-images-combined.tar.gz" "%GPU_TAR%"
    if !ERRORLEVEL! NEQ 0 (echo [ERROR] Reassembly failed. & pause & exit /b 1)
)
echo [OK] GPU tar ready: %GPU_TAR%

REM ---- Copy disc3 contents to packages ----
echo.
echo [2/3] Copying disc3 packages...
if exist "%DISC_ROOT%\disc3\pytorch-benchmark-cpu.tar" (
    copy "%DISC_ROOT%\disc3\pytorch-benchmark-cpu.tar" "%OUT%\docker\" >nul
    echo       pytorch-benchmark-cpu.tar copied.
)
if exist "%DISC_ROOT%\disc3\spark" (
    xcopy "%DISC_ROOT%\disc3\spark\*" "%OUT%\native\spark\" /Q >nul
    echo       Spark tgz copied.
)
if exist "%DISC_ROOT%\disc3\java" (
    xcopy "%DISC_ROOT%\disc3\java\*" "%OUT%\native\java\" /Q >nul
    echo       Java zip copied.
)
if exist "%DISC_ROOT%\disc3\wheels" (
    xcopy "%DISC_ROOT%\disc3\wheels\*" "%OUT%\native\wheels\" /Q >nul
    echo       Wheels copied.
)
echo [OK] Packages ready.

if exist "%DISC_ROOT%\disc3\container_toolkit" (
    echo       Copying Container Toolkit RPMs...
    if not exist "%OUT%\container_toolkit" mkdir "%OUT%\container_toolkit"
    xcopy "%DISC_ROOT%\disc3\container_toolkit\*" "%OUT%\container_toolkit\" /Q >nul
    echo [OK] Container Toolkit RPMs ready.
    echo.
    echo NOTE: To install NVIDIA Container Toolkit on RHEL, run as root:
    echo   sudo bash %OUT%\container_toolkit\install_container_toolkit.sh
    echo.
)

REM ---- Run install ----
echo.
echo [3/3] Packages assembled. Running install...
echo.
set AIRGAP_DIR=%DISC_ROOT%\disc3\project\airgap

call "%AIRGAP_DIR%\install_native.bat"
call "%AIRGAP_DIR%\load_docker_images.bat"

echo.
echo ============================================================
echo  Reassembly and install complete!
echo  Run simulate_airgap_test.bat to verify.
echo ============================================================
pause
