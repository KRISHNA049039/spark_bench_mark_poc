@echo off
REM ============================================================
REM AIRGAP STEP 2 — Run on AIRGAPPED machine (native install)
REM
REM Installs from airgap\packages\ (no internet needed):
REM   - Extracts portable Java 17 JRE
REM   - Extracts Spark 4.2.0
REM   - Installs Python wheels offline
REM   - Patches run_benchmark.bat + start_*.bat to use local paths
REM
REM Requirements:
REM   - Python 3.12 already installed on the airgapped machine
REM     (grab the Python installer offline: https://python.org/downloads)
REM   - This script run from inside the  spark_pytorch_poc\  folder
REM ============================================================
setlocal enabledelayedexpansion

set ROOT=%~dp0..
set PKG=%~dp0packages
set WHEELS=%PKG%\native\wheels
set SPARK_TGZ=%PKG%\native\spark\spark-4.2.0-bin-hadoop3.tgz
set JAVA_ZIP=%PKG%\native\java\OpenJDK17U-jre_x64_windows_hotspot_17.0.11_9.zip

set SPARK_DEST=%ROOT%\cluster\native\spark
set JAVA_DEST=%ROOT%\cluster\native\java

REM ---- Detect Python ----
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
    echo [ERROR] Python not found. Install Python 3.12 first.
    pause & exit /b 1
)

REM The wheels in packages\native\wheels are cp312-only — a different
REM interpreter version will fail here with a clearer message than pip's
REM "no matching distribution" further down.
"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] %PYTHON% is not Python 3.12 — the packaged wheels are cp312-only.
    echo         Install Python 3.12 ^(py install 3.12, or python.org^) and re-run.
    pause & exit /b 1
)
echo [OK] Python: %PYTHON%

REM ---- Validate packages exist ----
echo.
echo Checking packages...
if not exist "%WHEELS%" (
    echo [ERROR] Wheels not found: %WHEELS%
    echo         Run download_all.bat on an internet-connected machine first.
    pause & exit /b 1
)
if not exist "%SPARK_TGZ%" (
    echo [ERROR] Spark tgz not found: %SPARK_TGZ%
    pause & exit /b 1
)
if not exist "%JAVA_ZIP%" (
    echo [ERROR] Java zip not found: %JAVA_ZIP%
    pause & exit /b 1
)
echo [OK] All packages present.
echo.

REM ---- 1. Extract Java ----
echo [1/3] Extracting Java 17 JRE...
if exist "%JAVA_DEST%\bin\java.exe" (
    echo       Already extracted, skipping.
) else (
    if not exist "%JAVA_DEST%" mkdir "%JAVA_DEST%"
    "%PYTHON%" -c "
import zipfile, os
with zipfile.ZipFile(r'%JAVA_ZIP%') as z:
    z.extractall(r'%JAVA_DEST%')
# Move contents up one level (zip has a top-level folder)
import shutil
subdirs = [d for d in os.listdir(r'%JAVA_DEST%') if os.path.isdir(os.path.join(r'%JAVA_DEST%', d))]
if len(subdirs)==1:
    src = os.path.join(r'%JAVA_DEST%', subdirs[0])
    for item in os.listdir(src):
        shutil.move(os.path.join(src, item), r'%JAVA_DEST%')
    os.rmdir(src)
print('Done.')
"
    if !ERRORLEVEL! NEQ 0 (echo [ERROR] Java extraction failed. & pause & exit /b 1)
)
echo [OK] Java at: %JAVA_DEST%

REM ---- 2. Extract Spark ----
echo.
echo [2/3] Extracting Spark 4.2.0...
if exist "%SPARK_DEST%\spark-4.2.0-bin-hadoop3\bin\spark-submit.cmd" (
    echo       Already extracted, skipping.
) else (
    if not exist "%SPARK_DEST%" mkdir "%SPARK_DEST%"
    "%PYTHON%" -c "
import tarfile, os
print('  Extracting (may take ~1 min)...')
with tarfile.open(r'%SPARK_TGZ%', 'r:gz') as t:
    t.extractall(r'%SPARK_DEST%')
print('Done.')
"
    if !ERRORLEVEL! NEQ 0 (echo [ERROR] Spark extraction failed. & pause & exit /b 1)
)
echo [OK] Spark at: %SPARK_DEST%\spark-4.2.0-bin-hadoop3

REM ---- 3. Install Python wheels ----
echo.
echo [3/3] Installing Python wheels (offline)...
"%PYTHON%" -m pip install ^
    --no-index ^
    --find-links "%WHEELS%" ^
    -r "%ROOT%\pytorch_benchmark\requirements-torch-gpu.txt" ^
    -r "%ROOT%\pytorch_benchmark\requirements-base.txt"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Wheel install failed.
    pause & exit /b 1
)
echo [OK] Wheels installed.

REM ---- 4. Patch batch files to use local Java path ----
echo.
echo [4/4] Patching batch files with local Java path...
set JAVA_HOME_LOCAL=%JAVA_DEST%

REM Rewrite JAVA_HOME in run_benchmark.bat
"%PYTHON%" -c "
import re, pathlib
files = [
    r'%ROOT%\cluster\native\run_benchmark.bat',
    r'%ROOT%\cluster\native\start_master.bat',
    r'%ROOT%\cluster\native\start_worker.bat',
]
new_java = r'%JAVA_HOME_LOCAL%'
for f in files:
    p = pathlib.Path(f)
    if not p.exists():
        continue
    txt = p.read_text()
    # Replace any existing JAVA_HOME set line
    txt = re.sub(r'set JAVA_HOME=.*', f'set JAVA_HOME={new_java}', txt)
    p.write_text(txt)
    print(f'  Patched: {f}')
print('Done.')
"

echo.
echo ============================================================
echo  Native install complete!
echo.
echo  Run the benchmark:
echo    cd %ROOT%
echo    cluster\native\run_benchmark.bat
echo ============================================================
echo.
pause
