@echo off
REM ============================================================
REM DVD PREP - Split large files for 4.7 GB DVD burning
REM Run on INTERNET-CONNECTED machine after save_docker_images.bat
REM
REM Splits gpu-images-combined.tar.gz into 4.3 GB chunks
REM (4.3 GB leaves room for filesystem overhead on DVD)
REM
REM Output: airgap\dvd\disc1\  disc2\  disc3\
REM ============================================================
setlocal enabledelayedexpansion

set PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe
set PKG=%~dp0packages
set DVD=%~dp0dvd
set GPU_TAR=%PKG%\docker\gpu-images-combined.tar.gz
set CHUNK_SIZE=4300000000

echo.
echo ============================================================
echo  DVD PREP - Splitting packages for 4.7 GB DVDs
echo  Output: %DVD%
echo ============================================================
echo.

REM Check GPU tar exists
if not exist "%GPU_TAR%" (
    echo [ERROR] gpu-images-combined.tar.gz not found.
    echo         Run save_docker_images.bat first.
    pause & exit /b 1
)

REM Create disc folders
if not exist "%DVD%\disc1" mkdir "%DVD%\disc1"
if not exist "%DVD%\disc2" mkdir "%DVD%\disc2"
if not exist "%DVD%\disc3" mkdir "%DVD%\disc3"

REM ---- Split GPU tar into chunks ----
echo [1/4] Splitting gpu-images-combined.tar.gz into 4.3 GB chunks...
if exist "%DVD%\disc1\gpu-images-combined.tar.gz.001" (
    echo       Already split, skipping.
) else (
    "%PYTHON%" "%~dp0tests\split_file.py" "%GPU_TAR%" "%DVD%" %CHUNK_SIZE%
    if !ERRORLEVEL! NEQ 0 (echo [ERROR] Split failed. & pause & exit /b 1)
)
echo [OK] Split complete.
echo.

REM ---- Copy chunks to correct discs ----
REM disc1: chunk 001
REM disc2: chunk 002 ONLY — a 3rd+ chunk alongside it would overflow a
REM        single 4.7 GB DVD (e.g. 4.3 GB + 4.3 GB + ~1.4 GB needs 3
REM        discs, not 2 — chunk 002 alone already fills disc2)
REM disc3: chunk 003+ (if any) plus everything else — has room since it's
REM        otherwise just the much smaller CPU image + native packages + code
echo [2/4] Organizing chunks into disc folders...

REM Move chunk 001 to disc1
if exist "%DVD%\gpu-images-combined.tar.gz.001" (
    move "%DVD%\gpu-images-combined.tar.gz.001" "%DVD%\disc1\" >nul
)

REM Move chunk 002 to disc2 (only)
if exist "%DVD%\gpu-images-combined.tar.gz.002" (
    move "%DVD%\gpu-images-combined.tar.gz.002" "%DVD%\disc2\" >nul
)

REM Any remaining chunks (003+) go to disc3
for %%F in ("%DVD%\gpu-images-combined.tar.gz.00*") do (
    move "%%F" "%DVD%\disc3\" >nul 2>&1
)

echo [OK] Chunks organized.
echo.

REM ---- Copy remaining packages to disc3 ----
echo [3/4] Copying remaining packages to disc3...
if exist "%PKG%\docker\pytorch-benchmark-cpu.tar.gz" (
    copy "%PKG%\docker\pytorch-benchmark-cpu.tar.gz" "%DVD%\disc3\" >nul
    echo       pytorch-benchmark-cpu.tar.gz copied.
)
if exist "%PKG%\native\spark\spark-4.2.0-bin-hadoop3.tgz" (
    copy "%PKG%\native\spark\spark-4.2.0-bin-hadoop3.tgz" "%DVD%\disc3\spark\" >nul 2>&1
    if not exist "%DVD%\disc3\spark" mkdir "%DVD%\disc3\spark"
    copy "%PKG%\native\spark\spark-4.2.0-bin-hadoop3.tgz" "%DVD%\disc3\spark\" >nul
    echo       spark-4.2.0-bin-hadoop3.tgz copied.
)
if exist "%PKG%\native\java" (
    if not exist "%DVD%\disc3\java" mkdir "%DVD%\disc3\java"
    xcopy "%PKG%\native\java\*" "%DVD%\disc3\java\" /Q >nul
    echo       Java JRE (Windows) copied.
)
if exist "%PKG%\native\wheels" (
    if not exist "%DVD%\disc3\wheels" mkdir "%DVD%\disc3\wheels"
    xcopy "%PKG%\native\wheels\*" "%DVD%\disc3\wheels\" /Q >nul
    echo       Wheels copied.
)
if exist "%PKG%\container_toolkit" (
    if not exist "%DVD%\disc3\container_toolkit" mkdir "%DVD%\disc3\container_toolkit"
    xcopy "%PKG%\container_toolkit\*" "%DVD%\disc3\container_toolkit\" /Q >nul
    echo       NVIDIA Container Toolkit RPMs copied.
)

REM Copy project code (exclude large result files and packages)
echo       Copying project code...
if not exist "%DVD%\disc3\project" mkdir "%DVD%\disc3\project"
"%PYTHON%" "%~dp0tests\copy_project.py" "%~dp0.." "%DVD%\disc3\project"

echo [OK] Disc3 ready.
echo.

REM ---- Size report ----
echo [4/4] Disc size summary:
"%PYTHON%" -c "
import os
for disc in ['disc1','disc2','disc3']:
    path = r'%DVD%' + '\\' + disc
    if not os.path.exists(path):
        continue
    total = sum(os.path.getsize(os.path.join(dp,f)) for dp,dn,fn in os.walk(path) for f in fn)
    gb = total/1024**3
    status = 'OK' if gb < 4.5 else 'WARNING - TOO LARGE FOR DVD'
    print(f'  {disc}: {gb:.2f} GB  [{status}]')
"
echo.
echo ============================================================
echo  Done! Burn each disc\ folder to a separate 4.7 GB DVD.
echo.
echo  Burn order:
echo    DVD 1 - airgap\dvd\disc1\   (GPU tar.gz chunk 1)
echo    DVD 2 - airgap\dvd\disc2\   (GPU tar.gz chunk 2)
echo    DVD 3 - airgap\dvd\disc3\   (GPU tar.gz chunk 3+ if any, CPU image,
echo                                 native packages, Container Toolkit, code)
echo ============================================================
echo.
pause
