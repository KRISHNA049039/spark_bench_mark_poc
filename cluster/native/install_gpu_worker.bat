@echo off
REM ============================================
REM Install GPU PyTorch on worker machines
REM (For RTX 5060 / Blackwell sm_120)
REM
REM Installs the exact same pinned versions as the GPU Docker images and
REM the airgap wheel bundle (pytorch_benchmark/requirements-*.txt), so
REM native and Docker benchmark numbers stay comparable.
REM ============================================

set ROOT=%~dp0..\..

echo Installing PyTorch 2.11.0 with CUDA 12.8 support (for RTX 50-series)...
echo This requires NVIDIA driver 570+ installed.
echo.

pip install --no-cache-dir ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    -r "%ROOT%\pytorch_benchmark\requirements-torch-gpu.txt"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Torch GPU install failed. & pause & exit /b 1)

echo.
echo Installing other dependencies...
pip install --no-cache-dir -r "%ROOT%\pytorch_benchmark\requirements-base.txt"
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Dependency install failed. & pause & exit /b 1)

echo.
echo Verifying GPU...
python -c "import torch; t=torch.tensor([2.0]).cuda(); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Test: {t*t}'); print('SUCCESS')"

echo.
pause
