@echo off
REM ============================================
REM Install GPU PyTorch on worker machines
REM (For RTX 5060 / Blackwell sm_120)
REM ============================================

echo Installing PyTorch with CUDA 12.8 support (for RTX 50-series)...
echo This requires NVIDIA driver 570+ installed.
echo.

pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir

echo.
echo Installing other dependencies...
pip install pyspark==3.5.1 psutil==5.9.8 GPUtil==1.4.0 numpy==1.26.4 pandas==2.2.1 scikit-learn==1.4.2

echo.
echo Verifying GPU...
python -c "import torch; t=torch.tensor([2.0]).cuda(); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Test: {t*t}'); print('SUCCESS')"

echo.
pause
