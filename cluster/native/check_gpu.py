"""
GPU preflight check — run this with the EXACT python you're about to point
PYSPARK_PYTHON at, on EVERY machine (master + each worker), BEFORE starting
Spark. This is the same check torch.cuda.is_available() does inside a Spark
task, so if it fails here it will silently fall back to CPU inside Spark too.

Usage (on each node):
    cluster\\native\\spark\\...\\python.exe cluster\\native\\check_gpu.py
    # or, matching whatever you'll set PYSPARK_PYTHON to:
    <path-to-python.exe> cluster\\native\\check_gpu.py
"""
import sys
import platform


def main():
    print("=" * 60)
    print("GPU Preflight Check")
    print("=" * 60)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")

    try:
        import torch
    except ImportError as e:
        print(f"[FAIL] Could not import torch: {e}")
        print("       This interpreter has no PyTorch installed at all.")
        sys.exit(1)

    print(f"Torch version     : {torch.__version__}")
    print(f"Torch CUDA build  : {torch.version.cuda}")

    is_cpu_only_build = torch.version.cuda is None
    if is_cpu_only_build:
        print("[FAIL] This is a CPU-ONLY build of PyTorch (torch.version.cuda is None).")
        print("       Reinstall GPU PyTorch INTO THIS EXACT INTERPRETER, e.g.:")
        print(f'       "{sys.executable}" -m pip install --pre torch torchvision '
              "--index-url https://download.pytorch.org/whl/nightly/cu128")
        sys.exit(1)

    try:
        available = torch.cuda.is_available()
    except Exception as e:
        print(f"[FAIL] torch.cuda.is_available() raised: {e!r}")
        sys.exit(1)

    if not available:
        print("[FAIL] torch.cuda.is_available() == False")
        print("       Possible causes: NVIDIA driver missing/outdated,")
        print("       GPU not visible to this user/session, or a CUDA_VISIBLE_DEVICES")
        print("       env var hiding the device.")
        sys.exit(1)

    count = torch.cuda.device_count()
    print(f"[OK] CUDA available. Device count: {count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"     cuda:{i} -> {name} (compute capability {cap[0]}.{cap[1]})")

    try:
        t = torch.tensor([2.0], device="cuda:0")
        result = (t * t).item()
        print(f"[OK] Tensor op on cuda:0 succeeded: 2*2 = {result}")
    except Exception as e:
        print(f"[FAIL] Tensor op on cuda:0 raised: {e!r}")
        print("       Driver/CUDA-build mismatch is the most common cause")
        print("       (e.g. sm_120 Blackwell GPU needs a matching nightly cu128 build).")
        sys.exit(1)

    print("=" * 60)
    print("PASS: this interpreter is ready for Spark GPU tasks.")
    print("Set PYSPARK_PYTHON to this exact path before starting the worker:")
    print(f"  set PYSPARK_PYTHON={sys.executable}")
    print("=" * 60)


if __name__ == "__main__":
    main()
