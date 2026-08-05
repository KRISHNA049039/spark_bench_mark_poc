"""
GPU preflight check for the driver node.
Exits with code 1 if GPU/CUDA is not available.
"""
import sys

try:
    import torch
except ImportError:
    print("[ERROR] torch not installed in this Python environment.")
    print(f"        Python: {sys.executable}")
    sys.exit(1)

print(f"  Python     : {sys.executable}")
print(f"  PyTorch    : {torch.__version__}")
print(f"  CUDA built : {torch.version.cuda}")
print(f"  CUDA avail : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {props.name} ({props.total_memory // 1024**2} MB)")
    print("\n[OK] GPU ready on driver node.\n")
else:
    print("\n[WARNING] No CUDA GPU detected on driver node.")
    print("          Phases 2/3 (GPU) will be skipped or run on CPU.\n")
    # Not a hard failure — benchmark can still run CPU phases
