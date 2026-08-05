import torch
if not torch.cuda.is_available():
    print("      No GPU - skipping")
    exit(0)

a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = a @ b
torch.cuda.synchronize()
print(f"      matmul 512x512 on {torch.cuda.get_device_name(0)} OK")
