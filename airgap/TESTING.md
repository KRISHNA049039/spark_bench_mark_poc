# Pre-Transfer Testing Guide

You have one Windows machine right now (dev box, native + Docker both
available). Production will move to Linux later, Docker-only (see
[ARCHITECTURE.md](ARCHITECTURE.md) — there is no native Linux install path,
by design). This doc is the checklist to run **before** you spend time
packaging the airgap kit (`download_all.bat` etc.), and what changes once
a Linux box is available.

---

## 0. Hardware / driver compatibility

**Will RTX 5000-series + a driver reporting "CUDA Version 13.1" work with
the pinned `torch==2.11.0+cu128` build?** Yes. Two things to untangle:

- **"Driver version 13.1" is almost certainly the "CUDA Version" field in
  `nvidia-smi`'s header, not the driver's own build number.** NVIDIA driver
  builds are numbers like `570.65` or `580.xx` — `nvidia-smi` separately
  reports the *highest* CUDA version that driver supports. If that field
  reads `13.1`, the installed driver build is ≥580 (CUDA 13.x requires a
  ≥580 driver on both Windows and Linux).
- **CUDA drivers are backward compatible.** A driver satisfying the CUDA
  13.1 minimum (≥580) is well above the CUDA 12.8 minimum our pinned wheels
  need (**≥570.65 on Windows, ≥570.26 on Linux**) — it will run `+cu128`
  binaries with no changes. You do not need to downgrade the driver or
  re-pin torch to a "13.x" build to match it.

Run this to confirm on any machine before relying on it:
```cmd
nvidia-smi
REM Check the "CUDA Version:" field in the header — must read 12.8 or higher.

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REM Expect: 2.11.0+cu128 True <your GPU name>
```

**GPU coverage:**

| Hardware | Architecture | Compatible with pinned `+cu128` build? |
|---|---|---|
| RTX 50-series (5050 → 5090, GeForce) | Blackwell, sm_120 | ✅ Yes — this is the build line PyTorch ships specifically for Blackwell |
| RTX 5000 Ada Generation (workstation card, if that's what you mean by "RTX 5000") | Ada Lovelace, sm_89 | ✅ Yes — lower CUDA requirement than Blackwell, comfortably covered |
| RTX 40-series / 30-series (Ada / Ampere) | sm_89 / sm_86 | ✅ Yes — torch wheels bundle kernels for multiple compute capabilities, not just the newest |
| No NVIDIA GPU / driver below 570.65 (Win) / 570.26 (Linux) | — | ❌ Falls back to the CPU image/wheels (`requirements-torch-cpu.txt`) — not a failure, just no GPU phases |

If "RTX 5000" actually means the GeForce 50-series generically (the repo's
existing scripts already reference "RTX 5060 Blackwell sm_120"), the above
applies directly. If it's specifically the RTX 5000 Ada workstation card,
it's an older/lower architecture than the driver minimum quoted above, so
it's covered with more headroom, not less.

---

## 1. What one Windows box can and can't prove

| Can test now (Windows) | Can't test until you have a 2nd/Linux box |
|---|---|
| Native single-node Spark (`local[*]`) + GPU torch | Real multi-host native cluster (`start_master.bat` / `start_worker.bat` need separate machines with separate LAN IPs) |
| Docker CPU + GPU images, built and run | Docker Engine on Linux specifically (cgroups/networking differ from Docker Desktop's WSL2 backend) |
| Docker Compose multi-container Spark cluster (master + 2 workers, all as containers on this one box) — the closest single-machine proxy for the real distributed topology | NVIDIA Container Toolkit RPM install (Linux-only step, needs a RHEL box) |
| Offline/airgap simulation (`--no-index` pip imports, `--network none` Docker) | — |

The Docker Compose cluster test (`test_docker.bat`) is doing double duty:
it's your GPU worker distributed-execution test *and* your best available
stand-in for "will the driver/master split work" before you have physically
separate machines.

---

## 2. Windows test sequence

Run these in order. Each one is a gate — don't move to the next until the
current one passes.

### Phase A — Rebuild from the pinned versions
```cmd
rebuild.bat
```
Builds all 4 Docker images and refreshes the native env to
`torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` / `pyspark==4.2.0` /
Python 3.12. Use `rebuild.bat -docker-only` if you don't want to touch your
native Python env yet.

### Phase B — GPU preflight (native)
```cmd
cluster\native\check_gpu.py
```
Confirms `torch.cuda.is_available()` on the exact interpreter Spark will
launch tasks with — the #1 cause of "GPU silently falls back to CPU" is
running this against a *different* Python than the one wired into
`PYSPARK_PYTHON`.

### Phase C — Native quick test (single node)
```cmd
test_native.bat
```
Runs `pytorch_benchmark.cluster_benchmark` in `local[*]` mode: no
master/worker needed, exercises native Python 3.12 + pyspark 4.2.0 + GPU
torch together. Results land in `benchmark_results\`.

### Phase D — Docker quick tests (single container)
```cmd
docker compose up benchmark-quick
docker compose up benchmark-cpu
docker compose up benchmark-gpu
```
Or via `make run-quick` / `make run-cpu` / `make run-gpu`. Confirms each
image runs standalone before testing the multi-container cluster.

### Phase E — Docker cluster test (multi-container, one box)
```cmd
test_docker.bat
```
Starts `spark-master` + `spark-worker-1` + `spark-worker-2` as separate
containers, waits for them to register (check http://localhost:8080 —
both workers should be listed), then runs the driver
(`benchmark-cluster`). This is the strongest signal you'll get pre-Linux
that the distributed path works.

### Phase F — Airgap offline simulation (the real pre-transfer gate)
```cmd
airgap\simulate_airgap_test.bat
```
10 checks, no network access allowed during the test itself:

| # | Checks |
|---|---|
| A1 | torch import + CUDA GPU detection |
| A2 | pyspark import |
| A3 | Spark local session (`local[2]`, RDD sum) |
| A4 | GPU tensor matmul |
| A5 | `pytorch_benchmark` module importable |
| B1 | `pytorch-benchmark:cpu` image present |
| B2 | `pytorch-spark-worker:gpu` image present |
| B3 | Docker CPU container imports (`--network none`) |
| B4 | Docker GPU worker CUDA (`--network none`, `--gpus all`) |
| B5 | Docker Spark local session inside container (`--network none`) |

All 10 must pass before you run `download_all.bat` / `save_docker_images.bat`
— a failure here means the same failure happens on the real airgapped
machine, just with no internet to fix it. Log:
`benchmark_results\airgap_test\airgap_sim_<timestamp>.log`.

### Phase G — Package for transfer
Once Phase F is all-green:
```cmd
airgap\download_all.bat
airgap\save_docker_images.bat
airgap\split_for_dvd.bat
```
See [README.md](README.md) for the full transfer/burn/reassemble steps.

---

## 3. Later: testing on Linux (prod)

No native path on Linux — Docker only (see
[ARCHITECTURE.md](ARCHITECTURE.md)). Sequence:

1. **Rebuild (or `docker load` from the airgap tars) the images:**
   ```bash
   ./rebuild.sh
   # or, on the airgapped box: docker load < gpu-images-combined.tar
   ```
2. **Install the NVIDIA Container Toolkit** (GPU passthrough — Docker
   Engine on Linux doesn't get it for free the way Docker Desktop does on
   Windows):
   ```bash
   sudo bash container_toolkit/install_offline.sh
   ```
3. **Run the Docker-only airgap simulation:**
   ```bash
   ./airgap/simulate_airgap_test.sh
   ```
   Same B1-B5 checks as the Windows script's Docker half, ported to bash —
   there's no native A1-A5 section because there's nothing native to check
   on Linux. If `nvidia-smi` isn't found on the host, B4 (GPU check) skips
   instead of failing, so you can still validate the CPU path on a
   CPU-only Linux box.
4. **Cluster test** — same idea as Phase E above, just with
   `docker compose -f cluster/docker-compose.master.yml up` /
   `docker-compose.worker.yml up` on separate Linux hosts once you have
   more than one.

Everything that passed in the Windows Docker phases (D, E, F's B-checks)
is the same image running the same way — Linux is a new host, not a new
build. The native-only checks (Phase B/C, and A1/A2/A3/A5 in the airgap
sim) don't carry over; re-validate the Docker path specifically on the
Linux box rather than assuming the Windows Docker results transfer as-is.
