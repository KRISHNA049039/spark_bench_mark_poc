# Airgap Deployment Guide

Transfer and run the full benchmark suite on a machine with **no internet access**.

> **Before Step 1:** run through [TESTING.md](TESTING.md) on your dev
> machine first — it's the pre-transfer test sequence (native + Docker on
> Windows now, Docker-only on Linux later) plus a hardware/driver
> compatibility note for RTX 50-series GPUs. A failure caught there is a
> five-minute fix; the same failure caught after burning DVDs isn't.

---

## What's Included

| Script | Run On | Purpose |
|---|---|---|
| `download_all.bat` | Internet machine | Download wheels, Spark, Java, Container Toolkit repo |
| `save_docker_images.bat` | Internet machine | Build + save Docker images as .tar |
| `split_for_dvd.bat` | Internet machine | Split large files into 4.3 GB DVD chunks |
| `install_native.bat` | Airgapped machine (Windows) | Install native Python/Spark env |
| `load_docker_images.bat` | Airgapped machine | Load Docker images from .tar |
| `reassemble_dvd.bat` | Airgapped machine (Windows) | Reassemble split chunks + run install |
| `simulate_airgap_test.bat` | Windows (native + Docker) | Smoke test both environments |
| `simulate_airgap_test.sh` | Linux (Docker only) | Smoke test the Docker images |

---

## Step-by-Step

### On the INTERNET-connected machine (Windows)

**Step 1 — Download native packages + Container Toolkit repo**
```cmd
cd d:\spark_pytorch_poc
airgap\download_all.bat
```
Downloads into `airgap\packages\`:
- `native\wheels\` — Python wheels (torch 2.11.0+cu128, pyspark 4.2.0, all deps —
  pinned in `pytorch_benchmark\requirements-*.txt`, the same files the Docker
  images build from, so native and Docker are on identical versions)
- `native\spark\` — `spark-4.2.0-bin-hadoop3.tgz`
- `native\java\` — Java 17 JRE portable zip (Windows only — see note below)
- `container_toolkit\` — NVIDIA Container Toolkit repo file + libnvidia-container RPMs
  (only needed for GPU passthrough into the Docker images on a Linux target)

> **Native (non-Docker) install only works on Windows.** On a Linux/RHEL
> airgapped target, run the Docker images instead (Steps 8-9 below) — there
> is no native Linux install path in this kit.

**Step 2 — Fetch NVIDIA Container Toolkit RPMs (requires a RHEL VM or machine)**

The toolkit RPMs must be downloaded using `dnf` on a RHEL machine.
On any RHEL 8/9 machine **with internet access** (can be a VM):

```bash
# Copy fetch script from the packages folder to the RHEL machine, then:
chmod +x fetch_toolkit_rpms_on_rhel.sh
sudo bash fetch_toolkit_rpms_on_rhel.sh
```

This uses `dnf download --resolve` to pull `nvidia-container-toolkit` and all its
RPM dependencies into a `toolkit_rpms/` subfolder.

Copy the resulting `toolkit_rpms/*.rpm` files back into:
```
airgap\packages\container_toolkit\
```

**Step 3 — Build + save Docker images**
```cmd
airgap\save_docker_images.bat
```
Saves into `airgap\packages\docker\` (gzip-compressed — `docker save | gzip`):
- `gpu-images-combined.tar.gz` (both GPU images, layers deduplicated; smaller now that the GPU stages build from `python:3.12-slim` instead of a full CUDA base — check the size this script prints rather than trusting an old number)
- `pytorch-benchmark-cpu.tar.gz` (optional if target has GPU)

**Step 4 — Split for DVD burning**

If the GPU tar.gz exceeds DVD capacity (4.7 GB), split it:
```cmd
airgap\split_for_dvd.bat
```
Creates 4 disc folders under `airgap\dvd\`:

| DVD | Folder | Contents | Size |
|---|---|---|---|
| DVD 1 | `dvd\disc1\` | `gpu-images-combined.tar.gz` chunk 1 | ≤4.3 GB |
| DVD 2 | `dvd\disc2\` | `gpu-images-combined.tar.gz` chunks 2+ (if any) | ≤4.3 GB each |
| DVD 3 | `dvd\disc3\` | CPU image + native packages + project code | remainder |

> In practice disc3 holds all remaining files — burn `dvd\disc3\` to DVD 3.
> DVD count depends on how many chunks the (now smaller, compressed) GPU
> tar.gz splits into — `split_for_dvd.bat` prints the chunk count and disc
> sizes when it runs; it may now fit in 2 discs instead of 3.

**Step 5 — Burn DVDs**

Use any DVD burning software (ImgBurn, Windows built-in, Nero):
- Burn `dvd\disc1\` contents to DVD 1
- Burn `dvd\disc2\` contents to DVD 2
- Burn `dvd\disc3\` contents to DVD 3

---

### On the AIRGAPPED machine

Windows targets get both the native install and Docker images. Linux/RHEL
targets get Docker images only (Steps 8-9) — skip Step 7 there.

**Step 6 — Copy DVDs to a temp folder**

Insert each DVD and copy its contents to a local folder:
```bash
mkdir -p /tmp/airgap_restore/disc1
mkdir -p /tmp/airgap_restore/disc2
mkdir -p /tmp/airgap_restore/disc3

cp -r /dvd1/* /tmp/airgap_restore/disc1/
cp -r /dvd2/* /tmp/airgap_restore/disc2/
cp -r /dvd3/* /tmp/airgap_restore/disc3/
```

**Step 7 — Reassemble + install (Windows only)**

If the airgapped machine runs Windows, run:
```cmd
airgap\reassemble_dvd.bat
```
Enter the path where you copied the DVDs (e.g. `D:\airgap_restore`).
The script reassembles the GPU tar from chunks, copies packages into place,
then automatically runs `install_native.bat` and `load_docker_images.bat`.
Requires Python 3.12 already installed on the machine (`install_native.bat`
looks for it under the usual python.org / Windows Store install paths).

On Linux/RHEL, skip straight to Step 8 — copy `container_toolkit\` and
`docker\*.tar.gz` out of the reassembled folder and use `docker load`
(Step 9) directly; the `.bat` scripts don't run on Linux.

**Step 8 — Install NVIDIA Container Toolkit (RHEL only)**

After reassembly, install the toolkit RPMs:
```bash
sudo bash /path/to/container_toolkit/install_offline.sh
```

This script:
1. Installs all `*.rpm` files from the container_toolkit folder
2. Runs `nvidia-ctk runtime configure --runtime=docker`
3. Restarts Docker
4. Verifies with `nvidia-smi` inside a container

**Step 9 — Load Docker images**
```cmd
airgap\load_docker_images.bat
```
Loads `gpu-images-combined.tar.gz` and `pytorch-benchmark-cpu.tar.gz` into
Docker (`docker load` handles gzip-compressed tars directly). The GPU one
takes a few minutes depending on its size — check what `save_docker_images.bat`
printed on the build machine.

**Step 10 — Run simulation test**
```cmd
airgap\simulate_airgap_test.bat
```
All 10 tests should pass with no internet:

| Test | Checks |
|---|---|
| A1 | torch import + CUDA GPU detection |
| A2 | pyspark import |
| A3 | Spark local session (RDD sum) |
| A4 | GPU tensor matmul on RTX 5060 |
| A5 | pytorch_benchmark module importable |
| B1 | pytorch-benchmark:cpu image present |
| B2 | pytorch-spark-worker:gpu image present |
| B3 | Docker CPU container imports (--network none) |
| B4 | Docker GPU worker CUDA (--network none) |
| B5 | Docker Spark local session inside container |

Results saved to `benchmark_results\airgap_test\airgap_sim_<timestamp>.log`.

---

### Running the actual benchmark (airgapped)

**Native cluster:**
```cmd
REM Terminal 1 — start master
cluster\native\start_master.bat

REM Terminal 2 — start worker (if multi-node)
cluster\native\start_worker.bat

REM Terminal 3 — run benchmark
cluster\native\run_benchmark.bat
```

**Docker cluster:**
```cmd
REM Master node
docker compose -f cluster\docker-compose.master.yml up

REM Worker node(s)
docker compose -f cluster\docker-compose.worker.yml up
```

---

## Package Sizes

The GPU Docker images now build from `python:3.12-slim` instead of the full
CUDA devel/runtime base with Python compiled from source, so
`gpu-images-combined.tar.gz` is meaningfully smaller than older builds —
re-run `save_docker_images.bat` and check the printed sizes rather than
trusting a stale number here.

| Package | Notes |
|---|---|
| `gpu-images-combined.tar.gz` | Both GPU images; no CUDA base layer anymore — torch's cu128 wheel bundles its own CUDA/cuDNN runtime |
| `pytorch-benchmark-cpu.tar.gz` | Optional — skip if target has GPU |
| Python wheels (all deps) | torch 2.11.0+cu128 + pyspark 4.2.0 + deps, pinned in `pytorch_benchmark\requirements-*.txt` |
| `spark-4.2.0-bin-hadoop3.tgz` | |
| Java 17 JRE zip (Windows) | `native\java\OpenJDK17U-jre_x64_windows_hotspot_17.0.11_9.zip` |
| NVIDIA Container Toolkit RPMs | ~8 MB total — Linux Docker GPU passthrough only |
| NVIDIA driver installer | Download separately from nvidia.com |
| Python 3.12 installer | Download separately from python.org (Windows native install only) |
| Project code | |

---

## Requirements on the Airgapped Machine

### For native (Windows only — Linux has no native install path):
- Python 3.12 installed
- NVIDIA driver (Game Ready or Studio)
- Java not required — portable JRE is included in packages

### For Docker on Windows:
- Docker Desktop with WSL2 backend enabled
- NVIDIA driver (enables GPU passthrough — no Container Toolkit needed on Windows)

### For Docker on RHEL:
- Docker Engine installed
- NVIDIA driver installed
- NVIDIA Container Toolkit (installed via `install_offline.sh` in Step 8)

---

## Troubleshooting

**`pip install --no-index` fails with "no matching distribution"**
— Re-run `download_all.bat`; a wheel may have been missed.

**Docker image load hangs**
— Large tars (~10 GB) take 5-10 min. Wait for the `[OK]` line.

**Spark executor "system cannot find path"**
— Make sure `install_native.bat` completed successfully (Java + Spark extracted).

**GPU not detected in Docker on Windows**
— Verify `nvidia-smi` works natively first. Check Docker Desktop > Settings > Resources > GPU is enabled.

**GPU not detected in Docker on RHEL**
— Verify Container Toolkit is installed: `nvidia-ctk --version`
— Verify Docker runtime config: `docker info | grep -i runtime`
— Should show `nvidia` as a runtime. If not, re-run: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`

**`fetch_toolkit_rpms_on_rhel.sh` — dnf download fails**
— Make sure the NVIDIA repo was added: `cat /etc/yum.repos.d/nvidia-container-toolkit.repo`
— Try manually: `sudo dnf install -y nvidia-container-toolkit` on a connected machine and note all downloaded RPMs.

**Chunk reassembly fails — missing chunk**
— Make sure all DVDs were fully copied. Check chunk files exist:
  `ls /tmp/airgap_restore/disc*/gpu-images-combined.tar.*`
