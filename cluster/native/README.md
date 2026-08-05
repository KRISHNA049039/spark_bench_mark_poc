# Native Multi-Node Cluster Setup (No Docker)

This runs Spark + PyTorch natively on each Windows machine, avoiding Docker Desktop's networking issues.
This doc covers the multi-node Windows topology specifically; for plain
single-machine package install steps (Windows *and* Linux), see
[DOWNLOAD.md](DOWNLOAD.md).

## Prerequisites (install on ALL machines)

1. **Python 3.12** (pinned to match `Dockerfile`/`Dockerfile.worker`) — https://www.python.org/downloads/
2. **Java 17** — https://adoptium.net/temurin/releases/
3. **Set JAVA_HOME** — System Environment Variables → `JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17...`

## Machine Layout

| Machine | IP | Role |
|---------|-----|------|
| Node 1 | 192.168.4.100 | Master + Driver |
| Node 2 | 192.168.4.101 | Worker 1 |
| Node 3 | 192.168.4.102 | Worker 2 |

## Quick Start

### Step 1: Install on ALL machines

On the driver/master (usually no GPU needed):
```cmd
cd spark_bench_mark_poc
pip install -r cluster/native/requirements-native.txt
```

On EVERY machine that has a GPU (including a GPU driver node, if it's also
a worker), install GPU PyTorch **into that same interpreter** instead:
```cmd
cluster\native\install_gpu_worker.bat
```
Skipping this step — or running it against a different Python install than
the one Spark will actually launch — is the #1 reason GPU nodes silently
fall back to CPU with no error. requirements-native.txt alone installs
CPU-only torch.

### Step 2: Download Spark on ALL machines
```cmd
cd cluster/native
python download_spark.py
```

### Step 3: Verify GPU visibility on EVERY GPU machine

Run this with the exact `python.exe` you intend to use for Spark (see
`where python`, or your venv's path):
```cmd
<path-to-python.exe> cluster\native\check_gpu.py
```
It must print `PASS` and list your GPU. If it fails, fix that before
starting the worker — starting Spark won't surface the problem, it'll
just quietly run tasks on CPU.

Then open `start_worker.bat` (and `run_benchmark.bat` on the driver) and
set `PYSPARK_PYTHON` to that exact path. This pins which interpreter Spark
uses to run tasks — without it, Spark falls back to whatever `python` is
first on PATH, which may not be the one with GPU torch installed.

### Step 4: Start Master (Node 1 only)
```cmd
cluster\native\start_master.bat
```

### Step 5: Start Workers (Node 2 and Node 3)
```cmd
cluster\native\start_worker.bat
```
Each worker now runs the GPU preflight check automatically on startup and
warns you if it fails on that node.

### Step 6: Run Benchmark (Node 1)
```cmd
python -m pytorch_benchmark.cluster_benchmark
```

Check Spark UI: http://192.168.4.100:8080 — click into an executor and
confirm it's on the LAN IP of the worker machine you expect, not
`127.0.0.1` or a stale IP.

### Troubleshooting: GPU works locally but not on other nodes

This means each node's Spark *task* process (not just your interactive
shell) can't see CUDA. Checklist, per remote node:

1. Run `check_gpu.py` with the SAME python Spark will use (Step 3) — not
   just any `python` in a terminal. Multiple Python installs on Windows is
   the most common cause of this mismatch.
2. Confirm `PYSPARK_PYTHON` is set in that node's `start_worker.bat` before
   the worker starts (echoed at the top of the worker's console output).
3. Check the NVIDIA driver version on that node (`nvidia-smi`) — a
   too-old driver silently fails `torch.cuda.is_available()`.
4. Look at the per-partition `cuda_diagnostic` field in the benchmark's
   JSON output (or console log during a GPU phase) — it now reports the
   hostname, python path, torch build, and exact failure reason for every
   task, so you don't have to guess which node/interpreter is at fault.
