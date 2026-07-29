# Airgapped Deployment Guide
## Spark + PyTorch Distributed Benchmark (spark_bench_mark_poc)

Based on your `CLUSTER_SETUP.md`: Master runs **natively** (Python 3.14 + PySpark 4.2.0 + Java 17), Workers run in **Docker** (Python 3.14 + PySpark 4.2.0 + PyTorch + Spark 4.2.0). This guide covers getting the same setup working with **zero internet access** on the target machines, on **Windows now**, with a clean path to **Linux later**.

---

## 1. Airgap Strategy â€” The Core Idea

You cannot `pip install` or `docker pull` anything on the target machines. So the pattern is always:

```
[Build Machine â€” has internet]  â†’  [Transfer Media â€” USB / internal file share]  â†’  [Target Machines â€” no internet]
```

You need **one build machine** (can be a laptop, VM, or even a temporary cloud instance) that matches the target OS/architecture (Windows x64) to download and package everything. Do this once per Python/Spark/PyTorch version bump, not per deployment.

Two separate bundles are needed:
- **Native bundle** â†’ for the Master machine
- **Docker image bundle** â†’ for the two Worker machines

---

## 2. Complete Package Manifest

### 2.1 OS-level / system packages (both Master and Docker build)

| Package | Version | Purpose | Where to get offline installer |
|---|---|---|---|
| Python | 3.14.x (exact match) | Runtime | python.org embeddable/full installer (.exe) â€” download once |
| Java (Temurin JDK) | 17 (LTS) | Required by Spark JVM | adoptium.net `.msi` offline installer |
| Git | latest | Clone/version tracking (optional if you transfer as zip instead) | git-scm.com portable `.exe` |
| Docker Desktop | latest stable | Worker container runtime | docker.com offline installer (~500MB) |

### 2.2 Python packages (pip wheels â€” needed on Master natively, and baked into the Docker image)

| Package | Notes for airgap |
|---|---|
| `pyspark==4.2.0` | Pulls `py4j` as a sub-dependency â€” must bundle both |
| `torch` | **Largest item.** CPU-only wheel is ~200MB; CUDA-enabled (for GPU workers) is 2â€“3GB. Get the exact wheel from the PyTorch wheel index matching your CUDA version â€” do not use plain PyPI `torch`, it may resolve the wrong build |
| `torchvision` | Must match the exact `torch` version â€” mismatched pairs fail silently or crash on import |
| `numpy` | Torch/PySpark dependency, pin the version torch was built against |
| `pandas` | For results processing |
| `scikit-learn` | For benchmark model utilities |
| `matplotlib` | For `cluster_chart_*.png` generation |
| `psutil` | System stats collection |

### 2.3 Spark binary

| Item | Notes |
|---|---|
| `spark-4.2.0-bin-hadoop3.tgz` (or Windows-appropriate build) | Only needed if you're running the standalone Master process outside of what pip's `pyspark` package bundles. Since your setup launches `org.apache.spark.deploy.master.Master` using the JARs inside the `pyspark` pip package itself, **you likely don't need a separate Spark tarball** â€” the `pyspark==4.2.0` wheel already contains the Spark JARs. Confirm this before bundling a redundant copy. |

---

## 3. Phase A â€” Build Everything on an Internet-Connected Machine

Do this on any Windows x64 machine with internet (doesn't need to match your target hardware exactly, just OS/arch).

### 3.1 Download OS-level installers
Manually download and stage in a folder, e.g. `offline_bundle\installers\`:
- Python 3.14 full installer (`.exe`) â€” from python.org
- Temurin JDK 17 (`.msi`) â€” from adoptium.net
- Docker Desktop installer (`.exe`) â€” from docker.com (only needed on Worker machines)
- Git portable (optional)

### 3.2 Download all Python wheels into a local folder

```cmd
mkdir offline_bundle\wheels
cd offline_bundle\wheels

:: CPU-only torch/torchvision example â€” swap the index URL for a CUDA build if workers have GPUs
python -m pip download ^
  pyspark==4.2.0 ^
  torch torchvision --index-url https://download.pytorch.org/whl/cpu ^
  numpy pandas scikit-learn matplotlib psutil ^
  -d .
```

For GPU workers, use the matching CUDA index instead, e.g.:
```cmd
python -m pip download torch torchvision --index-url https://download.pytorch.org/whl/cu121 -d .
```
Check your worker GPU driver's supported CUDA version first (`nvidia-smi` on the worker) so the wheel actually runs there.

This produces a folder of `.whl` files â€” no internet needed to install from these later, using:
```cmd
python -m pip install --no-index --find-links=.\wheels pyspark==4.2.0 torch torchvision numpy pandas scikit-learn matplotlib psutil
```

### 3.3 Get the repo as a transferable archive
Since target machines have no GitHub access:
```cmd
git clone https://github.com/KRISHNA049039/spark_bench_mark_poc.git
cd spark_bench_mark_poc
git archive --format=zip -o ..\spark_bench_mark_poc.zip HEAD
```
This gives you a plain `.zip` â€” no git required on the target machine, just unzip it.

### 3.4 Build and export the Worker Docker image (see Section 5)

### 3.5 Transfer everything to targets
Copy to USB / internal file share:
```
offline_bundle/
â”œâ”€â”€ installers/          (python, java, docker desktop .exe/.msi)
â”œâ”€â”€ wheels/               (all .whl files)
â”œâ”€â”€ spark_bench_mark_poc.zip
â””â”€â”€ worker_image.tar      (from Section 5)
```

---

## 4. Native Deployment â€” Master (Windows)

1. **Install Python 3.14** from the staged `.exe` â€” check "Add Python to PATH."
2. **Install Java 17** from the staged `.msi` â€” check "Set JAVA_HOME."
3. **Unzip** `spark_bench_mark_poc.zip` to `d:\spark_pytorch_poc`
4. **Install Python packages from local wheels only:**
   ```cmd
   cd d:\spark_pytorch_poc
   python -m pip install --no-index --find-links=<path_to_wheels> pyspark==4.2.0 torch torchvision numpy pandas scikit-learn matplotlib psutil
   ```
   The `--no-index` flag is what stops pip from ever trying to reach PyPI â€” without it, pip will hang or fail trying to resolve dependencies online even if the wheel is present locally.
5. Verify: `python -c "import pyspark, torch; print(pyspark.__version__, torch.__version__)"`
6. Continue with the Master startup steps exactly as documented in `CLUSTER_SETUP.md` Step 1.

---

## 5. Docker Deployment â€” Workers (Windows, current)

The challenge here is that `docker compose build` normally does `apt-get`/`pip install` calls at build time, which need internet. You need to bake the offline wheels **into the build context** and modify the Dockerfile to install from local files instead of the network.

### 5.1 On the build machine â€” modify `Dockerfile.worker` to install offline

Adjust the pip install line to reference wheels bundled in the build context:
```dockerfile
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels pyspark==4.2.0 torch torchvision numpy pandas scikit-learn matplotlib psutil
```
Place your `wheels/` folder from Section 3.2 next to `Dockerfile.worker` before building, so `COPY` can see it.

If the base image itself (e.g. `python:3.14-slim`) also needs pulling, pull and tag it locally first â€” `docker pull` still needs internet, but only on the *build* machine, not the target:
```cmd
docker pull python:3.14-slim
```

### 5.2 Build the image on the internet-connected build machine
```cmd
cd cluster\hybrid
docker compose -f docker-compose.worker.yml build --no-cache
```

### 5.3 Export the built image to a portable file
```cmd
docker save -o worker_image.tar spark_bench_mark_poc-worker:latest
```
(Replace the image name with whatever `docker compose build` actually tagged it â€” check with `docker images`.)

Transfer `worker_image.tar` to each worker machine via USB/file share.

### 5.4 On each Worker machine (airgapped) â€” load and run
```cmd
docker load -i worker_image.tar
```
This registers the image locally â€” `docker compose up` will now find it without ever attempting a pull, **as long as your `docker-compose.worker.yml` doesn't have a `build:` directive active at runtime** (only `image:` referencing the loaded tag). If your compose file currently has both `build:` and `image:`, either remove `build:` on the worker machines' copy, or run `docker compose up --no-build`.

```cmd
cd spark_bench_mark_poc\cluster\hybrid
docker compose -f docker-compose.worker.yml up --no-build
```

---

## 6. Docker Desktop Windows â€” DNS / Containerâ†”Host Communication Fix

This is the part that trips people up in exactly this kind of cluster setup, so it's worth separating into the two distinct problems it actually is:

### Problem 1: Worker container â†’ Master machine (outbound)
This is **not actually a DNS problem** if you're already using raw IPs (`192.168.4.100`) as your setup does â€” Docker Desktop's NAT networking on Windows allows outbound connections from containers to any reachable LAN IP by default, whether the container runs on the WSL2 backend or Hyper-V backend. So `spark://192.168.4.100:7077` from inside the worker container should reach the native Master with no extra config, **as long as Windows Firewall on the Master machine allows inbound on 7077 and the RPC port range** (see the port table in your setup doc).

If it's still failing, check the *Master's* Windows Firewall â€” not the worker â€” since that's usually where inbound gets silently dropped:
```powershell
New-NetFirewallRule -DisplayName "Spark Master" -Direction Inbound -LocalPort 7077,8080,33000-33020 -Protocol TCP -Action Allow
```

### Problem 2: Master â†’ Worker container (callback/RPC) â€” the real DNS trap
This is the one your own troubleshooting table already flags: *"Worker shows 172.x.x.x IP â€” Docker internal IP leaking."*

Here's why it happens: by default, a process inside a Docker container reports its **own container-internal hostname/IP** (something like `172.17.0.2`) to anything it registers with â€” including the Spark Master. The Master then tries to open a return connection to `172.17.0.2`, which is only reachable *inside that container's own Docker network*, not from the Master machine or from the other worker. This looks like a DNS failure but it's actually an **address-advertisement** problem â€” Docker's embedded DNS resolver (`127.0.0.11` inside the container) is working fine; the container is just telling the outside world the wrong address to resolve.

**The fix â€” force the container to advertise its real LAN address, not its Docker-internal one:**

In `docker-compose.worker.yml`, you're already setting:
```yaml
environment:
  - SPARK_LOCAL_HOSTNAME=192.168.4.101   # or .102 for worker 2
```
This is correct and is the primary fix â€” make sure it's applied on **every** worker, not just referenced in the doc.

Also explicitly bind the Spark RPC layer to that same address rather than letting it default to the container's internal interface:
```yaml
environment:
  - SPARK_LOCAL_HOSTNAME=192.168.4.101
  - SPARK_LOCAL_IP=192.168.4.101
```

And set the container to use **host networking** if you want to eliminate the Docker NAT layer entirely (removes the container-internal IP concept altogether, so there's nothing to leak):
```yaml
network_mode: "host"
```
âš ï¸ **Caveat:** `network_mode: host` does **not work on Docker Desktop for Windows** (it's a Linux-kernel feature) â€” this only becomes an option once you migrate workers to native Linux (see Section 8). On Windows, stick with the `SPARK_LOCAL_HOSTNAME` / `SPARK_LOCAL_IP` explicit-binding approach above.

**Secondary fix â€” bypass DNS lookups for the Master's hostname entirely**, so the container never has to resolve anything, even if your config uses a hostname instead of an IP somewhere:
```yaml
extra_hosts:
  - "spark-master:192.168.4.100"
```
This writes a static entry into the container's `/etc/hosts`, so even if Docker Desktop's DNS forwarder is slow or misbehaving (which can happen on Windows when the embedded resolver tries to forward unresolved names to upstream DNS servers that aren't reachable in a locked-down network), the container never needs a DNS query for the Master at all.

**Quick verification from inside a running worker container:**
```cmd
docker exec -it <worker_container_name> ping 192.168.4.100
docker exec -it <worker_container_name> cat /etc/hosts
```

---

## 7. Verification Checklist (post-deployment, airgapped)

- [ ] `python -c "import pyspark, torch"` succeeds on Master with no network access
- [ ] `docker images` on each Worker shows the loaded image with no `docker pull` ever attempted
- [ ] Master log shows `I have been elected leader! New state: ALIVE`
- [ ] Master log shows both workers registering with their **real LAN IPs**, not `172.x.x.x`
- [ ] Spark Web UI (`http://192.168.4.100:8080`) shows 2 workers with correct core/memory counts
- [ ] `python cluster/hybrid/run_cluster.py` completes and writes to `benchmark_results/`

---

## 8. Migrating to Linux Later â€” What Actually Changes

| Area | Windows (now) | Linux (later) |
|---|---|---|
| Master install | `.exe`/`.msi` installers | `apt`/`yum` offline repo mirror, or plain `.tar.gz` extraction for Python/Java |
| Firewall | Windows Firewall rules (`New-NetFirewallRule`) | `ufw allow 7077,8080,33000:33020/tcp` or `iptables` |
| Docker networking | NAT only â€” `network_mode: host` unavailable | `network_mode: "host"` works natively, eliminating the entire DNS/IP-leak problem in Section 6 â€” containers share the host's network stack directly |
| Docker install | Docker Desktop (GUI + VM backend) | Docker Engine (native daemon, no VM layer, lighter weight) |
| Package offline install | `pip install --no-index --find-links=...` (same on both) | Identical `pip` approach, plus `apt-get install --no-download`/local `.deb` cache for system packages |
| Path separators in scripts | `d:\spark_pytorch_poc`, `set VAR=` | `/opt/spark_pytorch_poc`, `export VAR=` â€” your `run_cluster.py` and startup one-liners will need path/env-var syntax updates |

**Practical tip:** once on Linux, switch worker containers to `network_mode: host` and drop `SPARK_LOCAL_HOSTNAME` entirely â€” the container will simply use the host machine's real IP for everything, which removes an entire category of the DNS/IP-advertisement issues in Section 6 by design.

---

## 9. Airgap-Specific Troubleshooting Addendum

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` hangs for a long time then times out | Missing `--no-index`, pip is still trying to reach PyPI | Always pass `--no-index --find-links=<path>` |
| `docker compose build` fails inside airgapped network | Compose file still has a live `build:` step trying `apt-get`/`pip install` over network | Build only on the internet-connected build machine; on targets, `docker load` + `up --no-build` |
| `ImportError` for torch on worker but not on build machine | CPU-only wheel built but GPU driver present (or vice versa) â€” silent mismatch | Confirm `nvidia-smi` CUDA version on the actual worker hardware *before* building the offline bundle, not after |
| Container can't resolve `spark-master` or any hostname | Docker Desktop's embedded DNS forwards to an upstream resolver unreachable in the isolated network | Add `extra_hosts` static mapping (Section 6) or switch to raw IPs everywhere |
| Everything works on one worker, fails on the other | `SPARK_LOCAL_HOSTNAME` not updated per-machine (both still say `.101`) | Double-check the compose file was edited *after* copying to each worker, not before |

---

## 10. Mixed CPU/GPU Cluster â€” Packaging and Parameter-Based Scheduling

Your cluster can have a mix of CPU-only and GPU workers, with the job's compute mode (`cpu`, `gpu`, or `cpu+gpu`) selected as a run parameter. This requires changes at two layers: **image packaging** (Section 5) and **job scheduling** (how tasks land on the right worker). Both are detailed below, on top of the airgapped bundling already covered.

### 10.1 Why one Docker image can't serve both CPU and GPU workers

A given Python environment only has one `torch` installed at a time. A CPU-only `torch` wheel has no CUDA kernels compiled in â€” if a task calls `.cuda()` or `.to("cuda")` against it, it fails at runtime, not at build time, regardless of what scheduling parameter you pass. Conversely, a CUDA-enabled `torch` build is heavier and expects an NVIDIA driver + `nvidia-container-toolkit` to be present on the host, which a CPU-only box won't have. So you need **two separate images**, not one image with conditional logic inside.

### 10.2 Packaging â€” two Dockerfiles, two offline bundles

**`Dockerfile.worker-cpu`**
```dockerfile
FROM python:3.14-slim
COPY wheels-cpu/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels \
    pyspark==4.2.0 torch torchvision numpy pandas scikit-learn matplotlib psutil
COPY pytorch_benchmark/ /app/pytorch_benchmark/
WORKDIR /app
```

**`Dockerfile.worker-gpu`**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.14 python3-pip   # or bundle offline .deb equivalents, see 10.3
COPY wheels-gpu/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels \
    pyspark==4.2.0 torch torchvision numpy pandas scikit-learn matplotlib psutil
COPY pytorch_benchmark/ /app/pytorch_benchmark/
WORKDIR /app
```

**On the build machine (internet-connected), download both wheel sets separately:**
```cmd
:: CPU set
python -m pip download pyspark==4.2.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu numpy pandas scikit-learn matplotlib psutil -d wheels-cpu

:: GPU set â€” match the CUDA version to your actual worker GPU drivers, check with nvidia-smi on the target GPU worker first
python -m pip download pyspark==4.2.0 torch torchvision --index-url https://download.pytorch.org/whl/cu121 numpy pandas scikit-learn matplotlib psutil -d wheels-gpu
```

**Build, save, and load both images independently:**
```cmd
docker build -f Dockerfile.worker-cpu -t spark-worker-cpu:latest .
docker build -f Dockerfile.worker-gpu -t spark-worker-gpu:latest .
docker save -o worker_cpu_image.tar spark-worker-cpu:latest
docker save -o worker_gpu_image.tar spark-worker-gpu:latest
```
Transfer `worker_cpu_image.tar` only to CPU worker machines, and `worker_gpu_image.tar` only to GPU worker machines. `docker load -i <file>` on each, matching hardware to image.

### 10.3 GPU worker also needs `nvidia-container-toolkit` on the host

Unlike the CPU image, the GPU worker's Docker daemon itself needs the NVIDIA Container Toolkit installed **on the host machine** (not inside the image) so containers can see the physical GPU. This has its own offline install path â€” download the `.deb`/toolkit installer from NVIDIA on the build machine and transfer it separately; it's a one-time host-level setup, not part of the image.

In `docker-compose.worker-gpu.yml`, request the GPU explicitly:
```yaml
services:
  worker:
    image: spark-worker-gpu:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
Note: GPU passthrough to Windows containers via Docker Desktop has historically been limited/WSL2-backend-dependent â€” verify `nvidia-smi` works *inside* the running container before assuming this is wired correctly, since this is one of the more fragile parts of a Windows+GPU+Docker stack. This becomes noticeably more reliable once you migrate to native Linux + Docker Engine (Section 8), where GPU passthrough is a first-class, well-documented path.

### 10.4 Scheduling â€” let Spark route tasks by declared resource, not a manual split

Your current `CLUSTER_SETUP.md` Phase 3 hybrid mode uses "even partitions â†’ GPU, odd â†’ CPU" â€” a manual, fixed-ratio trick. For a real `cpu` / `gpu` / `cpu+gpu` parameter across *heterogeneous* workers, use Spark's built-in resource-aware scheduling (available since Spark 3.x, present in 4.2) instead, so Spark decides placement based on which workers actually declared a GPU.

**Each GPU worker, at startup, declares its resource:**
```cmd
docker compose -f docker-compose.worker-gpu.yml up \
  -e SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resource.gpu.discoveryScript=/app/getGpuResources.sh"
```

**`getGpuResources.sh`** (bundled into the GPU image, discovers the GPU index for Spark):
```bash
#!/bin/bash
ADDRS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
echo "{\"name\": \"gpu\", \"addresses\":[$ADDRS]}"
```

**CPU workers declare nothing extra** â€” they simply aren't eligible for GPU-tagged tasks and won't try to run them.

**In `run_cluster.py`, wire the mode parameter to Spark's task resource requirements:**
```python
import argparse
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cpu", "gpu", "cpu+gpu"], default="cpu")
args = parser.parse_args()

builder = SparkSession.builder.appName("pytorch-benchmark")

if args.mode == "gpu":
    # Only executors on GPU-registered workers get scheduled these tasks
    builder = builder.config("spark.task.resource.gpu.amount", "1")
    builder = builder.config("spark.executor.resource.gpu.amount", "1")
elif args.mode == "cpu+gpu":
    # Run as two separate job stages below, each with its own resource profile,
    # rather than forcing a fixed partition split â€” Spark load-balances naturally
    pass
# cpu mode: no GPU resource requested, any worker (CPU or GPU box) can take it

spark = builder.getOrCreate()
```

**For `cpu+gpu` mode**, submit the CPU-bound and GPU-bound portions of the benchmark as two separate stages, each tagged with its own `ResourceProfile`, instead of hardcoding an even/odd partition split:
```python
from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests

gpu_task_req = TaskResourceRequests().resource("gpu", 1)
gpu_profile = ResourceProfileBuilder().require(gpu_task_req).build()

cpu_rdd = data_rdd.map(run_cpu_inference)          # no profile â€” lands anywhere
gpu_rdd = data_rdd.map(run_gpu_inference).withResources(gpu_profile)  # GPU workers only
```

This approach keeps working correctly if you add a third worker later, or if a GPU worker is temporarily brought up in CPU-only mode â€” Spark re-evaluates placement based on declared resources each run, rather than you having to update a hardcoded ratio.

### 10.5 Verification additions for the mixed-cluster case

- [ ] `docker exec -it <gpu_worker_container> nvidia-smi` shows the GPU **inside the container**, not just on the host
- [ ] Spark Web UI resource tab shows GPU workers reporting `gpu: 1` under resources, CPU workers reporting none
- [ ] `--mode gpu` run's tasks only appear on GPU-worker executors in the Spark UI's stage detail view â€” if any land on a CPU worker, the resource declaration isn't being read correctly
- [ ] `--mode cpu+gpu` run shows two distinct stages in the UI, each pinned to the correct worker type

---

*Guide generated for spark_bench_mark_poc â€” pairs with your existing `CLUSTER_SETUP.md`.*