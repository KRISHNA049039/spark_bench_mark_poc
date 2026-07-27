# Cluster Setup Guide — PyTorch + Spark Distributed Benchmark

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  MASTER MACHINE (192.168.4.100)                            │
│                                                            │
│  ┌──────────────────┐  ┌───────────────────────────────┐  │
│  │ Spark Master     │  │ Benchmark Driver              │  │
│  │ Port: 7077       │  │ Runs: cluster_benchmark.py    │  │
│  │ Web UI: 8080     │  │ Submits tasks to workers      │  │
│  └──────────────────┘  └───────────────────────────────┘  │
│                                                            │
│  Runs NATIVELY (not Docker) — Python 3.14 + PySpark 4.2   │
└────────────────────────────────────────────────────────────┘
         │                              │
         │ spark://192.168.4.100:7077   │
         │                              │
┌────────▼───────────────┐  ┌──────────▼─────────────────┐
│ WORKER 1 (192.168.4.101)│  │ WORKER 2 (192.168.4.102) │
│                         │  │                           │
│  Docker container with: │  │  Docker container with:   │
│  - Python 3.14          │  │  - Python 3.14            │
│  - PySpark 4.2.0        │  │  - PySpark 4.2.0          │
│  - PyTorch (CPU/GPU)    │  │  - PyTorch (CPU/GPU)      │
│  - Spark 4.2.0          │  │  - Spark 4.2.0            │
│  - pytorch_benchmark/   │  │  - pytorch_benchmark/     │
│                         │  │                           │
│  20 cores, 28 GB RAM    │  │  20 cores, 28 GB RAM      │
│  8 GB VRAM (GPU)        │  │  8 GB VRAM (GPU)          │
└─────────────────────────┘  └───────────────────────────┘
```

## Version Requirements (MUST match across all nodes)

| Component | Version | Why |
|-----------|---------|-----|
| Python | 3.14 | Driver & worker must match exactly |
| PySpark | 4.2.0 | Spark serialization protocol must match |
| Spark | 4.2.0 | Master, worker, driver — all same version |
| Java | 17 | Required by Spark (on master machine) |

---

## One-Time Setup

### Master Machine (192.168.4.100)

**Install Python 3.14:**
- Download from https://www.python.org/downloads/
- Check "Add Python to PATH" during install

**Install Java 17:**
- Download from https://adoptium.net/temurin/releases/?os=windows&arch=x64&package=jdk&version=17
- Check "Set JAVA_HOME" during install

**Install Python packages:**
```cmd
python -m pip install pyspark==4.2.0 torch torchvision psutil numpy pandas scikit-learn matplotlib
```

**Clone the repo (if not already):**
```cmd
git clone https://github.com/KRISHNA049039/spark_bench_mark_poc.git
cd spark_bench_mark_poc
```

### Worker Machines (192.168.4.101, 192.168.4.102)

**Requirements:** Only Docker Desktop installed. Nothing else.

**Clone the repo:**
```cmd
git clone https://github.com/KRISHNA049039/spark_bench_mark_poc.git
cd spark_bench_mark_poc
```

**Edit the worker IP** in `cluster/hybrid/docker-compose.worker.yml`:
- Worker 1: Set `SPARK_LOCAL_HOSTNAME=192.168.4.101`
- Worker 2: Set `SPARK_LOCAL_HOSTNAME=192.168.4.102`

**Build the Docker image (one time only, takes ~10 min):**
```cmd
cd cluster/hybrid
docker compose -f docker-compose.worker.yml build --no-cache
```

---

## Starting the Cluster

### Step 1: Start Master (your machine)

Open a terminal:
```cmd
cd d:\spark_pytorch_poc
python -c "import pyspark, subprocess, os; subprocess.run(['java', '-cp', os.path.join(os.path.dirname(pyspark.__file__), 'jars', '*'), 'org.apache.spark.deploy.master.Master', '--host', '192.168.4.100', '--port', '7077', '--webui-port', '8080'])"
```

**Wait for:** `I have been elected leader! New state: ALIVE`

**Web UI:** Open http://192.168.4.100:8080 in browser

### Step 2: Start Workers (other machines)

On each worker machine:
```cmd
cd spark_bench_mark_poc/cluster/hybrid
docker compose -f docker-compose.worker.yml up
```

**Wait for:** Master logs show `Registering worker 192.168.4.101...` and `Registering worker 192.168.4.102...`

**Verify:** Spark UI (http://192.168.4.100:8080) shows 2 workers with cores and memory.

### Step 3: Run Benchmark (your machine, new terminal)

```cmd
cd d:\spark_pytorch_poc
python cluster/hybrid/run_cluster.py
```

---

## Startup Order & Timing

| Order | Component | Behavior if others aren't ready |
|:-----:|-----------|--------------------------------|
| 1st | Master | Waits indefinitely for workers |
| 2nd | Workers | Retry connecting to master every 5 seconds |
| 3rd | Benchmark | Checks master is up, then submits jobs |

- Workers can start in any order, any time apart
- If a worker joins mid-benchmark, Spark uses it for future tasks
- If a worker disconnects, Spark redistributes its tasks to remaining workers
- Master never times out — it waits as long as needed

---

## Stopping the Cluster

### Stop the benchmark:
Press `Ctrl+C` in the benchmark terminal.

### Stop workers:
On each worker machine:
```cmd
docker compose -f docker-compose.worker.yml down
```

### Stop master:
Press `Ctrl+C` in the master terminal.

---

## Running Individual Models

Instead of all 5 models, run one at a time:

Edit `cluster/hybrid/run_cluster.py` or set environment variable:
```cmd
set BENCHMARK_MODELS=resnet50
python cluster/hybrid/run_cluster.py
```

---

## What the Benchmark Runs

| Phase | Mode | Description |
|-------|------|-------------|
| Baseline | Local CPU | No Spark, inference on master machine only |
| Phase 1 | Distributed CPU | Tasks sent to worker CPUs via Spark |
| Phase 2 | Distributed GPU | Tasks sent to worker GPUs via Spark |
| Phase 3 | Hybrid CPU+GPU | Even partitions → GPU, Odd → CPU |

Models tested: ResNet-50, MobileNetV3, EfficientNet-B0, DistilBERT, TabularDeep

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `PYTHON_VERSION_MISMATCH` | Driver/worker Python versions differ | Rebuild workers with matching Python version in Dockerfile.worker |
| `serialVersionUID` mismatch | Spark version mismatch | Ensure pyspark==4.2.0 on master AND workers |
| `Initial job has not accepted any resources` | Workers not connected or resources exhausted | Check Spark UI, wait for workers, or reduce executor memory |
| Worker shows `172.x.x.x` IP | Docker internal IP leaking | Set `SPARK_LOCAL_HOSTNAME` to real LAN IP in compose |
| `Connection refused` to driver | Firewall blocking port 33000 | Disable firewall or allow port 33000-33020 |
| Workers retry connecting | Master not started yet | Start master first, workers will auto-connect |
| Port 7077 already in use | Old master process still running | `netstat -ano \| findstr :7077` then `taskkill /F /PID <pid>` |

---

## Network Requirements

Ensure these ports are open between all machines:

| Port | Direction | Purpose |
|------|-----------|---------|
| 7077 | Workers → Master | Spark cluster registration |
| 8080 | Browser → Master | Spark Web UI |
| 33000-33020 | Workers → Master | Driver RPC (executor callbacks) |
| 7078 | Master → Workers | Worker RPC |
| 35000-35050 | Master → Workers | Executor communication |
| 8081 | Browser → Workers | Worker Web UI (optional) |

---

## Updating Code

When you change `pytorch_benchmark/` source code:

**Master:** Changes take effect immediately (runs natively)

**Workers:** Rebuild the Docker image:
```cmd
cd cluster/hybrid
docker compose -f docker-compose.worker.yml build
docker compose -f docker-compose.worker.yml down
docker compose -f docker-compose.worker.yml up
```

Or pull latest and rebuild:
```cmd
git pull origin main
docker compose -f docker-compose.worker.yml build
docker compose -f docker-compose.worker.yml up
```

---

## Results

After benchmark completes, results are saved in `benchmark_results/`:
- `cluster_benchmark_<timestamp>.json` — Raw data
- `CLUSTER_BENCHMARK_REPORT.md` — Formatted comparison report
- `cluster_chart_*.png` — Visualization charts

---

*Last updated: 2026-07-27*
