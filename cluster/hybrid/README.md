# Hybrid Cluster: Native Driver + Docker Workers

This approach solves Docker Desktop's networking problem on Windows by:
- Running the **driver/master natively** on the host (real LAN IP)
- Running **workers in Docker** containers (pre-built, zero manual setup)

## Why This Works

| Component | Runs In | Why |
|-----------|---------|-----|
| Master | Host | Needs real IP for workers to register |
| Driver | Host | Needs real IP for executors to callback |
| Workers | Docker | Only needs to ACCEPT connections (port-mapped) |

The problem with full-Docker was: the driver inside Docker advertised `192.168.65.x` (Docker VM).
Now the driver runs on the host at `192.168.4.100` — executors can reach it directly.

## Prerequisites

### Master Machine (192.168.4.100) — install once:
```cmd
python -m pip install pyspark torch torchvision psutil numpy pandas scikit-learn matplotlib
```
That's it. No Java needed (PySpark bundles it).

### Worker Machines (192.168.4.101, .102) — just Docker:
Already have Docker installed. Nothing else needed.

## Steps

### 1. Build worker image (each worker machine, one time only):
```cmd
cd spark_bench_mark_poc/cluster/hybrid
docker compose -f docker-compose.worker.yml build
```

### 2. Start workers (each worker machine):
```cmd
cd spark_bench_mark_poc/cluster/hybrid
docker compose -f docker-compose.worker.yml up
```

### 3. Run benchmark (master machine, host terminal):
```cmd
cd spark_bench_mark_poc
python cluster/hybrid/run_cluster.py
```

## That's it. 3 commands total across all machines.
