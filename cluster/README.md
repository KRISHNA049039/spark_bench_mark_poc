# Multi-Node Spark Cluster Setup

## What Goes Where

### Node 1 (Master + Driver)

```
node1/
├── Dockerfile.worker          <- shared image
├── pytorch_benchmark/         <- full source code
├── benchmark_results/         <- results appear here
└── cluster/
    └── docker-compose.master.yml
```

### Node 2+ (Workers)

```
node2/
├── Dockerfile.worker          <- same image
├── pytorch_benchmark/         <- workers need this to run model code
└── cluster/
    └── docker-compose.worker.yml
```

---

## Step-by-Step

### 1. Copy files to both machines

On both machines, you need:
- `Dockerfile.worker`
- `pytorch_benchmark/` folder (full code)
- The respective compose file

### 2. Start Master (Node 1)

```bash
cd cluster
docker compose -f docker-compose.master.yml build
docker compose -f docker-compose.master.yml up
```

Note the IP address of Node 1 (e.g., `192.168.1.100`)
Check master is running: http://node1-ip:8080

### 3. Start Worker (Node 2)

```bash
cd cluster
docker compose -f docker-compose.worker.yml build

# Set MASTER_IP to Node 1's IP address
MASTER_IP=192.168.1.100 docker compose -f docker-compose.worker.yml up
```

On Windows:
```cmd
set MASTER_IP=192.168.1.100
docker compose -f docker-compose.worker.yml up
```

### 4. Verify

Open http://node1-ip:8080 — you should see:
- 1 worker connected
- Worker resources (4 cores, 4GB)

### 5. Benchmark runs automatically

The driver (on Node 1) waits 15 seconds for workers to connect, then runs:
- All 5 models (ResNet-50, MobileNetV3, EfficientNet-B0, DistilBERT, TabularDeep)
- Modes: torch_cpu (local) + spark_cpu (distributed to workers)
- Results saved to `benchmark_results/`

---

## Adding More Workers

Just run `docker-compose.worker.yml` on more machines with the same MASTER_IP.
Each worker adds more parallelism for the Spark tasks.

## Network Requirements

- Node 1 port **7077** must be accessible from workers (Spark master)
- Node 1 port **8080** for web UI (optional)
- Workers need outbound access to Node 1

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Worker can't connect | Check firewall allows port 7077 |
| "No module pytorch_benchmark" | Ensure `pytorch_benchmark/` is copied to worker |
| Out of memory | Reduce `SPARK_WORKER_MEMORY` or `--samples` |
| Slow first run | First run downloads model weights (~100MB) |
