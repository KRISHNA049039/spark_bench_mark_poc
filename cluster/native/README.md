# Native Multi-Node Cluster Setup (No Docker)

This runs Spark + PyTorch natively on each Windows machine, avoiding Docker Desktop's networking issues.

## Prerequisites (install on ALL machines)

1. **Python 3.11+** — https://www.python.org/downloads/
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
```cmd
cd spark_bench_mark_poc
pip install -r cluster/native/requirements-native.txt
```

### Step 2: Download Spark on ALL machines
```cmd
cd cluster/native
python download_spark.py
```

### Step 3: Start Master (Node 1 only)
```cmd
cluster\native\start_master.bat
```

### Step 4: Start Workers (Node 2 and Node 3)
```cmd
cluster\native\start_worker.bat
```

### Step 5: Run Benchmark (Node 1)
```cmd
python -m pytorch_benchmark.cluster_benchmark
```

Check Spark UI: http://192.168.4.100:8080
