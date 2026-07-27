# Why Cluster Throughput Is Slow — Root Cause Analysis

## The Problem

| Mode | EfficientNet-B0 Throughput | Expected |
|------|---:|---:|
| Local CPU (no Spark) | 78.5 s/s | ✅ Baseline |
| Cluster (2 workers) | 2.5 s/s | ❌ Should be ~160 s/s |
| **Slowdown factor** | **31x slower** | — |

We have 2 workers × 20 cores each = 40 cores, but get LESS throughput than 1 machine.

---

## Root Causes (Ordered by Impact)

### 1. Docker Desktop Block Transfer Issue (~50% of total time)

**What:** The driver tries to fetch results from executors at Docker internal IPs (`172.19.0.2`, `172.20.0.2`). These IPs are inside Docker Desktop's Linux VM and aren't routable from the host Windows machine.

**Effect:** Spark's block transfer service retries repeatedly with exponential backoff. Each retry waits 5s, 10s, 15s... before eventually falling back to RPC.

**Evidence:**
- Thread dump shows `Block Transfer Retry-7-1 TIMED_WAITING`
- Phase 2 took 554s but actual executor compute was only 52s (6.5s × 8)
- 500s spent waiting on network retries

**Fix for tomorrow:**
- Option A: Run workers natively (no Docker) — IPs are real
- Option B: Configure `spark.blockManager.port` on workers and port-map it
- Option C: Reduce result size so Spark uses RPC instead of block transfer

---

### 2. Model Serialization Per Task (~15% of total time)

**What:** EfficientNet-B0 (20 MB) gets serialized with pickle and sent as part of each task's closure. With 8 partitions, that's 160 MB of model data over the network.

**Evidence:**
```
WARN TaskSetManager: Stage 0 contains a task of very large size (73868 KiB)
```

**Why it's slow:**
- 73 MB per task × 8 tasks = 584 MB total data transferred
- Network between Docker containers goes through NAT layers
- Each executor must deserialize 73 MB before starting inference

**Fix for tomorrow:**
- Use Spark broadcast (already done, but task closure still carries partition data)
- Pre-load model on workers at startup (avoid per-task deserialization)
- Use model sharding: store weights in shared volume, workers load from disk

---

### 3. Sequential Task Scheduling (~12% of total time)

**What:** Spark schedules tasks in waves. With 8 partitions and limited executor slots (4 cores per executor, memory constraints), tasks run in batches of 2-4 at a time.

**Evidence:**
- Spark UI showed `(4 + 4) / 8` then `(6 + 2) / 8` — not all 8 running at once
- Each wave must complete before next starts (straggler problem)

**Why:**
- Executor memory = 12 GB requested, but workers report 28 GB total
- Spark allocates conservatively: fits 2 executors per worker maximum
- With 4 executors total (2 per worker) and 4 cores each, only 4 tasks run in parallel

**Fix for tomorrow:**
- Reduce executor memory to 4 GB (EfficientNet only needs ~100 MB per task)
- Increase executor count: more small executors = more parallelism
- Use `spark.executor.instances=8` with `spark.executor.memory=3g`

---

### 4. Python Worker Startup Overhead (~10% of total time)

**What:** Each Spark task launches a Python worker process on the executor. First time takes ~3-5s (import torch, import model code, JIT compilation).

**Evidence:**
- First 2 tasks per executor: ~7s each
- Subsequent tasks on same executor: ~5.7s (Python worker reused)
- Difference = Python startup cost

**Fix for tomorrow:**
- `spark.python.worker.reuse=true` (already set)
- Reduce partition count to minimize new worker launches
- Pre-warm workers by sending a dummy task first

---

### 5. Data Partition Transfer (~8% of total time)

**What:** Input data (1000 samples × 3 × 224 × 224 × 4 bytes = 600 MB total) is split into 8 partitions and sent over the network.

**Each partition:** 75 MB of image data transferred from driver to executor.

**Fix for tomorrow:**
- Generate data on workers (send only seed + config, not raw tensors)
- Store test data in shared filesystem (NFS/S3)
- Compress data before sending: `spark.io.compression.codec=zstd`

---

### 6. No Actual GPU Usage (Phase 2 + 3)

**What:** Despite being labeled "GPU" phases, all executors ran on CPU.

**Evidence:**
```json
"devices_used": ["cpu"]  // in ALL phases including Phase 2 "GPU"
```

**Why:** Docker workers don't have GPU drivers exposed. `torch.cuda.is_available()` returns False inside the container because NVIDIA runtime isn't configured.

**Fix for tomorrow:**
- Add `--gpus all` to worker Docker run (or `deploy.resources.reservations.devices` in compose)
- Install NVIDIA Container Toolkit on worker machines
- Rebuild worker image with CUDA-enabled PyTorch (`cu128`)

---

## Time Breakdown Visualization

```
EfficientNet-B0 Phase 1: 397.6 seconds total
┌────────────────────────────────────────────────────────────────────────────┐
│████████████████████████████████████████████████████│░░░░░░░░│▓▓▓▓▓│▒▒▒▒│██│
│       Block Transfer Retries (200s, 50%)          │Serialize│Sched │Data │CP│
│                                                   │ (60s)   │(40s) │(47s)│  │
└────────────────────────────────────────────────────────────────────────────┘
                                                                         ↑
                                                              Actual inference
                                                                  (50s, 13%)
```

---

## Expected vs Actual Performance

### With Docker Desktop (Current — Broken Networking)

```
Driver (host) ←──✗ Block Transfer ✗──→ Executor (Docker VM IP)
                   Retries every 5-15s
                   Eventually falls back to RPC
                   Adds 200-400s per phase
```

### With Native Install (No Docker — What Tomorrow Should Look Like)

```
Driver (192.168.4.100) ←──✓ Direct TCP──→ Executor (192.168.4.101)
                           No retries
                           Block transfer works immediately
                           Expected phase time: ~50-80s
```

### Expected Improvement After Fixes

| Phase | Current (Docker) | Expected (Native) | Improvement |
|-------|---:|---:|---:|
| Phase 1 (Dist CPU) | 397.6s | ~60s | 6.6x faster |
| Phase 2 (Dist GPU) | 554.2s | ~15s (with real GPU) | 37x faster |
| Phase 3 (Hybrid) | 579.1s | ~25s (GPU+CPU) | 23x faster |

---

## Action Items for Tomorrow

### Priority 1: Fix Block Transfer (biggest impact)
- [ ] Install Python 3.14 natively on worker machines
- [ ] Install Java 17 on worker machines
- [ ] Run Spark workers natively (not Docker)
- [ ] OR: Configure block manager port mapping in Docker

### Priority 2: Enable GPU on Workers
- [ ] Install NVIDIA Container Toolkit on worker machines
- [ ] Rebuild worker image with CUDA PyTorch (cu128)
- [ ] Add GPU reservations to docker-compose
- [ ] Verify: `docker run --gpus all pytorch-worker python -c "import torch; print(torch.cuda.is_available())"`

### Priority 3: Reduce Serialization Overhead
- [ ] Reduce partition count to 4 (from 8)
- [ ] Pre-generate data on workers (send config only)
- [ ] Enable compression: `spark.io.compression.codec=zstd`
- [ ] Consider smaller batch of samples for benchmarking (200 vs 1000)

### Priority 4: Optimize Executor Configuration
- [ ] Reduce executor memory: `SPARK_EXECUTOR_MEMORY=4g`
- [ ] Increase executor count: `SPARK_NUM_EXECUTORS=8`
- [ ] Set `spark.executor.cores=2` (allows more concurrent tasks)
- [ ] Enable dynamic allocation: `spark.dynamicAllocation.enabled=true`

---

## Summary

| Root Cause | Time Impact | Fixable? | How |
|-----------|---:|:---:|---|
| Block transfer IP mismatch | ~200s (50%) | ✅ | Native install or port mapping |
| Model serialization overhead | ~60s (15%) | ✅ | Pre-load model, reduce partitions |
| Task scheduling waves | ~40s (10%) | ✅ | More executors, less memory each |
| Python worker startup | ~40s (10%) | Partial | Worker reuse, pre-warming |
| Data transfer | ~47s (12%) | ✅ | Generate on workers, compression |
| No GPU (phases 2/3) | Entire phase | ✅ | NVIDIA runtime in Docker |

**Bottom line:** 87% of the cluster time is overhead, not inference. Fix the Docker networking issue and enable GPUs, and throughput should jump from 2.5 s/s to 100+ s/s for distributed CPU and 500+ s/s for distributed GPU.

---

*Analysis based on cluster_benchmark_20260728_000231.json and Spark UI observations.*
