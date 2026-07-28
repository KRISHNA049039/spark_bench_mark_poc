# Low-RPC Methodology — Eliminating Network Overhead in Spark Cluster

## Problem Statement

Yesterday's cluster benchmark showed **87% of execution time was overhead, not inference**:

| Root Cause | Time Wasted | % of Total |
|-----------|---:|:---:|
| Block transfer retries (Docker IP mismatch) | 200s | 50% |
| Model serialization over network (73 MB/task) | 60s | 15% |
| Task scheduling + dispatch | 40s | 10% |
| Data partition transfer (75 MB/task) | 47s | 12% |
| Actual inference | 50s | **13%** |

**Total:** 397.6 seconds for EfficientNet-B0 on 1000 samples.

---

## Solution: Self-Contained Worker Tasks

### Core Principle

> **Send instructions, not data.** Workers already have everything they need — let them load models and generate data locally.

### Architecture Change

```
BEFORE (Heavy RPC):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Driver                                          Worker
  ┌──────────┐                                   ┌──────────┐
  │ 1. Pickle │  ──── 73 MB model weights ──────►│ Unpickle │
  │    model  │  ──── 75 MB input data ─────────►│ Load     │
  │           │                                   │ Infer    │
  │           │◄──── large result arrays ────────│ Return   │
  │           │      (triggers block transfer)    │          │
  └──────────┘      ↑ FAILS on Docker Desktop    └──────────┘
                    ↑ 584 MB total per phase

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AFTER (Low RPC):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Driver                                          Worker
  ┌──────────┐                                   ┌──────────┐
  │ Send tiny │  ──── 200 bytes config ─────────►│ Load     │
  │ config:   │       (model_name, seed,          │ model    │
  │ model name│        partition_id, phase)        │ from     │
  │ seed      │                                   │ cache    │
  │ phase     │                                   │          │
  │           │◄──── 500 bytes metrics ──────────│ Generate │
  │           │      (hash, time, throughput)      │ data     │
  └──────────┘      ✅ Fits in RPC always         │ locally  │
                                                  │ Infer    │
                                                  │ Return   │
                                                  │ hash     │
                                                  └──────────┘
                    Total: ~3 KB per phase
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## What Changed

### 1. No Model Serialization Over Network

| Before | After |
|--------|-------|
| Driver pickles model (73 MB) | Driver sends model NAME (20 bytes) |
| Sent via Spark task closure | Worker loads from torchvision cache |
| Every task deserializes 73 MB | Model downloaded once, cached on disk |

**How:** Workers call `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` directly. The pretrained weights are downloaded once to `~/.cache/torch/hub/checkpoints/` and reused.

### 2. No Input Data Transfer

| Before | After |
|--------|-------|
| Driver generates all data | Worker generates its partition |
| 75 MB numpy array per task | Worker uses `np.random.RandomState(seed + partition_id)` |
| Serialized and sent over network | Generated locally in ~0.1s |

**How:** Each worker creates its own data slice deterministically from `seed + partition_id`. Same seed + same partition_id = same data every time = reproducible.

### 3. Tiny Results (No Block Transfer)

| Before | After |
|--------|-------|
| Return full prediction arrays | Return prediction HASH only |
| Large numpy arrays → block transfer | 16-char hex string |
| Block transfer fails on Docker Desktop | Fits in Spark's RPC message |

**How:** Worker computes `sha256(predictions.tobytes())[:16]` and returns just the hash + timing metrics. Total result size: ~500 bytes.

### 4. Single SparkSession (Reused)

| Before | After |
|--------|-------|
| New SparkSession per phase | One session for entire benchmark |
| `spark.stop()` + reconnect overhead | Session stays alive |
| ~10s overhead per phase | 0s overhead between phases |

**How:** Global `_spark_session` variable, created once, reused across all models and phases.

### 5. Increased RPC Message Size

| Before | After |
|--------|-------|
| Default 128 MB max RPC message | 256 MB max |
| Results could exceed limit → block transfer | Results are 500 bytes (always fits) |

**Config:** `spark.rpc.message.maxSize=256`

---

## File Created

**`pytorch_benchmark/cluster_benchmark_low_rpc.py`**

Key functions:
- `worker_inference(config_tuple)` — Self-contained worker function, receives ~200 bytes, returns ~500 bytes
- `run_distributed_phase(model_name, phase)` — Distributes tiny configs via Spark
- `run_baseline(model_name)` — Local CPU reference
- `get_spark()` — Reusable SparkSession singleton

---

## How to Run

```cmd
cd d:\spark_pytorch_poc

REM Start master (Terminal 1)
python -c "import pyspark, subprocess, os; subprocess.run(['java', '-cp', os.path.join(os.path.dirname(pyspark.__file__), 'jars', '*'), 'org.apache.spark.deploy.master.Master', '--host', '192.168.4.100', '--port', '7077', '--webui-port', '8080'])"

REM Run benchmark (Terminal 2)
set SPARK_MASTER=spark://192.168.4.100:7077
set SPARK_DRIVER_HOST=192.168.4.100
set SPARK_DRIVER_PORT=33000
set BENCHMARK_MODELS=efficientnet_b0
set BENCHMARK_SAMPLES=200
set BENCHMARK_PARTITIONS=4
python -m pytorch_benchmark.cluster_benchmark_low_rpc
```

---

## Expected Performance Comparison

| Metric | Before (Heavy RPC) | After (Low RPC) | Improvement |
|--------|---:|---:|---:|
| Network transfer per phase | 584 MB | 3 KB | **194,000x less** |
| Block transfer retries | 200s | 0s | **Eliminated** |
| Model serialization | 60s | 0s | **Eliminated** |
| Data transfer | 47s | 0s | **Eliminated** |
| Task scheduling | 40s | ~5s | **8x faster** |
| Total time (EfficientNet) | 397s | ~30-50s | **8-13x faster** |
| Throughput | 2.5 s/s | ~20-40 s/s | **8-16x** |

---

## Reproducibility Guarantee

Even though workers generate data locally (not from driver), reproducibility is guaranteed because:

1. **Fixed seed:** `np.random.RandomState(seed + partition_id)` — same partition always gets same data
2. **Deterministic model:** Same `torch.manual_seed(seed)` before model creation
3. **Same weights:** torchvision pretrained weights are identical on every machine
4. **Deterministic inference:** `torch.backends.cudnn.deterministic = True`

**Verification:** Hash from each partition is combined deterministically:
```python
combined_hash = sha256("".join(partition_hashes)).hexdigest()[:16]
```

---

## Trade-offs

| Aspect | Before | After | Trade-off |
|--------|--------|-------|-----------|
| Reproducibility vs single-machine | Same data everywhere | Partition-based data generation | Hashes differ from single-machine (different data slicing) |
| Model freshness | Exact model from driver | torchvision cached version | Same if cache is up to date |
| Custom models | Any model can be sent | Only models workers can load | Need code on workers |
| Flexibility | Driver controls everything | Workers are self-sufficient | Workers need pytorch_benchmark installed |

---

## When to Use Which Approach

| Scenario | Use Heavy RPC | Use Low RPC |
|----------|:---:|:---:|
| Small model (<1 MB), few samples | ✅ | Overkill |
| Large model (>10 MB), network | ❌ | ✅ |
| Docker Desktop on Windows | ❌ Block transfer fails | ✅ No block transfer |
| Native Linux / cloud | ✅ Works fine | Also works (faster) |
| Custom model (not pretrained) | ✅ (must send weights) | Need shared filesystem |
| Production batch scoring | ❌ Wastes bandwidth | ✅ Efficient |

---

## Summary

The fundamental insight: **Don't send data over the network when workers can produce it locally.** In ML inference:
- Model weights are publicly available (torchvision, HuggingFace)
- Input data can be generated deterministically from a seed
- Only the final prediction matters — send the hash, not the array

This eliminates all 5 overhead sources from yesterday's benchmark and bypasses Docker Desktop's block transfer limitation entirely.

---

*Created: 2026-07-28. Addresses issues documented in SLOW_THROUGHPUT_ANALYSIS.md.*
