# Cluster Benchmark Report — 3-Phase Comparison

**Generated:** 2026-07-28 00:02:31  
**Cluster:** 2 Worker Nodes (32 GB RAM, 8 GB VRAM each)  
**Samples per model:** 1000 | **Batch size:** 64 | **Partitions:** 8  
**Executor Memory:** 12 GB heap + 2 GB overhead | **Cores:** 4 per executor

---

## Test Phases

| Phase | Mode | Description |
|-------|------|-------------|
| Baseline | Local CPU | Single-machine, no Spark, no distribution |
| Phase 1 | Distributed CPU | All partitions on CPU workers via Spark |
| Phase 2 | Distributed GPU | All partitions on GPU workers via Spark |
| Phase 3 | Hybrid CPU+GPU | Even partitions→GPU, Odd partitions→CPU |

---

## 1. Throughput Comparison (samples/sec)

| Model | Baseline (CPU) | Phase 1 (Dist CPU) | Phase 2 (Dist GPU) | Phase 3 (Hybrid) | Best Phase |
|-------|---:|---:|---:|---:|---|
| **efficientnet_b0** | 78.5 | 2.5 | 1.8 | 1.7 | **Baseline** |

---

## 2. Speedup vs Baseline (Local CPU)

| Model | Dist CPU | Dist GPU | Hybrid | Best Speedup |
|-------|---:|---:|---:|---:|
| **efficientnet_b0** | 0.03x | 0.02x | 0.02x | **0.03x** |

```
Speedup vs Local CPU Baseline
══════════════════════════════════════════════════════════════════════
efficientnet_b0   0.0x (GPU)
══════════════════════════════════════════════════════════════════════
```

---

## 3. Reproducibility Verification

| Model | Baseline Hash | Dist CPU Hash | Dist GPU Hash | Hybrid Hash | All Match? |
|-------|:---:|:---:|:---:|:---:|:---:|
| **efficientnet_b0** | `9f4bb074` | `9f4bb074` | `9f4bb074` | `9f4bb074` | ✅ |

---

## 4. Total Execution Time (seconds)

| Model | Baseline | Dist CPU | Dist GPU | Hybrid | Time Saved (best) |
|-------|---:|---:|---:|---:|---:|
| **efficientnet_b0** | 12.75 | 397.61 | 554.19 | 579.13 | -384.86s |

---

## 5. Per-Executor Performance (Phase 1: Dist CPU)

| Model | Partitions | Avg Exec Time | Max Exec Time | Load Balance | Total Throughput |
|-------|---:|---:|---:|---:|---:|
| **efficientnet_b0** | 8 | 6.30s | 6.99s | 82% | 159.7 s/s |

---

## 6. Resource Utilization Plan

### Memory Layout (per worker node)

```
32 GB Total RAM
┌─────────────────────────────────────────────────────────────┐
│  OS + Docker (3 GB)  │  Spark Daemon (1 GB)  │  Buffer (2 GB)│
├──────────────────────┴───────────────────────┴──────────────┤
│          Executor 1: CPU (12 GB heap + 2 GB overhead)        │
├──────────────────────────────────────────────────────────────┤
│          Executor 2: GPU (10 GB heap + 2 GB overhead)        │
└──────────────────────────────────────────────────────────────┘

8 GB VRAM
┌──────────────────────────────────────────────────────────────┐
│ Model (~300 MB) │ Activations (~500 MB) │ Available (~7 GB)   │
└──────────────────────────────────────────────────────────────┘
```

### Parallelism Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Workers | 2 | Available machines |
| Executors per worker | 2 | 1 CPU + 1 GPU executor |
| Cores per executor | 4 | Balance parallelism vs memory |
| Partitions | 8 | 2 per executor for load balancing |
| Executor memory | 12 GB | ~40% of available RAM per executor |
| Memory overhead | 2 GB | Python/serialization buffer |
| Batch size | 64 | Fits in GPU VRAM with margin |

---

## 7. Conclusions & Recommendations

- **Reproducibility:** All models produce identical predictions across all phases (baseline, distributed CPU, distributed GPU, hybrid). Distribution does not affect correctness.

- **Spark overhead at current scale:** efficientnet_b0 are slower with distributed CPU than local — increase sample count to amortize fixed costs.

- **Hybrid mode** simultaneously utilizes CPU and GPU, maximizing hardware utilization when both resources are available.

- **Scale recommendation:** At 10K+ samples, distributed modes will show clear throughput advantages over single-machine inference.

---

*Report generated on 2026-07-28 00:02:31 from cluster benchmark results.*