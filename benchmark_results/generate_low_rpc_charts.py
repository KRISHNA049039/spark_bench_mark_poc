"""Generate charts from Low-RPC cluster benchmark results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(RESULTS_DIR, "cluster_low_rpc_20260728_180310.json")) as f:
    data = json.load(f)

models = [k for k in data.keys() if not k.startswith("_")]
print(f"Loaded {len(models)} models from cluster_low_rpc_20260728_180310.json")
COLORS = {"baseline_cpu": "#607D8B", "phase1_dist_cpu": "#2196F3", "phase2_dist_gpu": "#4CAF50", "phase3_hybrid": "#FF9800"}
LABELS = {"baseline_cpu": "Baseline CPU", "phase1_dist_cpu": "Dist CPU", "phase2_dist_gpu": "Dist GPU", "phase3_hybrid": "Hybrid"}

# Chart 1: Throughput
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(models))
width = 0.2
phases = list(COLORS.keys())

for i, phase in enumerate(phases):
    tps = [data[m].get(phase, {}).get("throughput_samples_per_sec", 0) for m in models]
    ax.bar(x + i*width, tps, width, label=LABELS[phase], color=COLORS[phase], edgecolor='white')

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylabel("Throughput (samples/sec)")
ax.set_title("Low-RPC Benchmark — Throughput (All Models, All Phases)", fontweight='bold')
ax.set_yscale("log")
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "low_rpc_chart_throughput.png"), dpi=150)
plt.close()
print("Saved: low_rpc_chart_throughput.png")

# Chart 2: GPU Speedup per partition
fig, ax = plt.subplots(figsize=(12, 6))
gpu_speedups = []
model_labels = []
for m in models:
    p1 = data[m].get("phase1_dist_cpu", {}).get("executor_metrics", {}).get("avg_exec_time_sec", 1)
    p2 = data[m].get("phase2_dist_gpu", {}).get("executor_metrics", {}).get("avg_exec_time_sec", 1)
    if p2 > 0:
        gpu_speedups.append(p1 / p2)
        model_labels.append(m)

colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax.barh(model_labels, gpu_speedups, color=colors[:len(model_labels)], edgecolor='white', height=0.6)
for bar, val in zip(bars, gpu_speedups):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}x', va='center', fontweight='bold')
ax.set_xlabel("GPU Speedup (per-executor: CPU time / GPU time)")
ax.set_title("Per-Executor GPU Speedup (RTX 5060)", fontweight='bold')
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "low_rpc_chart_gpu_speedup.png"), dpi=150)
plt.close()
print("Saved: low_rpc_chart_gpu_speedup.png")

# Chart 3: GPU VRAM usage
fig, ax = plt.subplots(figsize=(10, 6))
vram = []
model_names = []
for m in models:
    partitions = data[m].get("phase2_dist_gpu", {}).get("executor_metrics", {}).get("per_partition", [])
    if partitions and partitions[0].get("gpu", {}).get("peak_memory_mb", 0) > 0:
        vram.append(partitions[0]["gpu"]["peak_memory_mb"])
        model_names.append(m)

bars = ax.bar(model_names, vram, color=['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC'], edgecolor='white')
for bar, val in zip(bars, vram):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f} MB', ha='center', fontweight='bold')
ax.set_ylabel("Peak GPU VRAM (MB)")
ax.set_title("GPU Memory Usage Per Model (Phase 2: Dist GPU)", fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "low_rpc_chart_gpu_vram.png"), dpi=150)
plt.close()
print("Saved: low_rpc_chart_gpu_vram.png")

# Chart 4: Hybrid breakdown (GPU vs CPU partitions)
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
for idx, m in enumerate(models):
    ax = axes[idx]
    partitions = data[m].get("phase3_hybrid", {}).get("executor_metrics", {}).get("per_partition", [])
    if not partitions:
        continue
    ids = [p["partition_id"] for p in partitions]
    times = [p["execution_time_sec"] for p in partitions]
    colors_p = ['#4CAF50' if p["device"] == "cuda:0" else '#2196F3' for p in partitions]
    ax.bar(ids, times, color=colors_p, edgecolor='white')
    ax.set_title(m, fontweight='bold', fontsize=10)
    ax.set_xlabel("Partition")
    if idx == 0:
        ax.set_ylabel("Exec Time (s)")
import matplotlib.patches as mpatches
legend = [mpatches.Patch(color='#4CAF50', label='GPU'), mpatches.Patch(color='#2196F3', label='CPU')]
fig.legend(handles=legend, loc='upper right')
fig.suptitle("Phase 3 Hybrid: GPU (even) vs CPU (odd) Partition Times", fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "low_rpc_chart_hybrid_breakdown.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: low_rpc_chart_hybrid_breakdown.png")

# Chart 5: Total time comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.2
for i, phase in enumerate(phases):
    times = [data[m].get(phase, {}).get("total_time_sec", 0) for m in models]
    ax.bar(x + i*width, times, width, label=LABELS[phase], color=COLORS[phase], edgecolor='white')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylabel("Total Time (seconds)")
ax.set_title("Total Inference Time — All Phases (200 samples)", fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "low_rpc_chart_total_time.png"), dpi=150)
plt.close()
print("Saved: low_rpc_chart_total_time.png")

print("\nAll charts generated!")
