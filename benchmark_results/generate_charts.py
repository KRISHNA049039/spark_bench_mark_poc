"""
Generate colorful charts from inference benchmark results.
Run inside Docker: docker compose run --rm inference-resnet50 python /app/benchmark_results/generate_charts.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

RESULTS_DIR = "/app/benchmark_results"
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the all-models run
report_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith("inference_only_") and f.endswith(".json")])
if not report_files:
    raise FileNotFoundError("No inference_only_*.json found")

# Use the latest file (which has all 5 models)
for rf in reversed(report_files):
    with open(os.path.join(RESULTS_DIR, rf)) as f:
        data = json.load(f)
    if len(data) >= 5:
        break

print(f"Loaded {len(data)} models from {rf}")

# Style setup
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'torch_cpu': '#2196F3',   # Blue
    'torch_gpu': '#4CAF50',   # Green
    'spark_cpu': '#FF9800',   # Orange
    'spark_gpu': '#9C27B0',   # Purple
}
MODE_LABELS = {
    'torch_cpu': 'Torch+CPU',
    'torch_gpu': 'Torch+GPU',
    'spark_cpu': 'Spark+CPU',
    'spark_gpu': 'Spark+GPU',
}

MODEL_COLORS = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
models = [d['model'] for d in data]


# ---------------------------------------------------------------------------
# Chart 1: Throughput Comparison (Grouped Bar)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(models))
width = 0.2
modes = ['torch_cpu', 'torch_gpu', 'spark_cpu', 'spark_gpu']

for i, mode in enumerate(modes):
    throughputs = []
    for d in data:
        if mode in d['modes'] and 'error' not in d['modes'][mode]:
            throughputs.append(d['modes'][mode]['throughput_samples_per_sec'])
        else:
            throughputs.append(0)

    bars = ax.bar(x + i * width, throughputs, width,
                  label=MODE_LABELS[mode], color=COLORS[mode], edgecolor='white', linewidth=0.5)

ax.set_xlabel('Model')
ax.set_ylabel('Throughput (samples/sec)')
ax.set_title('Inference Throughput — All Models x All Modes', fontweight='bold', pad=15)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([d['model'] for d in data], fontweight='bold')
ax.set_yscale('log')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, axis='y')
ax.set_axisbelow(True)

# Add value labels on bars
for i, mode in enumerate(modes):
    for j, d in enumerate(data):
        if mode in d['modes'] and 'error' not in d['modes'][mode]:
            val = d['modes'][mode]['throughput_samples_per_sec']
            if val > 1000:
                label = f'{val/1000:.1f}K'
            else:
                label = f'{val:.0f}'
            ax.text(j + i * width, val * 1.15, label,
                    ha='center', va='bottom', fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_01_throughput_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_01_throughput_all.png")


# ---------------------------------------------------------------------------
# Chart 2: GPU Speedup
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

speedups = []
model_names = []
for d in data:
    if 'torch_cpu' in d['modes'] and 'torch_gpu' in d['modes']:
        cpu_t = d['modes']['torch_cpu']['throughput_samples_per_sec']
        gpu_t = d['modes']['torch_gpu']['throughput_samples_per_sec']
        speedups.append(gpu_t / cpu_t)
        model_names.append(d['model'])

# Sort by speedup
sorted_idx = np.argsort(speedups)[::-1]
speedups = [speedups[i] for i in sorted_idx]
model_names = [model_names[i] for i in sorted_idx]

bars = ax.barh(model_names, speedups, color=MODEL_COLORS[:len(model_names)],
               edgecolor='white', linewidth=1.5, height=0.6)

# Add value labels
for bar, val in zip(bars, speedups):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}x', ha='left', va='center', fontweight='bold', fontsize=12)

ax.set_xlabel('Speedup (GPU / CPU)', fontweight='bold')
ax.set_title('GPU Speedup Over CPU (RTX 5060 vs CPU)', fontweight='bold', pad=15)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlim(0, max(speedups) * 1.2)
ax.grid(True, axis='x')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_02_gpu_speedup.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_02_gpu_speedup.png")


# ---------------------------------------------------------------------------
# Chart 3: Latency Heatmap
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

latency_data = []
for d in data:
    row = []
    for mode in modes:
        if mode in d['modes'] and 'error' not in d['modes'][mode]:
            row.append(d['modes'][mode]['avg_latency_ms'])
        else:
            row.append(0)
    latency_data.append(row)

latency_arr = np.array(latency_data)

im = ax.imshow(latency_arr, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, label='Avg Latency (ms)')

ax.set_xticks(range(len(modes)))
ax.set_xticklabels([MODE_LABELS[m] for m in modes], fontweight='bold')
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontweight='bold')
ax.set_title('Inference Latency Heatmap (ms/sample)', fontweight='bold', pad=15)

# Add text annotations
for i in range(len(models)):
    for j in range(len(modes)):
        val = latency_arr[i, j]
        color = 'white' if val > latency_arr.max() * 0.6 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_03_latency_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_03_latency_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 4: GPU Memory Usage
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

gpu_mem = []
model_sizes = []
gpu_models = []

for d in data:
    if 'torch_gpu' in d['modes'] and 'error' not in d['modes'][mode]:
        res = d['modes']['torch_gpu'].get('resources', {})
        mem = res.get('gpu_peak_memory_mb', 0)
        if mem > 0:
            gpu_mem.append(mem)
            model_sizes.append(d['modes']['torch_gpu']['model_info']['size_mb'])
            gpu_models.append(d['model'])

x_pos = np.arange(len(gpu_models))
bar_width = 0.35

bars1 = ax.bar(x_pos - bar_width/2, model_sizes, bar_width,
               label='Model Weights', color='#42A5F5', edgecolor='white')
bars2 = ax.bar(x_pos + bar_width/2, gpu_mem, bar_width,
               label='Peak GPU VRAM', color='#EF5350', edgecolor='white')

ax.set_xlabel('Model')
ax.set_ylabel('Memory (MB)')
ax.set_title('GPU Memory: Model Size vs Peak VRAM Usage', fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(gpu_models, fontweight='bold')
ax.legend()
ax.grid(True, axis='y')
ax.set_axisbelow(True)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_04_gpu_memory.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_04_gpu_memory.png")


# ---------------------------------------------------------------------------
# Chart 5: Reproducibility Matrix
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

repro_matrix = []
for d in data:
    hashes = d['reproducibility']['hashes']
    row = []
    for m1 in modes:
        for m2 in modes:
            if m1 in hashes and m2 in hashes:
                row.append(1 if hashes[m1] == hashes[m2] else 0)
            else:
                row.append(-1)
    repro_matrix.append(row)

# Simplified: just show per-model if all match
match_data = np.zeros((len(models), 4))  # models x mode-pairs
pair_labels = ['CPU modes\nmatch', 'GPU modes\nmatch', 'All modes\nmatch', 'Cross-device\nmatch']

for i, d in enumerate(data):
    hashes = d['reproducibility']['hashes']
    cpu_hashes = [hashes.get('torch_cpu', ''), hashes.get('spark_cpu', '')]
    gpu_hashes = [hashes.get('torch_gpu', ''), hashes.get('spark_gpu', '')]

    # CPU modes match
    match_data[i, 0] = 1 if len(set(cpu_hashes)) == 1 and cpu_hashes[0] != '' else 0
    # GPU modes match
    match_data[i, 1] = 1 if len(set(gpu_hashes)) == 1 and gpu_hashes[0] != '' else 0
    # All match
    match_data[i, 2] = 1 if d['reproducibility']['all_match'] else 0
    # Cross-device
    match_data[i, 3] = 1 if cpu_hashes[0] == gpu_hashes[0] and cpu_hashes[0] != '' else 0

# Custom colormap: red=fail, green=pass
cmap = matplotlib.colors.ListedColormap(['#EF5350', '#4CAF50'])
im = ax.imshow(match_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(4))
ax.set_xticklabels(pair_labels, fontweight='bold')
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontweight='bold')
ax.set_title('Reproducibility Matrix', fontweight='bold', pad=15)

# Add check/cross marks
for i in range(len(models)):
    for j in range(4):
        symbol = '✓' if match_data[i, j] == 1 else '✗'
        color = 'white'
        ax.text(j, i, symbol, ha='center', va='center', color=color, fontweight='bold', fontsize=16)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', label='Match (Reproducible)'),
    mpatches.Patch(facecolor='#EF5350', label='Differ (Expected for some GPU ops)'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_05_reproducibility.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_05_reproducibility.png")


# ---------------------------------------------------------------------------
# Chart 6: Spark Overhead
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

overhead_pct = []
model_labels = []
for d in data:
    if 'torch_cpu' in d['modes'] and 'spark_cpu' in d['modes']:
        cpu_time = d['modes']['torch_cpu']['total_time_sec']
        spark_time = d['modes']['spark_cpu']['total_time_sec']
        overhead = ((spark_time - cpu_time) / cpu_time) * 100
        overhead_pct.append(overhead)
        model_labels.append(d['model'])

# Sort
sorted_idx = np.argsort(overhead_pct)[::-1]
overhead_pct = [overhead_pct[i] for i in sorted_idx]
model_labels = [model_labels[i] for i in sorted_idx]

colors_overhead = ['#EF5350' if v > 50 else '#FF9800' if v > 20 else '#4CAF50' for v in overhead_pct]
bars = ax.barh(model_labels, overhead_pct, color=colors_overhead, edgecolor='white', height=0.6)

for bar, val in zip(bars, overhead_pct):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'+{val:.0f}%', ha='left', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Overhead (%)', fontweight='bold')
ax.set_title('Spark CPU Overhead vs Direct Torch CPU', fontweight='bold', pad=15)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax.grid(True, axis='x')
ax.set_axisbelow(True)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#EF5350', label='>50% overhead (compute-light models)'),
    mpatches.Patch(facecolor='#FF9800', label='20-50% overhead (medium models)'),
    mpatches.Patch(facecolor='#4CAF50', label='<20% overhead (compute-heavy models)'),
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_06_spark_overhead.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_06_spark_overhead.png")


# ---------------------------------------------------------------------------
# Chart 7: Model Size vs Throughput (Bubble Chart)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

for i, d in enumerate(data):
    if 'torch_gpu' in d['modes'] and 'error' not in d['modes']['torch_gpu']:
        size_mb = d['modes']['torch_gpu']['model_info']['size_mb']
        gpu_throughput = d['modes']['torch_gpu']['throughput_samples_per_sec']
        cpu_throughput = d['modes']['torch_cpu']['throughput_samples_per_sec']
        params = d['modes']['torch_gpu']['model_info']['total_params']

        # Bubble size proportional to params
        bubble_size = max(100, params / 5000)

        ax.scatter(size_mb, gpu_throughput, s=bubble_size, color=MODEL_COLORS[i],
                   alpha=0.7, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(f"  {d['model']}\n  ({params/1e6:.1f}M params)",
                    (size_mb, gpu_throughput), fontsize=10, fontweight='bold',
                    color=MODEL_COLORS[i])

        # Also plot CPU as smaller dots
        ax.scatter(size_mb, cpu_throughput, s=bubble_size * 0.3, color=MODEL_COLORS[i],
                   alpha=0.4, marker='s', zorder=4)

ax.set_xlabel('Model Size (MB)', fontweight='bold')
ax.set_ylabel('Throughput (samples/sec)', fontweight='bold')
ax.set_title('Model Size vs Inference Speed (circles=GPU, squares=CPU)', fontweight='bold', pad=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_07_size_vs_speed.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_07_size_vs_speed.png")


# ---------------------------------------------------------------------------
# Chart 8: Total Time Breakdown (Stacked)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.18

for i, mode in enumerate(modes):
    times = []
    for d in data:
        if mode in d['modes'] and 'error' not in d['modes'][mode]:
            times.append(d['modes'][mode]['total_time_sec'])
        else:
            times.append(0)
    bars = ax.bar(x + i * width, times, width,
                  label=MODE_LABELS[mode], color=COLORS[mode], edgecolor='white')

ax.set_xlabel('Model')
ax.set_ylabel('Total Inference Time (seconds)')
ax.set_title('Total Inference Time — 200 Samples Per Model', fontweight='bold', pad=15)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_08_total_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart_08_total_time.png")


print(f"\n{'='*60}")
print(f"All 8 charts saved to: {RESULTS_DIR}/")
print(f"{'='*60}")
