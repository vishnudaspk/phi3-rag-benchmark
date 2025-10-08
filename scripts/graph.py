import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSVs
combined_path = "results/analysis/phi3_baseline_combined.csv"
summary_path  = "results/analysis/phi3_baseline_summary.csv"

combined = pd.read_csv(combined_path)
summary  = pd.read_csv(summary_path)

# --------------------------
# Prepare single figure with 2x2 subplots
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16,12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --------------------------
# 1. Inference Time per Quantization (Bar)
# --------------------------
sns.barplot(x='Quantization', y='inference_time_sec', data=summary, ax=axes[0,0], palette='Blues_d')
axes[0,0].set_title("Inference Time per Quantization")
axes[0,0].set_ylabel("Time (sec)")

# --------------------------
# 2. GPU Memory Usage (Start & End) (Line)
# --------------------------
ax = axes[0,1]
ax.plot(summary['Quantization'], summary['gputil_mem_start_MB'], marker='o', label='GPU Mem Start')
ax.plot(summary['Quantization'], summary['gputil_mem_end_MB'], marker='o', label='GPU Mem End')
ax.set_title("GPU Memory Usage (MB)")
ax.set_ylabel("Memory (MB)")
ax.legend()
ax.set_xlabel("Quantization")

# --------------------------
# 3. GPU Load (%) (Start & End) (Line)
# --------------------------
ax = axes[1,0]
ax.plot(summary['Quantization'], summary['gputil_load_start_pct'], marker='s', label='GPU Load Start')
ax.plot(summary['Quantization'], summary['gputil_load_end_pct'], marker='s', label='GPU Load End')
ax.set_title("GPU Load (%)")
ax.set_ylabel("Load (%)")
ax.legend()
ax.set_xlabel("Quantization")

# --------------------------
# 4. NVSMM Metrics (Memory & Load) (Optional Line)
# --------------------------
ax = axes[1,1]
ax.plot(summary['Quantization'], summary['nvsmi_mem_start_MB'], marker='^', label='NVSMI Mem Start')
ax.plot(summary['Quantization'], summary['nvsmi_mem_end_MB'], marker='^', label='NVSMI Mem End')
ax.plot(summary['Quantization'], summary['nvsmi_load_start_pct'], marker='x', label='NVSMI Load Start')
ax.plot(summary['Quantization'], summary['nvsmi_load_end_pct'], marker='x', label='NVSMI Load End')
ax.set_title("NVSMI Metrics")
ax.set_ylabel("Value (MB / %)")
ax.legend()
ax.set_xlabel("Quantization")

plt.tight_layout()
plt.show()
