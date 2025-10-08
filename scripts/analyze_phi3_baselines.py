import os
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
BASE_DIR = r"C:\Users\vishnuu\Projects\RAG\Phi-3-mini-4k-instruct\results"
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Files to analyze ===
files = {
    "4-bit": os.path.join(BASE_DIR, "phi3_baseline_results_4bit.csv"),
    "8-bit": os.path.join(BASE_DIR, "phi3_baseline_results_8bit.csv"),
    "16-bit": os.path.join(BASE_DIR, "phi3_baseline_results_16bit.csv"),
    "32-bit": os.path.join(BASE_DIR, "phi3_baseline_results_32bit.csv"),
}

# === Load and merge all CSVs ===
dataframes = []
for quant, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Quantization"] = quant
        dataframes.append(df)
    else:
        print(f"⚠️ File not found: {path}")

if not dataframes:
    raise FileNotFoundError("No CSV files were found. Please check the paths.")

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv(os.path.join(OUTPUT_DIR, "phi3_baseline_combined.csv"), index=False)
print(f"✅ Combined CSV saved at: {OUTPUT_DIR}/phi3_baseline_combined.csv")

# === Compute Summary Stats ===
numeric_cols = [col for col in combined_df.select_dtypes(include="number").columns]
summary = (
    combined_df.groupby("Quantization")[numeric_cols]
    .mean()
    .round(3)
    .reset_index()
)

summary.to_csv(os.path.join(OUTPUT_DIR, "phi3_baseline_summary.csv"), index=False)
print(f"✅ Summary CSV saved at: {OUTPUT_DIR}/phi3_baseline_summary.csv")

print("\n=== Summary Overview ===")
print(summary)

# === Visualization ===
def plot_metric(metric_name):
    if metric_name not in combined_df.columns:
        print(f"⚠️ Metric '{metric_name}' not found in CSVs.")
        return
    plt.figure(figsize=(7, 5))
    plt.title(f"{metric_name} vs Quantization Level")
    plt.xlabel("Quantization")
    plt.ylabel(metric_name)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.plot(summary["Quantization"], summary[metric_name], marker="o", linewidth=2)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{metric_name}_vs_quantization.png")
    plt.savefig(save_path, dpi=300)
    print(f"📊 Saved plot: {save_path}")
    plt.close()

# Try plotting these if they exist
for metric in ["Latency", "GPU_Utilization", "Memory_Used", "Accuracy", "Time_Taken"]:
    if metric in combined_df.columns:
        plot_metric(metric)

print("\n✅ All analysis complete.")
print(f"📁 Check output folder: {OUTPUT_DIR}")
