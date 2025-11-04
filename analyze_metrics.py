import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Read parameters from command line or environment
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

df = pd.read_csv(f"training_a{alpha:g}_g{gamma:g}_e{epsilon:g}.csv")


def compute_iqm(series):
    """Compute Interquartile Mean: mean of middle 50%"""
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    return series[(series >= q25) & (series <= q75)].mean()


def detect_convergence(series, threshold=0.95, window=1000):
    """Detect convergence point where utility reaches threshold of max"""
    max_val = series.max()
    target = max_val * threshold
    converged = series[series >= target]
    return converged.index[0] if len(converged) > 0 else None


# Group statistics
grouped = df.groupby("episode")["utility"].agg(
    [
        ("mean", "mean"),
        ("std", "std"),
        ("q25", lambda x: x.quantile(0.25)),
        ("q75", lambda x: x.quantile(0.75)),
        ("iqm", compute_iqm),
    ]
)

# Detect convergence
convergence_point = detect_convergence(grouped["iqm"], threshold=0.99)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
if epsilon == 0:
    fig.suptitle(f"Training Metrics (α={alpha}, γ={gamma}, ε=variable)", fontsize=16, fontweight="bold")
else:
    fig.suptitle(f"Training Metrics (α={alpha}, γ={gamma}, ε={epsilon})", fontsize=16, fontweight="bold")

# 1. IQM with convergence indicator
ax = axes[0]
ax.plot(grouped.index, grouped["iqm"], label="IQM", linewidth=2.5, color="#2E86AB")
ax.fill_between(grouped.index, grouped["q25"], grouped["q75"], alpha=0.1, color="#2E86AB", label="IQR")
if convergence_point:
    ax.axvline(x=convergence_point, color="red", linestyle="--", label=f"99% Convergence (ep {convergence_point})")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Utility", fontsize=11)
ax.set_title("IQM with Convergence Detection", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Variability over time
ax = axes[1]
ax.plot(grouped.index, grouped["std"], linewidth=2, color="#C73E1D")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Standard Deviation", fontsize=11)
ax.set_title("Stability Metric: Standard Deviation Over Time", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

# 3. Coefficient of Variation
ax = axes[2]
cv = (grouped["std"] / grouped["mean"]) * 100
ax.plot(grouped.index, cv, linewidth=2, color="#6A994E")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("CV (%)", fontsize=11)
ax.set_title("Coefficient of Variation (Lower = More Stable)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout()
filename = f"metrics_a{alpha}_g{gamma}_e{epsilon}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")
plt.close()

summary_file = f"convergence_a{alpha}_g{gamma}_e{epsilon}.txt"
with open(summary_file, "w") as f:
    f.write(f"Training Configuration:\n")
    f.write(f"  Alpha: {alpha}\n")
    f.write(f"  Gamma: {gamma}\n")
    f.write(f"  Epsilon: {epsilon}\n\n")
    f.write(f"Convergence Analysis:\n")
    if convergence_point:
        f.write(f"  95% Convergence at episode: {convergence_point}\n")
    else:
        f.write(f"  Did not reach 95% convergence\n")
    f.write(f"  Final IQM: {grouped['iqm'].iloc[-1]:.4f}\n")
    f.write(f"  Final Std Dev: {grouped['std'].iloc[-1]:.4f}\n")
    f.write(f"  Final CV: {cv.iloc[-1]:.2f}%\n")
