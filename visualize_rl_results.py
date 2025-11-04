import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("training_curves.csv")


def compute_iqm(series):
    """Compute Interquartile Mean: mean of middle 50%"""
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    return series[(series >= q25) & (series <= q75)].mean()


grouped = df.groupby("episode")["utility"].agg(
    [
        ("mean", "mean"),
        ("median", "median"),
        ("q25", lambda x: x.quantile(0.25)),
        ("q75", lambda x: x.quantile(0.75)),
        ("iqm", compute_iqm),
    ]
)

plt.figure(figsize=(12, 6))
plt.plot(grouped.index, grouped["iqm"], label="IQM (1000-ep avg)", linewidth=2.5, color="#2E86AB")
plt.fill_between(grouped.index, grouped["q25"], grouped["q75"], alpha=0.3, color="#2E86AB", label="Interquartile Range")
plt.xlabel("Episode (end of 100-episode window)", fontsize=12)
plt.ylabel("Average Utility (per 100 episodes)", fontsize=12)
plt.title("Interquantile mean", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve_1000ep_avg.png", dpi=300, bbox_inches="tight")
plt.show()
