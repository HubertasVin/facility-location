import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

csv_files = sorted(glob.glob("training_a*_g*_e*.csv"))

if not csv_files:
    print("No training files found!")
    exit(1)


def parse_params(filename):
    """Extract alpha, gamma, epsilon from filename"""
    parts = filename.replace("training_", "").replace(".csv", "").split("_")
    return {
        "alpha": float(parts[0].replace("a", "")),
        "gamma": float(parts[1].replace("g", "")),
        "epsilon": float(parts[2].replace("e", "")),
    }


def plot_parameter_comparison(files, param_name, param_values, fixed_params):
    """Create comparison plot for one parameter variation"""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e63946"]

    max_value = 31.7959
    convergence_threshold = 0.9 * max_value

    # Create three separate figures
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)

    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)

    for idx, (file, value) in enumerate(zip(files, param_values)):
        df = pd.read_csv(file)
        grouped = df.groupby("episode")["utility"].agg(
            [
                ("mean", "mean"),
                ("std", "std"),
                ("q25", lambda x: x.quantile(0.25)),
                ("q75", lambda x: x.quantile(0.75)),
                ("iqm", lambda s: s[(s >= s.quantile(0.25)) & (s <= s.quantile(0.75))].mean()),
            ]
        )

        if param_name == "epsilon" and value == 0:
            label = f"{param_name}=variable"
        else:
            label = f"{param_name}={value}"
        color = colors[idx % len(colors)]

        # 1. IQM with IQR
        ax1.plot(grouped.index, grouped["iqm"], label=label, linewidth=2.5, color=color)
        ax1.fill_between(grouped.index, grouped["q25"], grouped["q75"], alpha=0.1, color=color)

        convergence_episodes = grouped.index[grouped["iqm"] >= convergence_threshold]
        if len(convergence_episodes) > 0:
            convergence_episode = convergence_episodes[0]
            convergence_value = grouped.loc[convergence_episode, "iqm"]
            ax1.plot(
                convergence_episode,
                convergence_value,
                "x",
                markersize=14,
                markeredgewidth=4,
                color="black",
            )

            ax1.plot(
                convergence_episode,
                convergence_value,
                "x",
                markersize=12,
                markeredgewidth=2,
                color=color,
                label=f"90% convergence (ep {convergence_episode})",
            )
            print(f"{label}: Converged at episode {convergence_episode} with IQM={convergence_value:.4f}")

        # 2. Standard Deviation
        ax2.plot(grouped.index, grouped["std"], label=label, linewidth=2, color=color)

        # 3. Coefficient of Variation
        cv = (grouped["std"] / grouped["mean"]) * 100
        ax3.plot(grouped.index, cv, label=label, linewidth=2, color=color)

    # Configure and save Figure 1 (IQM)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Utility", fontsize=11)
    ax1.set_title(f"IQM with IQR - Varying {param_name.upper()}", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    filename1 = f"comparison_{param_name}_iqm.png"
    fig1.savefig(filename1, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename1}")
    plt.close(fig1)

    # Configure and save Figure 2 (Std Dev)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Standard Deviation", fontsize=11)
    ax2.set_title(f"Variability Over Time - Varying {param_name.upper()}", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    filename2 = f"comparison_{param_name}_std.png"
    fig2.savefig(filename2, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename2}")
    plt.close(fig2)

    # Configure and save Figure 3 (CV)
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("CV (%)", fontsize=11)
    ax3.set_title(f"Coefficient of Variation - Varying {param_name.upper()}", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    filename3 = f"comparison_{param_name}_cv.png"
    fig3.savefig(filename3, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename3}")
    plt.close(fig3)
    
# --- Total Reward Q-value vs Standard Q-learning Comparison ---

def plot_totalQ_comparison():
    # Look for both files
    file_std = "training_a0.4_g0.4_e0.3_totalQoff.csv"
    file_total = "training_a0.4_g0.4_e0.3_totalQon.csv"
    if not (os.path.exists(file_std) and os.path.exists(file_total)):
        print("Total Q-value comparison files not found, skipping totalQ comparison plot.")
        return

    df_std = pd.read_csv(file_std)
    df_total = pd.read_csv(file_total)

    # Group by episode and compute IQM and IQR for both
    def get_iqm_stats(df):
        grouped = df.groupby("episode")["utility"].agg(
            [
                ("mean", "mean"),
                ("std", "std"),
                ("q25", lambda x: x.quantile(0.25)),
                ("q75", lambda x: x.quantile(0.75)),
                ("iqm", lambda s: s[(s >= s.quantile(0.25)) & (s <= s.quantile(0.75))].mean()),
            ]
        )
        return grouped

    grouped_std = get_iqm_stats(df_std)
    grouped_total = get_iqm_stats(df_total)

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_std.index, grouped_std["iqm"], label="Standard Q-learning", color="#1f77b4", linewidth=2.5)
    plt.fill_between(grouped_std.index, grouped_std["q25"], grouped_std["q75"], alpha=0.1, color="#1f77b4")

    plt.plot(grouped_total.index, grouped_total["iqm"], label="Total Reward Q-value", color="#e63946", linewidth=2.5)
    plt.fill_between(grouped_total.index, grouped_total["q25"], grouped_total["q75"], alpha=0.1, color="#e63946")

    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Utility", fontsize=11)
    plt.title("IQM with IQR: Standard Q-learning vs Total Reward Q-value\n(a=0.4, g=0.4, e=0.3)", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = "comparison_totalQ_a0.4_g0.4_e0.3.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


alpha_files = []
gamma_files = []
epsilon_files = []

for f in csv_files:
    params = parse_params(f)

    if params["gamma"] == 0.2 and params["epsilon"] == 0.2:
        alpha_files.append(f)
    elif params["alpha"] == 0.8 and params["epsilon"] == 0.2:
        gamma_files.append(f)
    elif params["alpha"] == 0.8 and params["gamma"] == 0.2:
        epsilon_files.append(f)

# Sort files by the varying parameter value
alpha_files.sort(key=lambda f: parse_params(f)["alpha"])
gamma_files.sort(key=lambda f: parse_params(f)["gamma"])
epsilon_files.sort(key=lambda f: parse_params(f)["epsilon"])

print(f"Found {len(alpha_files)} alpha variations")
print(f"Found {len(gamma_files)} gamma variations")
print(f"Found {len(epsilon_files)} epsilon variations")

# Create comparison plots
if alpha_files:
    alpha_values = [parse_params(f)["alpha"] for f in alpha_files]
    plot_parameter_comparison(alpha_files, "alpha", alpha_values, {"gamma": 0.2, "epsilon": 0.2})

if gamma_files:
    gamma_values = [parse_params(f)["gamma"] for f in gamma_files]
    plot_parameter_comparison(gamma_files, "gamma", gamma_values, {"alpha": 0.8, "epsilon": 0.2})

if epsilon_files:
    epsilon_values = [parse_params(f)["epsilon"] for f in epsilon_files]
    plot_parameter_comparison(epsilon_files, "epsilon", epsilon_values, {"alpha": 0.8, "gamma": 0.2})
    
plot_totalQ_comparison()

# Generate summary table
print("\nGenerating summary statistics...")
summary_data = []

for csv_file in csv_files:
    params = parse_params(csv_file)
    df = pd.read_csv(csv_file)

    grouped = df.groupby("episode")["utility"]
    final_iqm = grouped.apply(lambda s: s[(s >= s.quantile(0.25)) & (s <= s.quantile(0.75))].mean()).iloc[-1]
    final_std = grouped.std().iloc[-1]
    final_cv = (final_std / grouped.mean().iloc[-1]) * 100

    summary_data.append(
        {
            "Alpha": params["alpha"],
            "Gamma": params["gamma"],
            "Epsilon": params["epsilon"],
            "Final_IQM": final_iqm,
            "Final_CV": final_cv,
        }
    )

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values("Final_IQM", ascending=False)
summary_df.to_csv("summary_rankings.csv", index=False)
print("\nSaved: summary_rankings.csv")
print("\nTop configurations by Final IQM:")
print(summary_df.head())
