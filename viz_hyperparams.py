#!/usr/bin/env python3
import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILENAME_RE = re.compile(
    r"^training_a(?P<a>[-+]?\d*\.?\d+)_g(?P<g>[-+]?\d*\.?\d+)_e(?P<e>[-+]?\d*\.?\d+)"
    r"(?P<suffix>_finalRewardON|_finalRewardOFF)?\.csv$"
)


@dataclass(frozen=True)
class RunMeta:
    path: str
    alpha_str: str
    gamma_str: str
    epsilon_str: str
    alpha: float
    gamma: float
    epsilon: float
    suffix: Optional[str]


def parse_training_filename(path: str) -> RunMeta:
    base = os.path.basename(path)
    m = FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Unrecognized filename format: {base}")

    a_str = m.group("a")
    g_str = m.group("g")
    e_str = m.group("e")
    suffix = m.group("suffix")

    return RunMeta(
        path=path,
        alpha_str=a_str,
        gamma_str=g_str,
        epsilon_str=e_str,
        alpha=float(a_str),
        gamma=float(g_str),
        epsilon=float(e_str),
        suffix=suffix,
    )


def iqm_iqr_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: episode, utility
    Returns index=episode with columns: mean, std, q25, q75, iqm
    """
    def _iqm(s: pd.Series) -> float:
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        mid = s[(s >= q25) & (s <= q75)]
        return float(mid.mean()) if len(mid) else float("nan")

    grouped = df.groupby("episode")["utility"].agg(
        mean="mean",
        std="std",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        iqm=_iqm,
    )
    return grouped


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def pretty_fixed_params(fixed: Dict[str, float]) -> str:
    # stable ordering for titles/filenames
    keys = ["alpha", "gamma", "epsilon"]
    parts = [f"{k}={fixed[k]:g}" for k in keys if k in fixed]
    return ", ".join(parts)


def fixed_suffix_for_filename(fixed: Dict[str, float]) -> str:
    # Used to avoid overwriting when you have multiple sweeps
    keys = ["alpha", "gamma", "epsilon"]
    parts = [f"{k[0]}{fixed[k]:g}" for k in keys if k in fixed]
    return "_".join(parts) if parts else "all"


def plot_parameter_comparison(
    metas: List[RunMeta],
    vary_param: str,
    fixed_params: Dict[str, float],
    outdir: str,
    target_utility: Optional[float],
    treat_e0_as_variable: bool,
) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e63946", "#9467bd", "#8c564b"]

    # Load & compute stats first (also allows dynamic threshold if no target_utility)
    stats_by_meta: List[Tuple[RunMeta, pd.DataFrame]] = []
    max_iqm_seen = -np.inf

    for meta in metas:
        df = pd.read_csv(meta.path)
        st = iqm_iqr_stats(df)
        stats_by_meta.append((meta, st))
        local_max = np.nanmax(st["iqm"].to_numpy())
        if np.isfinite(local_max):
            max_iqm_seen = max(max_iqm_seen, local_max)

    if target_utility is not None:
        max_value = float(target_utility)
    else:
        if not np.isfinite(max_iqm_seen):
            print(f"[WARN] No finite IQM values found for varying {vary_param}, skipping plots.")
            return
        max_value = float(max_iqm_seen)

    convergence_threshold = 0.9 * max_value

    # --- create three figures
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Sort by varied parameter value
    metas_sorted = sorted(metas, key=lambda m: getattr(m, vary_param))

    for idx, meta in enumerate(metas_sorted):
        st = next(s for (m, s) in stats_by_meta if m == meta)

        value = getattr(meta, vary_param)
        value_str = getattr(meta, f"{vary_param}_str")

        if vary_param == "epsilon" and treat_e0_as_variable and abs(value) < 1e-12:
            label = "epsilon=variable"
        else:
            label = f"{vary_param}={value_str}"

        color = colors[idx % len(colors)]

        # 1) IQM with IQR
        ax1.plot(st.index, st["iqm"], label=label, linewidth=2.5, color=color)
        ax1.fill_between(st.index, st["q25"], st["q75"], alpha=0.12, color=color)

        convergence_eps = st.index[st["iqm"] >= convergence_threshold]
        if len(convergence_eps) > 0:
            ep0 = int(convergence_eps[0])
            y0 = float(st.loc[ep0, "iqm"])

            print(f"{vary_param} sweep ({pretty_fixed_params(fixed_params)}): {label} converged at ep {ep0} (IQM={y0:.4f})")

        # 2) Standard deviation
        ax2.plot(st.index, st["std"], label=label, linewidth=2.0, color=color)

        # 3) Coefficient of variation (%)
        cv = (st["std"] / st["mean"]) * 100.0
        ax3.plot(st.index, cv, label=label, linewidth=2.0, color=color)

    # Titles / labels
    fixed_txt = pretty_fixed_params(fixed_params)
    fixed_file = fixed_suffix_for_filename(fixed_params)

    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Utility", fontsize=11)
    ax1.set_title(f"IQM with IQR — varying {vary_param.upper()}\nFixed: {fixed_txt}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Standard Deviation", fontsize=11)
    ax2.set_title(f"STD over time — varying {vary_param.upper()}\nFixed: {fixed_txt}", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("CV (%)", fontsize=11)
    ax3.set_title(f"Coefficient of Variation — varying {vary_param.upper()}\nFixed: {fixed_txt}", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ensure_outdir(outdir)
    fn1 = os.path.join(outdir, f"comparison_{vary_param}_{fixed_file}_iqm.png")
    fn2 = os.path.join(outdir, f"comparison_{vary_param}_{fixed_file}_std.png")
    fn3 = os.path.join(outdir, f"comparison_{vary_param}_{fixed_file}_cv.png")

    fig1.tight_layout()
    fig1.savefig(fn1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(fn2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3.tight_layout()
    fig3.savefig(fn3, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved: {fn1}")
    print(f"Saved: {fn2}")
    print(f"Saved: {fn3}")


def build_groups_for_param(metas: List[RunMeta], vary_param: str) -> Dict[Tuple[float, float], List[RunMeta]]:
    """
    For a chosen vary_param, group runs by the other two params.
    Example: vary_param='alpha' => group key=(gamma, epsilon) -> list of metas.
    """
    if vary_param == "alpha":
        key_fn = lambda m: (m.gamma, m.epsilon)
    elif vary_param == "gamma":
        key_fn = lambda m: (m.alpha, m.epsilon)
    elif vary_param == "epsilon":
        key_fn = lambda m: (m.alpha, m.gamma)
    else:
        raise ValueError(f"Unknown vary_param: {vary_param}")

    groups: Dict[Tuple[float, float], List[RunMeta]] = {}
    for m in metas:
        groups.setdefault(key_fn(m), []).append(m)
    return groups


def fixed_params_from_key(vary_param: str, key: Tuple[float, float]) -> Dict[str, float]:
    if vary_param == "alpha":
        g, e = key
        return {"gamma": g, "epsilon": e}
    if vary_param == "gamma":
        a, e = key
        return {"alpha": a, "epsilon": e}
    if vary_param == "epsilon":
        a, g = key
        return {"alpha": a, "gamma": g}
    raise ValueError(vary_param)


def generate_summary(metas: List[RunMeta], outdir: str) -> None:
    rows = []
    for meta in metas:
        df = pd.read_csv(meta.path)
        st = iqm_iqr_stats(df)

        final_iqm = float(st["iqm"].iloc[-1])
        final_std = float(st["std"].iloc[-1])
        final_mean = float(st["mean"].iloc[-1])
        final_cv = float((final_std / final_mean) * 100.0) if final_mean != 0 else float("nan")

        rows.append(
            {
                "file": os.path.basename(meta.path),
                "Alpha": meta.alpha,
                "Gamma": meta.gamma,
                "Epsilon": meta.epsilon,
                "Final_IQM": final_iqm,
                "Final_CV": final_cv,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("Final_IQM", ascending=False)
    ensure_outdir(outdir)
    outpath = os.path.join(outdir, "summary_rankings.csv")
    summary_df.to_csv(outpath, index=False)

    print(f"Saved: {outpath}")
    print("\nTop configurations by Final IQM:")
    print(summary_df.head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter sweep visualisations (auto-grouped by filename).")
    ap.add_argument("--dir", default=".", help="Directory containing training_*.csv files (default: .)")
    ap.add_argument("--outdir", default=".", help="Where to write plots/csv (default: .)")
    ap.add_argument(
        "--target-utility",
        type=float,
        default=None,
        help="If set, convergence is 90%% of this value. Otherwise uses 90%% of max IQM seen in each plot.",
    )
    ap.add_argument(
        "--include-finalreward",
        action="store_true",
        help="Also include *_finalRewardON/off.csv in hyperparameter plots/summary (default: excluded).",
    )
    ap.add_argument(
        "--treat-e0-as-variable",
        action="store_true",
        help="If epsilon==0 in filename, label as variable epsilon (default: off).",
    )
    args = ap.parse_args()

    pattern = os.path.join(args.dir, "training_a*_g*_e*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No training files found in: {args.dir}")
        return 1

    metas_all: List[RunMeta] = []
    for p in paths:
        try:
            meta = parse_training_filename(p)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        if (not args.include_finalreward) and meta.suffix in ("_finalRewardON", "_finalRewardOFF"):
            continue

        metas_all.append(meta)

    if not metas_all:
        print("No usable training files after filtering.")
        return 1

    # 1) Make plots for each parameter, automatically grouped by the other two.
    for vary in ["alpha", "gamma", "epsilon"]:
        groups = build_groups_for_param(metas_all, vary)
        for key, group_metas in groups.items():
            distinct_vals = sorted({getattr(m, vary) for m in group_metas})
            if len(distinct_vals) < 2:
                continue  # not actually a sweep

            fixed = fixed_params_from_key(vary, key)
            print(f"\nPlotting sweep: vary={vary}, fixed={pretty_fixed_params(fixed)} (n={len(group_metas)})")
            plot_parameter_comparison(
                metas=group_metas,
                vary_param=vary,
                fixed_params=fixed,
                outdir=args.outdir,
                target_utility=args.target_utility,
                treat_e0_as_variable=args.treat_e0_as_variable,
            )

    # 2) Summary table across all included runs
    print("\nGenerating summary statistics...")
    generate_summary(metas_all, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
