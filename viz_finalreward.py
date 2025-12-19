#!/usr/bin/env python3
import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


FILENAME_RE = re.compile(
    r"^training_a(?P<a>[-+]?\d*\.?\d+)_g(?P<g>[-+]?\d*\.?\d+)_e(?P<e>[-+]?\d*\.?\d+)"
    r"(?P<suffix>_finalRewardON|_finalRewardOFF)\.csv$"
)


@dataclass(frozen=True)
class FinalRewardMeta:
    path: str
    alpha_str: str
    gamma_str: str
    epsilon_str: str
    alpha: float
    gamma: float
    epsilon: float
    suffix: str  # "_finalRewardON" or "_finalRewardOFF"


def parse_totalq_filename(path: str) -> FinalRewardMeta:
    base = os.path.basename(path)
    m = FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Unrecognized finalReward filename format: {base}")

    a_str = m.group("a")
    g_str = m.group("g")
    e_str = m.group("e")
    suffix = m.group("suffix")

    return FinalRewardMeta(
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


def main():
    ap = argparse.ArgumentParser(description="FinalReward on/off comparison plots (auto-paired by filename).")
    ap.add_argument("--dir", default=".", help="Directory containing training_*_finalRewardON/off.csv (default: .)")
    ap.add_argument("--outdir", default=".", help="Where to write plots (default: .)")
    args = ap.parse_args()

    pattern = os.path.join(args.dir, "training_a*_g*_e*_finalReward*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No finalReward comparison files found in: {args.dir}")
        return 0

    metas: List[FinalRewardMeta] = []
    for p in paths:
        try:
            metas.append(parse_totalq_filename(p))
        except ValueError as e:
            print(f"[WARN] {e}")

    # Pair by (a,g,e)
    by_key: Dict[Tuple[str, str, str], Dict[str, FinalRewardMeta]] = {}
    for m in metas:
        key = (m.alpha_str, m.gamma_str, m.epsilon_str)
        by_key.setdefault(key, {})[m.suffix] = m

    ensure_outdir(args.outdir)

    made_any = False
    for (a_str, g_str, e_str), d in sorted(by_key.items()):
        if "_finalRewardOFF" not in d or "_finalRewardON" not in d:
            missing = [s for s in ("_finalRewardOFF", "_finalRewardON") if s not in d]
            print(f"[WARN] Missing {missing} for a={a_str}, g={g_str}, e={e_str}; skipping.")
            continue

        off = d["_finalRewardOFF"]
        on = d["_finalRewardON"]

        df_off = pd.read_csv(off.path)
        df_on = pd.read_csv(on.path)

        st_off = iqm_iqr_stats(df_off)
        st_on = iqm_iqr_stats(df_on)

        plt.figure(figsize=(10, 6))

        plt.plot(st_off.index, st_off["iqm"], label="Standard Q-learning (finalReward off)", color="#1f77b4", linewidth=2.5)
        plt.fill_between(st_off.index, st_off["q25"], st_off["q75"], alpha=0.12, color="#1f77b4")

        plt.plot(st_on.index, st_on["iqm"], label="Total Reward Q-value (finalReward on)", color="#e63946", linewidth=2.5)
        plt.fill_between(st_on.index, st_on["q25"], st_on["q75"], alpha=0.12, color="#e63946")

        plt.xlabel("Episode", fontsize=11)
        plt.ylabel("Utility", fontsize=11)
        plt.title(f"IQM with IQR: finalReward off vs on\n(a={a_str}, g={g_str}, e={e_str})", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        outname = os.path.join(args.outdir, f"comparison_finalReward_a{a_str}_g{g_str}_e{e_str}.png")
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {outname}")
        made_any = True

    if not made_any:
        print("No complete finalReward pairs found (need both *_finalRewardOFF.csv and *_finalRewardON.csv).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
