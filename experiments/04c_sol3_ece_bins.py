"""Solution 4c.3: ECE is an estimate, and the bins matter.

ECE bins confidence and averages |accuracy - confidence| per bin. The number
of bins and the binning scheme change the estimate: too few bins hide
miscalibration (averaging opposite errors within a wide bin); too many bins
make each estimate noisy. Equal-mass (adaptive) bins are more stable than
equal-width bins when confidence concentrates near 1. We sweep the bin count
for both schemes on a fixed overconfident model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import CalibrationDataset, ece


def main() -> None:
    ds = CalibrationDataset(n_items=8000, sharpness=2.5, seed=0)
    bin_counts = [2, 3, 5, 8, 10, 15, 20, 30, 50, 80]

    equal_width = [ece(ds.confidence, ds.correct, n_bins=b, adaptive=False)
                   for b in bin_counts]
    equal_mass = [ece(ds.confidence, ds.correct, n_bins=b, adaptive=True)
                  for b in bin_counts]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(bin_counts, equal_width, "o-", color="#c44e52", linewidth=2,
            markersize=6, label="equal-width bins")
    ax.plot(bin_counts, equal_mass, "s-", color="#3a7ebf", linewidth=2,
            markersize=6, label="equal-mass (adaptive) bins")

    ax.set_xscale("log")
    ax.set_xlabel("number of bins (log scale)")
    ax.set_ylabel("estimated ECE")
    ax.set_title(
        "ECE is a binned estimate. Too few bins understate miscalibration;\n"
        "equal-mass bins are steadier than equal-width when confidence clusters.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, which="both")
    ax.set_axisbelow(True)
    ax.set_xticks(bin_counts); ax.set_xticklabels(bin_counts)

    print(f"  equal-width ECE range: {min(equal_width):.3f}-{max(equal_width):.3f}")
    fig.tight_layout()
    out = Path("figures/04c_sol3_ece_bins.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
