"""Figure: the signature of overconfidence in the confidence histogram.

Overconfident models pile probability mass near 1.0 -- they are sure about
almost everything. We show the confidence distribution and the gap between
mean confidence and actual accuracy for a calibrated vs an overconfident
model with identical underlying skill. The overconfident model's average
confidence sits far above its accuracy; the calibrated model's matches.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import CalibrationDataset


def panel(ax, beta, color, title):
    ds = CalibrationDataset(n_items=12000, sharpness=beta, seed=0)
    ax.hist(ds.confidence, bins=30, range=(0, 1), color=color, alpha=0.65,
            edgecolor="white", linewidth=0.5)
    acc = ds.accuracy
    mean_conf = ds.confidence.mean()
    ax.axvline(acc, color="#222", linestyle="-", linewidth=2,
               label=f"accuracy = {acc:.2f}")
    ax.axvline(mean_conf, color=color, linestyle="--", linewidth=2,
               label=f"mean confidence = {mean_conf:.2f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("reported confidence")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    gap = mean_conf - acc
    ax.annotate(f"over-confidence\ngap = {gap:+.2f}", (0.5, 0.78),
                xycoords="axes fraction", fontsize=10, ha="center",
                color=color)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    panel(axes[0], 1.0, "#55a467", "Calibrated (β=1.0)")
    panel(axes[1], 2.5, "#c44e52", "Overconfident (β=2.5)")
    axes[0].set_ylabel("count")
    fig.suptitle(
        "Overconfidence piles mass near 1.0: the model is sure about almost "
        "everything,\nso mean confidence drifts above accuracy.",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path("figures/04c_confidence_histograms.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
