"""Figure: the reliability diagram.

A model is calibrated if, among predictions it makes with confidence p, the
fraction correct is p -- i.e. the reliability curve lies on the diagonal. We
plot three models with the same underlying accuracy: calibrated (beta=1),
overconfident (beta=2.5, curve sags below the diagonal -- confidence exceeds
accuracy), and underconfident (beta=0.5, curve bows above). ECE is the
shaded average gap to the diagonal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import CalibrationDataset, ece, reliability_curve


def main() -> None:
    configs = [
        (1.0, "#55a467", "calibrated (β=1.0)"),
        (2.5, "#c44e52", "overconfident (β=2.5)"),
        (0.5, "#3a7ebf", "underconfident (β=0.5)"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=2,
            label="perfect calibration")

    for beta, color, label in configs:
        ds = CalibrationDataset(n_items=12000, sharpness=beta, seed=0)
        conf, acc, count = reliability_curve(ds.confidence, ds.correct, n_bins=12)
        e = ece(ds.confidence, ds.correct, n_bins=12)
        ax.plot(conf, acc, "o-", color=color, linewidth=2, markersize=6,
                label=f"{label}  ·  ECE={e:.3f}")

    # shade the calibration gap for the overconfident model
    ds = CalibrationDataset(n_items=12000, sharpness=2.5, seed=0)
    conf, acc, _ = reliability_curve(ds.confidence, ds.correct, n_bins=12)
    ax.fill_between(conf, acc, conf, color="#c44e52", alpha=0.10)

    ax.set_xlabel("reported confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(
        "The reliability diagram. Below the diagonal = overconfident\n"
        "(confidence exceeds accuracy); above = underconfident.",
        fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")

    fig.tight_layout()
    out = Path("figures/04c_reliability_diagram.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
