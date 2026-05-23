"""Figure: temperature scaling snaps the reliability curve onto the diagonal.

We fit a single temperature T on held-out data (minimizing NLL) and divide
the logits by it. The overconfident curve, which sagged below the diagonal,
moves onto it. One parameter, fit post-hoc, with no retraining.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import (
    CalibrationDataset,
    apply_temperature,
    ece,
    reliability_curve,
    temperature_scale,
)
from nano_agents.calibration.model import sigmoid


def main() -> None:
    beta = 2.5
    ds = CalibrationDataset(n_items=16000, sharpness=beta, seed=0)
    (cal_logit, cal_y), (test_logit, test_y) = ds.split(frac=0.5, seed=1)

    T = temperature_scale(cal_logit, cal_y)
    raw_conf = sigmoid(test_logit)
    fixed_conf = apply_temperature(test_logit, T)

    conf_raw, acc_raw, _ = reliability_curve(raw_conf, test_y, n_bins=12)
    conf_fix, acc_fix, _ = reliability_curve(fixed_conf, test_y, n_bins=12)
    ece_raw = ece(raw_conf, test_y, n_bins=12)
    ece_fix = ece(fixed_conf, test_y, n_bins=12)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=2,
            label="perfect calibration")
    ax.plot(conf_raw, acc_raw, "o-", color="#c44e52", linewidth=2, markersize=6,
            label=f"before (overconfident) · ECE={ece_raw:.3f}")
    ax.plot(conf_fix, acc_fix, "s-", color="#55a467", linewidth=2, markersize=6,
            label=f"after T={T:.2f} scaling · ECE={ece_fix:.3f}")
    ax.fill_between(conf_raw, acc_raw, conf_raw, color="#c44e52", alpha=0.08)

    ax.set_xlabel("reported confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(
        f"Temperature scaling: one parameter (T≈{T:.1f}≈β) fit on held-out data\n"
        "pulls the overconfident curve back onto the diagonal.",
        fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")

    print(f"  fitted T={T:.3f} (true beta={beta}); ECE {ece_raw:.3f} -> {ece_fix:.3f}")
    fig.tight_layout()
    out = Path("figures/04c_temperature_scaling.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
