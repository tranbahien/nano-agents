"""Figure: how bad overconfidence gets, and how well temperature scaling fixes it.

Sweep the sharpness beta from underconfident (<1) through calibrated (1) to
badly overconfident (>1). Raw ECE grows steeply with miscalibration in either
direction; after fitting a temperature on held-out data, ECE stays near zero
across the whole range -- a single parameter absorbs the entire logit
rescaling, because that is exactly the form the miscalibration takes here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import (
    CalibrationDataset,
    apply_temperature,
    ece,
    temperature_scale,
)
from nano_agents.calibration.model import sigmoid


def main() -> None:
    betas = np.linspace(0.4, 4.0, 19)
    raw, fixed = [], []
    for beta in betas:
        ds = CalibrationDataset(n_items=16000, sharpness=float(beta), seed=0)
        (cal_logit, cal_y), (test_logit, test_y) = ds.split(frac=0.5, seed=1)
        raw.append(ece(sigmoid(test_logit), test_y, n_bins=15))
        T = temperature_scale(cal_logit, cal_y)
        fixed.append(ece(apply_temperature(test_logit, T), test_y, n_bins=15))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(betas, raw, "o-", color="#c44e52", linewidth=2, markersize=6,
            label="raw ECE")
    ax.plot(betas, fixed, "s-", color="#55a467", linewidth=2, markersize=6,
            label="after temperature scaling")
    ax.axvline(1.0, color="#888", linestyle=":", linewidth=2,
               label="calibrated (β=1)")

    ax.set_xlabel("sharpness β  (← underconfident · overconfident →)")
    ax.set_ylabel("expected calibration error (ECE)")
    ax.set_title(
        "Miscalibration grows steeply with β; one fitted temperature\n"
        "absorbs it across the whole range.",
        fontsize=11)
    ax.legend(loc="upper center"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/04c_overconfidence_growth.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
