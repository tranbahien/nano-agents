"""Solution 4c.4: ensembles vs temperature scaling.

Temperature scaling needs a held-out set and only recalibrates. Averaging an
*ensemble* of independent predictors softens overconfident probabilities for
free: the mean of several extreme-but-noisy sigmoid outputs is pulled toward
the middle, reducing overconfidence without fitting anything -- and, in
practice (though not in this single-outcome toy), ensembles also raise
accuracy. We grow the ensemble size and watch ECE fall toward the
temperature-scaled floor.
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
    beta = 2.5
    base = CalibrationDataset(n_items=12000, sharpness=beta, seed=0)
    latent = base.latent_logit
    correct = base.correct
    rng = np.random.default_rng(7)

    sizes = [1, 2, 3, 5, 8, 12, 20]
    ens_ece = []
    # each member reports sigmoid(beta * (latent + noise)); ensemble averages probs
    member_noise = rng.normal(0, 0.8, size=(max(sizes), len(latent)))
    for m in sizes:
        probs = np.mean(
            [sigmoid(beta * (latent + member_noise[k])) for k in range(m)], axis=0
        )
        ens_ece.append(ece(probs, correct, n_bins=15))

    # temperature-scaled single model (held-out fit) as a reference floor
    (cl, cy), (tl, ty) = base.split(frac=0.5, seed=1)
    T = temperature_scale(cl, cy)
    ts_ece = ece(apply_temperature(base.report_logit, T), correct, n_bins=15)
    raw_ece = ece(base.confidence, correct, n_bins=15)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(sizes, ens_ece, "o-", color="#3a7ebf", linewidth=2, markersize=6,
            label="ensemble (average of members)")
    ax.axhline(raw_ece, color="#c44e52", linestyle="--", linewidth=1.5,
               label=f"single overconfident model ({raw_ece:.3f})")
    ax.axhline(ts_ece, color="#55a467", linestyle="--", linewidth=1.5,
               label=f"temperature scaling floor ({ts_ece:.3f})")

    ax.set_xlabel("ensemble size")
    ax.set_ylabel("expected calibration error (ECE)")
    ax.set_title(
        "Ensembling softens overconfidence for free (no held-out fit).\n"
        "It complements temperature scaling, which needs a calibration set.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(sizes)

    print(f"  raw {raw_ece:.3f} -> ensemble-20 {ens_ece[-1]:.3f}; temp floor {ts_ece:.3f}")
    fig.tight_layout()
    out = Path("figures/04c_sol4_ensembles.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
