"""Solution 4c.2: calibration is not discrimination.

A subtle, important point. Temperature scaling is a *monotonic* rescaling of
the logits, so it leaves the *ranking* of items by confidence unchanged --
and therefore leaves unchanged everything that depends only on ranking: the
ROC/AUC, and the best achievable accuracy of any single-threshold selective
policy (the risk-coverage curve). We show the risk-coverage curve (accuracy
on answered items vs the fraction answered) is identical before and after
temperature scaling, even though ECE collapses. Calibration fixes *what the
numbers mean*, not *how well they separate right from wrong*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import (
    CalibrationDataset,
    apply_temperature,
    ece,
)
from nano_agents.calibration.model import sigmoid


def risk_coverage(confidence, correct, n=40):
    """Accuracy on the most-confident fraction (coverage) of items."""
    order = np.argsort(-confidence)
    correct = np.asarray(correct, float)[order]
    covs = np.linspace(0.05, 1.0, n)
    accs = [correct[: max(1, int(c * len(correct)))].mean() for c in covs]
    return covs, np.array(accs)


def main() -> None:
    ds = CalibrationDataset(n_items=16000, sharpness=2.5, seed=0)
    raw = sigmoid(ds.report_logit)
    fixed = apply_temperature(ds.report_logit, 2.44)

    cov_r, acc_r = risk_coverage(raw, ds.correct)
    cov_f, acc_f = risk_coverage(fixed, ds.correct)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(cov_r, acc_r, "o-", color="#c44e52", linewidth=2, markersize=5,
            label=f"before scaling (ECE={ece(raw, ds.correct):.3f})")
    ax.plot(cov_f, acc_f, "s", color="#55a467", markersize=7, fillstyle="none",
            markeredgewidth=1.8,
            label=f"after scaling (ECE={ece(fixed, ds.correct):.3f})")
    ax.axhline(ds.accuracy, color="#888", linestyle="--", linewidth=1.5,
               label=f"answer-all accuracy {ds.accuracy:.2f}")

    ax.set_xlabel("coverage (fraction of items answered, most-confident first)")
    ax.set_ylabel("accuracy on answered items")
    ax.set_title(
        "Calibration ≠ discrimination. Temperature scaling collapses ECE but leaves\n"
        "the risk–coverage curve (and AUC) untouched — the ranking is identical.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    print(f"  curves identical: {np.allclose(acc_r, acc_f)}")
    fig.tight_layout()
    out = Path("figures/04c_sol2_calibration_vs_discrimination.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
