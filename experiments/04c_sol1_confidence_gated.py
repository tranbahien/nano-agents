"""Solution 4c.1: the agentic payoff -- confidence-gated decisions.

An agent that "retrieves / reflects / defers when unsure" needs a confidence
threshold that means what it says. We gate on a *fixed, semantically chosen*
rule -- keep the model's answer when confidence >= threshold, otherwise take
a fallback action that succeeds with probability `fb`. We sweep the threshold
for the raw overconfident model versus the same model after temperature
scaling. At face-value thresholds (e.g. 0.8) the overconfident model keeps
too many wrong answers; recalibration routes them to the better fallback,
lifting accuracy and making the operating point predictable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import (
    CalibrationDataset,
    apply_temperature,
    gated_decision_accuracy,
    temperature_scale,
)
from nano_agents.calibration.model import sigmoid


def main() -> None:
    fb = 0.88
    ds = CalibrationDataset(n_items=16000, sharpness=2.5, seed=0)
    (cal_logit, cal_y), (test_logit, test_y) = ds.split(frac=0.5, seed=1)
    T = temperature_scale(cal_logit, cal_y)

    raw_conf = sigmoid(test_logit)
    fixed_conf = apply_temperature(test_logit, T)

    thr = np.linspace(0.5, 0.98, 40)
    raw_acc = [gated_decision_accuracy(raw_conf, test_y, t, fb)[0] for t in thr]
    fix_acc = [gated_decision_accuracy(fixed_conf, test_y, t, fb)[0] for t in thr]
    base = test_y.mean()

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(thr, raw_acc, "o-", color="#c44e52", linewidth=2, markersize=5,
            label="overconfident (raw confidence)")
    ax.plot(thr, fix_acc, "s-", color="#55a467", linewidth=2, markersize=5,
            label="after temperature scaling")
    ax.axhline(base, color="#888", linestyle="--", linewidth=1.5,
               label=f"never gate (answer all): {base:.2f}")
    ax.axhline(fb, color="#bbb", linestyle=":", linewidth=1.5,
               label=f"always fall back: {fb:.2f}")
    ax.axvline(0.8, color="#3a7ebf", linestyle=":", linewidth=1.5,
               label="face-value threshold 0.80")

    ax.set_xlabel("confidence threshold for keeping the model's answer")
    ax.set_ylabel("gated-policy accuracy")
    ax.set_title(
        "Calibration makes a face-value threshold mean what it says.\n"
        "At 0.80 the overconfident model keeps wrong answers; recalibration defers them.",
        fontsize=11)
    ax.legend(loc="lower center", fontsize=9); ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    i80 = int(np.argmin(np.abs(thr - 0.8)))
    print(f"  at tau=0.80: raw={raw_acc[i80]:.3f}, temp-scaled={fix_acc[i80]:.3f}")
    fig.tight_layout()
    out = Path("figures/04c_sol1_confidence_gated.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
