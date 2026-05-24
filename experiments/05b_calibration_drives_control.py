"""Figure: calibration drives the controller.

The controller's decisions are only as good as its belief is honest. We sweep
the generator's overconfidence and compare two controllers: one that trusts
the raw confidence, and one that temperature-scales it first (Post 4c). As
overconfidence grows, the naive controller's inflated belief makes "answer
now" look attractive everywhere, so it stops too early -- gathering less
evidence (lower cost) and scoring lower. The calibrated controller keeps
making the right value-of-information calls.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.agent import NanoAgent, SimulatedWorld, run_agent


def main() -> None:
    betas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    naive_acc, calib_acc, naive_cost, calib_cost = [], [], [], []
    for b in betas:
        world = SimulatedWorld(n_questions=5000, overconfidence=b, seed=0)
        a_n, c_n, _ = run_agent(world, NanoAgent(calibration_T=1.0), seed=1)
        a_c, c_c, _ = run_agent(world, NanoAgent(calibration_T=b), seed=1)
        naive_acc.append(a_n); calib_acc.append(a_c)
        naive_cost.append(c_n); calib_cost.append(c_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot(betas, calib_acc, "o-", color="#55a467", linewidth=2, markersize=7,
             label="calibrated controller (T = β)")
    ax1.plot(betas, naive_acc, "s-", color="#c44e52", linewidth=2, markersize=7,
             label="naive controller (trusts raw confidence)")
    ax1.set_xlabel("generator overconfidence β")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Overconfidence degrades the controller's accuracy", fontsize=11)
    ax1.legend(loc="lower left"); ax1.grid(alpha=0.3); ax1.set_axisbelow(True)

    ax2.plot(betas, calib_cost, "o-", color="#55a467", linewidth=2, markersize=7,
             label="calibrated")
    ax2.plot(betas, naive_cost, "s-", color="#c44e52", linewidth=2, markersize=7,
             label="naive (stops early)")
    ax2.set_xlabel("generator overconfidence β")
    ax2.set_ylabel("mean actions cost")
    ax2.set_title("The overconfident controller is lazier", fontsize=11)
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3); ax2.set_axisbelow(True)

    fig.suptitle(
        "Calibration drives control: an honest belief is the precondition for "
        "good value-of-information decisions.", fontsize=11, y=1.02)
    print(f"  naive_acc={np.round(naive_acc,3)}, calib_acc={np.round(calib_acc,3)}")
    fig.tight_layout()
    out = Path("figures/05b_calibration_drives_control.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
