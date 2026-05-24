"""Figure: value-of-information stopping produces adaptive compute.

The controller keeps acting only while an action's expected gain beats its
cost, so easy questions (high belief) stop after one step and hard ones draw
several. We bin questions by difficulty and plot the mean number of actions.
The agent spends compute where it is needed -- without being told to.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.agent import NanoAgent, SimulatedWorld, run_agent


def main() -> None:
    world = SimulatedWorld(n_questions=8000, overconfidence=1.0, seed=0)
    _, _, traces = run_agent(world, NanoAgent(calibration_T=1.0), seed=1)
    nact = np.array([len(t.actions) for t in traces])

    # bin by per-question base accuracy (lower accuracy = harder)
    acc = np.array([world._gen_accuracy(q) for q in range(world.n_questions)])
    bins = np.linspace(acc.min(), acc.max(), 9)
    idx = np.digitize(acc, bins)
    centers, means = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 20:
            centers.append(acc[m].mean()); means.append(nact[m].mean())

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(centers, means, "o-", color="#3a7ebf", linewidth=2, markersize=7)
    ax.invert_xaxis()   # hard (low accuracy) on the right -> read left-to-right easy->hard? invert so hard right
    ax.set_xlabel("base-model accuracy on the question  (harder →)")
    ax.set_ylabel("mean actions taken by the controller")
    ax.set_title(
        "Adaptive compute, for free. Value-of-information stopping spends\n"
        "more actions on hard questions and fewer on easy ones.",
        fontsize=11)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    print(f"  actions range: {min(means):.2f}-{max(means):.2f}")
    fig.tight_layout()
    out = Path("figures/05b_adaptive_compute.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
