"""Figure: a chain of agents is only as strong as the product of its links.

In a sequential pipeline (planner -> researcher -> writer -> checker), every
stage must succeed for the whole task to succeed, so reliability is p**L.
Even highly reliable agents compound into an unreliable pipeline as it grows
-- the same error-compounding cliff as long-horizon tool use (Post 3b). More
stages is not more capability unless each stage is near-perfect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.multiagent import pipeline_accuracy


def main() -> None:
    stages = np.arange(1, 13)
    configs = [
        (0.95, "#55a467", "per-stage 0.95"),
        (0.90, "#3a7ebf", "per-stage 0.90"),
        (0.80, "#dd8452", "per-stage 0.80"),
        (0.70, "#c44e52", "per-stage 0.70"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for p, color, label in configs:
        ys = [pipeline_accuracy(p, L) for L in stages]
        ax.plot(stages, ys, "o-", color=color, linewidth=2, markersize=6,
                label=label)
    ax.axhline(0.5, color="#bbb", linestyle=":", linewidth=1,
               label="coin-flip reliability")

    ax.set_xlabel("number of pipeline stages (agents in series)")
    ax.set_ylabel("end-to-end success probability")
    ax.set_title(
        "Pipelines compound errors: reliability = p^L. Even 0.95-per-stage\n"
        "agents fall below 0.6 by a dozen stages — chains amplify weakness.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(stages); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = Path("figures/05a_pipeline_compounding.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
