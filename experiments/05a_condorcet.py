"""Figure: the Condorcet jury theorem.

Majority-vote accuracy of a committee of independent agents, each correct
with probability p, as the committee grows. The theorem's knife-edge: if each
agent is better than chance (p > 1/2), the majority converges to certainty;
if worse (p < 1/2), it converges to certain failure; at exactly 1/2 it stays
at 1/2. Independence is doing all the work -- the analytic claim that
motivates the rest of the post.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.multiagent import condorcet_accuracy


def main() -> None:
    sizes = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    configs = [
        (0.65, "#55a467", "p = 0.65 (competent)"),
        (0.55, "#3a7ebf", "p = 0.55 (just above chance)"),
        (0.50, "#888888", "p = 0.50 (chance)"),
        (0.45, "#c44e52", "p = 0.45 (worse than chance)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for p, color, label in configs:
        ys = [condorcet_accuracy(p, n) for n in sizes]
        ax.plot(sizes, ys, "o-", color=color, linewidth=2, markersize=6,
                label=label)
    ax.axhline(0.5, color="#bbb", linestyle=":", linewidth=1)

    ax.set_xlabel("committee size (independent agents)")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title(
        "The Condorcet jury theorem. Independent agents above chance vote\n"
        "their way to certainty; below chance, to certain failure.",
        fontsize=11)
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(sizes); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = Path("figures/05a_condorcet.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
