"""Figure: correlated errors cap what voting can achieve.

The Condorcet promise -- accuracy to 1 as the committee grows -- holds only
for *independent* errors. We sweep committee size for several lockstep
correlations rho (the fraction of items the agents get wrong in unison,
because they share blind spots). At rho = 0 voting climbs toward 1; at rho >
0 it plateaus at 1 - rho*(1-p), no matter how many agents you add. Diversity
is the resource; correlation is what destroys the committee's edge.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.multiagent import (
    Committee,
    committee_accuracy,
    correlated_vote_ceiling,
)


def main() -> None:
    p = 0.6
    sizes = [1, 3, 5, 9, 15, 25, 41]
    rhos = [
        (0.0, "#55a467", "ρ = 0.0 (independent)"),
        (0.3, "#3a7ebf", "ρ = 0.3"),
        (0.6, "#dd8452", "ρ = 0.6"),
        (0.9, "#c44e52", "ρ = 0.9 (near-clones)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for rho, color, label in rhos:
        ys = [committee_accuracy(
            Committee(n_agents=n, n_answers=2, accuracy=p, correlation=rho,
                      seed=0), n_items=6000, seed=1) for n in sizes]
        ax.plot(sizes, ys, "o-", color=color, linewidth=2, markersize=6,
                label=label)
        ceiling = correlated_vote_ceiling(p, rho)
        if rho > 0:
            ax.axhline(ceiling, color=color, linestyle=":", linewidth=1.2,
                       alpha=0.7)

    ax.set_xlabel("committee size")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title(
        "Correlated errors cap voting (p=0.6). Independent committees approach 1;\n"
        "correlated ones plateau at 1 − ρ(1−p) (dotted), however many you add.",
        fontsize=11)
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(sizes); ax.set_ylim(0.5, 1.02)

    fig.tight_layout()
    out = Path("figures/05a_correlation_kills_voting.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
