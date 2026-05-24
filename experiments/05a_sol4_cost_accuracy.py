"""Solution 5a.4: the cost of a committee, and when to stop adding agents.

A committee of N agents costs N times one agent, but the accuracy gain
saturates (Condorcet's returns shrink with N). We plot majority-vote accuracy
against committee size for an independent and a correlated committee, and the
*marginal* accuracy per added agent. The marginal gain collapses fast --
beyond a handful of agents you are paying linearly for almost nothing,
especially once errors are correlated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.multiagent import Committee, committee_accuracy


def main() -> None:
    p = 0.6
    sizes = [1, 2, 3, 5, 7, 9, 13, 17, 25, 35]
    indep = [committee_accuracy(Committee(n, 2, p, 0.0, 0), 6000, 1)
             for n in sizes]
    corr = [committee_accuracy(Committee(n, 2, p, 0.3, 0), 6000, 1)
            for n in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    ax1.plot(sizes, indep, "o-", color="#55a467", linewidth=2, markersize=6,
             label="independent (ρ=0)")
    ax1.plot(sizes, corr, "s-", color="#c44e52", linewidth=2, markersize=6,
             label="correlated (ρ=0.3)")
    ax1.set_xlabel("committee size (= cost, linear)")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Accuracy saturates as cost grows linearly", fontsize=11)
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3); ax1.set_axisbelow(True)

    # marginal accuracy gain per added agent (per unit cost)
    marg_i = np.diff(indep) / np.diff(sizes)
    marg_c = np.diff(corr) / np.diff(sizes)
    mids = (np.array(sizes[:-1]) + np.array(sizes[1:])) / 2
    ax2.plot(mids, marg_i, "o-", color="#55a467", linewidth=2, markersize=6,
             label="independent")
    ax2.plot(mids, marg_c, "s-", color="#c44e52", linewidth=2, markersize=6,
             label="correlated")
    ax2.axhline(0, color="#888", linewidth=1)
    ax2.set_xlabel("committee size")
    ax2.set_ylabel("marginal accuracy per added agent")
    ax2.set_title("Marginal return collapses fast", fontsize=11)
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3); ax2.set_axisbelow(True)

    fig.suptitle(
        "Cost is linear, benefit saturates: a few diverse agents capture most "
        "of the gain.", fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path("figures/05a_sol4_cost_accuracy.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
