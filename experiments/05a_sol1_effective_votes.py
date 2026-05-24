"""Solution 5a.1: how correlation collapses the effective committee size.

A committee of N equicorrelated agents carries the information of only

    N_eff = N / (1 + (N - 1) * rho)

independent ones -- the classic effective-sample-size formula for correlated
estimators. As rho rises, N_eff collapses toward 1: forty correlated agents
can be worth barely more than one. This is the quantitative core of "diversity
is the resource," and it is the same variance-reduction arithmetic behind
self-consistency (Post 3a) and ensembles (Post 4c).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def n_eff(N, rho):
    return N / (1.0 + (N - 1) * rho)


def main() -> None:
    rhos = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for N, color in [(40, "#c44e52"), (20, "#dd8452"), (10, "#3a7ebf"),
                     (5, "#55a467")]:
        ax.plot(rhos, n_eff(N, rhos), "-", color=color, linewidth=2,
                label=f"N = {N} agents")

    ax.set_xlabel("error correlation ρ")
    ax.set_ylabel("effective number of independent votes  $N_{eff}$")
    ax.set_title(
        "Correlation collapses the committee. $N_{eff}=N/(1+(N-1)ρ)$:\n"
        "forty correlated agents can be worth barely more than one.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_yscale("log"); ax.set_ylim(0.9, 50)

    print(f"  N=40: N_eff(0.1)={n_eff(40,0.1):.1f}, N_eff(0.5)={n_eff(40,0.5):.1f}")
    fig.tight_layout()
    out = Path("figures/05a_sol1_effective_votes.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
