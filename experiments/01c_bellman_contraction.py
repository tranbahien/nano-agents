"""Figure: Bellman operator as a contraction mapping.

The Bellman optimality operator T satisfies ||T V - T V'||_inf <= gamma * ||V - V'||_inf.
Banach's fixed-point theorem then guarantees that iterating T produces a
unique fixed point V*, and the convergence is geometric with rate gamma.

We verify this empirically by tracking ||V_k - V*||_inf over iterations
for several values of gamma. On a log scale these should be straight lines
with slope log(gamma).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration


def main() -> None:
    gammas = [0.5, 0.8, 0.95, 0.99]
    colors = ["#888888", "#dd8452", "#3a7ebf", "#c44e52"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for gamma, color in zip(gammas, colors):
        env = TwoGoalGridWorld(slip=0.1, step_reward=-0.04)
        V_star, _, hist = value_iteration(env, gamma=gamma, tol=1e-12,
                                            max_iters=600)
        snapshots = hist["V_snapshots"]
        errors = [np.max(np.abs(V - V_star)) for V in snapshots]
        # Clip tiny errors that go below double precision noise.
        errors = np.maximum(errors, 1e-14)
        ax.semilogy(errors, color=color, linewidth=2,
                     label=rf"$\gamma = {gamma}$")
        # Overlay theoretical bound: ||V_k - V*||_inf <= C * gamma^k.
        if errors[0] > 1e-12:
            ks = np.arange(len(errors))
            theory = errors[0] * gamma ** ks
            ax.semilogy(ks, theory, color=color, linestyle=":",
                         linewidth=1, alpha=0.6)

    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\| V_k - V^\star \|_\infty$  (log scale)")
    ax.set_title(
        "Bellman operator is a γ-contraction: error shrinks geometrically.\n"
        "Solid = measured.   Dotted = theoretical $C \\cdot \\gamma^k$.",
        fontsize=12,
    )
    ax.legend(loc="upper right")
    ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(1e-12, 100)

    fig.tight_layout()
    out = Path("figures/01c_bellman_contraction.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
