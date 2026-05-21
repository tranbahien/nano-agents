"""Solution experiment 1c.4: policy iteration vs value iteration.

PI typically takes far fewer outer iterations than VI, but each PI iteration
involves a full policy-evaluation inner loop. We compare both on a sequence
of gridworld sizes and discount factors.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import GridWorld, policy_iteration, value_iteration


def make_grid(n, gamma_effective=1.0):
    """Plain nxn gridworld with goal at the opposite corner."""
    return GridWorld(rows=n, cols=n, terminals={(n - 1, n - 1): 1.0},
                      step_reward=-0.01, slip=0.1)


def main() -> None:
    sizes = [5, 8, 12, 16, 20]
    gammas_to_test = [0.9, 0.99]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax_idx, gamma in enumerate(gammas_to_test):
        vi_iters, pi_iters, vi_times, pi_times = [], [], [], []
        for n in sizes:
            env = make_grid(n)
            t0 = time.perf_counter()
            _, _, hist_vi = value_iteration(env, gamma=gamma, tol=1e-8)
            vi_times.append(time.perf_counter() - t0)
            vi_iters.append(hist_vi["iters"])

            t0 = time.perf_counter()
            _, _, hist_pi = policy_iteration(env, gamma=gamma, tol=1e-10)
            pi_times.append(time.perf_counter() - t0)
            pi_iters.append(hist_pi["iters"])

        ax = axes[ax_idx]
        ax.plot(sizes, vi_iters, "o-", color="#3a7ebf", linewidth=2,
                 markersize=8, markeredgecolor="white", label="value iteration")
        ax.plot(sizes, pi_iters, "s-", color="#c44e52", linewidth=2,
                 markersize=8, markeredgecolor="white", label="policy iteration")
        ax.set_xlabel("grid size (n × n)")
        ax.set_ylabel("outer iterations to convergence")
        ax.set_title(rf"$\gamma = {gamma}$", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_yscale("log")

        # Annotate with wall-clock times.
        for i, n in enumerate(sizes):
            ax.text(n, vi_iters[i] * 1.4, f"{vi_times[i]*1000:.0f}ms",
                    color="#3a7ebf", fontsize=8, ha="center")
            ax.text(n, pi_iters[i] * 0.65, f"{pi_times[i]*1000:.0f}ms",
                    color="#c44e52", fontsize=8, ha="center")

    fig.suptitle(
        "Policy iteration takes far fewer outer iterations than value iteration. "
        "Wall-clock time is closer because each PI iteration is more expensive.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path("figures/01c_sol4_pi_vs_vi.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
