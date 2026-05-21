"""Solution experiment 1b.4: LinUCB exploration parameter sweep.

Sweep alpha over a wide range. Plot final cumulative regret as a function
of alpha. There's typically a U-shape: too little exploration is bad (the
algorithm can lock onto suboptimal arms), too much is also bad (wastes pulls).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.contextual import LinearContextualBandit, LinUCB, run_contextual


def main() -> None:
    K, d = 5, 8
    T, n_runs = 1500, 20

    alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    final_regrets = []
    final_regret_stds = []

    for alpha in alphas:
        print(f"  α = {alpha}...")
        finals = []
        for run_idx in range(n_runs):
            bandit = LinearContextualBandit(K=K, d=d, sigma=0.1, seed=run_idx)
            agent = LinUCB(K, d, alpha=alpha)
            cum = run_contextual(agent, bandit, T)
            finals.append(cum[-1])
        final_regrets.append(np.mean(finals))
        final_regret_stds.append(np.std(finals) / np.sqrt(n_runs))  # SEM

    final_regrets = np.array(final_regrets)
    final_regret_stds = np.array(final_regret_stds)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(alphas, final_regrets, yerr=final_regret_stds,
                fmt="o-", color="#3a7ebf", linewidth=2, markersize=9,
                capsize=4, markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"exploration parameter $\alpha$")
    ax.set_ylabel(f"cumulative regret at T = {T}")
    ax.set_title(f"LinUCB sensitivity to exploration parameter\n"
                 f"K={K}, d={d}, mean ± SEM over {n_runs} runs")
    ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate the minimum
    best_idx = int(np.argmin(final_regrets))
    ax.annotate(rf"best: $\alpha = {alphas[best_idx]}$",
                xy=(alphas[best_idx], final_regrets[best_idx]),
                xytext=(alphas[best_idx] * 0.6, final_regrets[best_idx] + 15),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#222"))

    fig.tight_layout()
    out = Path("figures/01b_sol4_alpha_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
