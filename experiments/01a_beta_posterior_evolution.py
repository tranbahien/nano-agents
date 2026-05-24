"""Figure: Beta posterior evolution under Thompson Sampling.

Shows how the Beta posterior tightens and separates over time, illustrating
why uncertainty *automatically* drives exploration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from nano_agents.bandits import BernoulliBandit, ThompsonSampling


# Update matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"], 
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True 
})



def run_to_time(T_targets, true_mus, seed=0):
    """Run Thompson Sampling and snapshot (alpha, beta) at each T_target."""
    np.random.seed(seed)
    bandit = BernoulliBandit(true_mus)
    agent = ThompsonSampling(bandit.K)

    snapshots = {}
    T_max = max(T_targets)
    for t in range(1, T_max + 1):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        if t in T_targets:
            snapshots[t] = (agent.alpha.copy(), agent.beta.copy())
    return snapshots


def main() -> None:
    true_mus = [0.3, 0.6, 0.7]
    K = len(true_mus)
    T_targets = [10, 100, 1000, 5000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    snaps = run_to_time(T_targets, true_mus, seed=0)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True)
    x = np.linspace(0, 1, 500)

    for ax, T in zip(axes, T_targets):
        alpha_vec, beta_vec = snaps[T]
        for k in range(K):
            pdf = beta.pdf(x, alpha_vec[k], beta_vec[k])
            n_k = int(alpha_vec[k] + beta_vec[k] - 2)  # subtract uniform prior
            ax.plot(x, pdf, color=colors[k], linewidth=2,
                    label=f"Arm {k+1} ($\\mu_k={true_mus[k]}$, $n_k={n_k}$)")
            ax.fill_between(x, pdf, alpha=0.15, color=colors[k])
            # True mean as vertical line
            ax.axvline(true_mus[k], color=colors[k], linestyle=":", alpha=0.7, linewidth=1)
        ax.set_title(f"t = {T}", fontsize=12)
        ax.set_xlabel(r"$\mu$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("posterior density")
    fig.suptitle(
        "Beta posteriors under Thompson Sampling. "
        "Uncertainty (width) shrinks where it matters and stays wider where it doesn't.",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path("figures/01a_beta_posterior_evolution.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
