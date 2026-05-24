"""Solution experiment 1a.4: Thompson Sampling prior sensitivity.

Run Thompson Sampling with four different Beta priors:
  Beta(0.5, 0.5): Jeffreys, encourages extremes
  Beta(1, 1):     uniform (the default)
  Beta(10, 10):   moderately strong belief around 0.5
  Beta(100, 100): very strong belief around 0.5

Demonstrates that strong miscalibrated priors are expensive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.bandits import BernoulliBandit, ThompsonSampling, run_many


# Update matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"], 
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True 
})


def main() -> None:
    T, n_runs = 5000, 200
    K = 5
    probs = [0.3, 0.5, 0.7, 0.75, 0.8]

    def bf():
        return BernoulliBandit(probs)

    agents = {
        "Beta(0.5, 0.5)": lambda: ThompsonSampling(K, alpha=0.5, beta=0.5),
        "Beta(1, 1) - uniform": lambda: ThompsonSampling(K, alpha=1.0, beta=1.0),
        "Beta(10, 10)": lambda: ThompsonSampling(K, alpha=10.0, beta=10.0),
        "Beta(100, 100)": lambda: ThompsonSampling(K, alpha=100.0, beta=100.0),
    }
    colors = ["#dd8452", "#3a7ebf", "#55a467", "#c44e52"]

    results = run_many(agents, bf, T, n_runs=n_runs, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3),
                              gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: regret curves
    ax = axes[0]
    for (name, r), c in zip(results.items(), colors):
        ax.plot(r, label=name, color=c, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative regret")
    ax.set_title("Thompson Sampling regret under different Beta priors")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # Right: visualize the four priors
    from scipy.stats import beta as beta_dist
    ax = axes[1]
    x = np.linspace(0.001, 0.999, 500)
    priors = [(0.5, 0.5), (1, 1), (10, 10), (100, 100)]
    for (a, b), c, name in zip(priors, colors, results.keys()):
        pdf = beta_dist.pdf(x, a, b)
        ax.plot(x, pdf, color=c, linewidth=2, label=name)
        ax.fill_between(x, pdf, alpha=0.1, color=c)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("prior density")
    ax.set_title("Prior shapes")
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/01a_sol4_prior_sensitivity.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
