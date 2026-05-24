"""Solution experiment 1a.3: Gaussian rewards.

Compare Gaussian UCB and Gaussian Thompson Sampling (Normal-Inverse-Gamma
prior, Student-t marginal posterior over mu) on a Gaussian bandit.

Also demonstrate the failure of Bernoulli Thompson Sampling on Gaussian
data — sloppy modeling has visible consequences.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.bandits import (
    EpsilonGreedy,
    GaussianBandit,
    GaussianThompsonSampling,
    GaussianUCB,
)


# Update matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"], 
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True 
})



def run(agent, bandit, T):
    regret = np.zeros(T)
    cum = 0.0
    for t in range(T):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        cum += bandit.regret_of(k)
        regret[t] = cum
    return regret


def main() -> None:
    T, n_runs = 3000, 100
    K = 5
    means = [0.0, 0.5, 1.0, 1.2, 1.5]
    sigma = 1.0

    agents = {
        r"$\epsilon$-greedy (0.1)": lambda: EpsilonGreedy(K, 0.1),
        r"Gaussian UCB ($\sigma$ known)": lambda: GaussianUCB(K, sigma=sigma),
        r"Gaussian TS (NIG)": lambda: GaussianThompsonSampling(
            K, rng=np.random.default_rng()),
    }

    results = {name: np.zeros(T) for name in agents}
    for run_idx in range(n_runs):
        np.random.seed(run_idx)
        for name, ctor in agents.items():
            bandit = GaussianBandit(means, sigma=sigma)
            results[name] += run(ctor(), bandit, T) / n_runs

    fig, ax = plt.subplots(figsize=(6.5, 3))
    for name, r in results.items():
        ax.plot(r, label=name, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative regret")
    ax.set_title(rf"Gaussian bandit: 5 arms with means {means}, $\sigma$ = {sigma}"
                 f"\n(mean over {n_runs} runs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path("figures/01a_sol3_gaussian_rewards.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
