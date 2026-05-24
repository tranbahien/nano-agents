"""Solution experiment 1a.2: non-stationary rewards.

Arm means drift sinusoidally over time. Demonstrates that all three
stationary algorithms break, but in different ways.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.bandits import (
    DriftingBernoulliBandit,
    EpsilonGreedy,
    ThompsonSampling,
    UCB1,
)


# Update matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"], 
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True 
})



def run_track(agent, bandit, T):
    regret = np.zeros(T)
    actions = np.zeros(T, dtype=int)
    best_arms = np.zeros(T, dtype=int)
    cum = 0.0
    for t in range(T):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        cum += bandit.regret_of(k)
        regret[t] = cum
        actions[t] = k
        # best arm BEFORE the pull-advancing the timer is messy;
        # we just use the current means after pull-already advanced t-1 -> t.
        best_arms[t] = int(np.argmax(bandit.probs))
    return regret, actions, best_arms


def main() -> None:
    T = 8000
    K = 3
    np.random.seed(0)

    def make_bandit():
        # Slow-drift means with substantial amplitude.
        return DriftingBernoulliBandit(
            base=[0.5, 0.5, 0.5], amplitude=0.35, period=2500.0, seed=42)

    agents = {
        r"$\epsilon$-greedy (0.1)": lambda: EpsilonGreedy(K, 0.1),
        "UCB1": lambda: UCB1(K),
        "Thompson": lambda: ThompsonSampling(K),
    }

    # Run once each (no averaging) so we can show the realized choice trajectory.
    results = {}
    for name, ctor in agents.items():
        bandit = make_bandit()
        np.random.seed(7)  # same random rewards for fair comparison
        results[name] = run_track(ctor(), bandit, T)

    # We also need to know the true best arm over time. Use a fresh probe bandit.
    probe = make_bandit()
    best_over_time = np.zeros(T, dtype=int)
    means_over_time = np.zeros((T, K))
    for t in range(T):
        probe.t = t
        m = probe.probs
        best_over_time[t] = int(np.argmax(m))
        means_over_time[t] = m

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5),
                             gridspec_kw={"height_ratios": [1.5, 1.0]},
                             sharex=True)

    # Top: regret curves
    ax = axes[0]
    for name, (regret, _, _) in results.items():
        ax.plot(regret, label=name, linewidth=2)
    ax.set_ylabel("cumulative regret")
    ax.set_title("Non-stationary Bernoulli bandit: sinusoidally drifting arm means")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # Bottom: true arm means over time + which arm each algorithm picks (running mode)
    ax = axes[1]
    colors_arm = ["#888888", "#3a7ebf", "#55a467"]
    for k in range(K):
        ax.plot(means_over_time[:, k], color=colors_arm[k], linewidth=1.0,
                linestyle="--", alpha=0.7, label=fr"Arm {k+1} true $\mu$" if True else None)
    ax.set_ylabel(r"true $\mu$ over time")
    ax.set_xlabel("t")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0.05, 0.95)

    fig.tight_layout()
    out = Path("figures/01a_sol2_nonstationary.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
