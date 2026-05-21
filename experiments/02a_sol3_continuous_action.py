"""Solution experiment 2a.3: REINFORCE on a continuous action space.

Setup: a 'continuous bandit' — single state, action a ∈ R, reward
r(a) = exp(-(a - 2.0)^2). Optimum is a = 2.0; reward decays Gaussianly.

Q-learning can't do this without discretization. REINFORCE with a Gaussian
policy can. We additionally learn the *variance* of the policy by
gradient ascent on log-sigma, which lets the policy 'tighten' around the
optimum as it learns.

This is also the foundation of continuous-control policy gradient methods
(DDPG, SAC, PPO).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def reward(a):
    return float(np.exp(-((a - 2.0) ** 2)))


class LearnableGaussian:
    """1D Gaussian with both mu and log_sigma learnable.

    pi(a) = N(mu, sigma^2),  sigma = exp(log_sigma).
    Score functions:
        d log pi(a) / d mu        = (a - mu) / sigma^2
        d log pi(a) / d log_sigma = ((a - mu)^2 / sigma^2) - 1
    """

    def __init__(self, mu=0.0, log_sigma=0.0):
        self.mu = float(mu)
        self.log_sigma = float(log_sigma)

    @property
    def sigma(self):
        return float(np.exp(self.log_sigma))

    def sample(self, rng):
        return float(rng.normal(self.mu, self.sigma))

    def grads(self, a):
        s2 = self.sigma ** 2
        d_mu = (a - self.mu) / s2
        d_log_sigma = ((a - self.mu) ** 2 / s2) - 1.0
        return d_mu, d_log_sigma


def main() -> None:
    rng = np.random.default_rng(0)
    policy = LearnableGaussian(mu=0.0, log_sigma=0.5)  # sigma ≈ 1.65
    n_steps = 800
    batch = 16
    lr = 0.02

    mus = []
    sigmas = []
    rewards = []
    running_mean = 0.0
    n_seen = 0

    for step in range(n_steps):
        # Collect a batch of actions.
        actions = np.array([policy.sample(rng) for _ in range(batch)])
        rs = np.array([reward(a) for a in actions])
        # Update running-mean baseline.
        for r in rs:
            n_seen += 1
            running_mean += (r - running_mean) / n_seen
        advs = rs - running_mean

        # Aggregate gradients.
        g_mu = 0.0; g_logs = 0.0
        for a, A in zip(actions, advs):
            d_mu, d_logs = policy.grads(a)
            g_mu += A * d_mu / batch
            g_logs += A * d_logs / batch

        policy.mu += lr * g_mu
        policy.log_sigma += lr * g_logs
        # Clip log_sigma to avoid degenerate variances.
        policy.log_sigma = max(min(policy.log_sigma, 2.0), -3.0)

        mus.append(policy.mu)
        sigmas.append(policy.sigma)
        rewards.append(float(rs.mean()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: mu and sigma over time.
    ax = axes[0]
    ax.plot(mus, color="#3a7ebf", linewidth=2, label=r"$\mu$ (policy mean)")
    ax.axhline(2.0, color="#3a7ebf", linestyle="--", linewidth=1, alpha=0.5,
                label="optimal mean")
    ax2 = ax.twinx()
    ax2.plot(sigmas, color="#c44e52", linewidth=2, label=r"$\sigma$ (policy std)")
    ax.set_xlabel("gradient step")
    ax.set_ylabel(r"$\mu$", color="#3a7ebf")
    ax2.set_ylabel(r"$\sigma$", color="#c44e52")
    ax.tick_params(axis="y", labelcolor="#3a7ebf")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax.set_title("Policy parameters over training")
    ax.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # Panel 2: average reward.
    ax = axes[1]
    ax.plot(rewards, color="#55a467", linewidth=2)
    ax.axhline(1.0, color="#222", linestyle="--", linewidth=1, alpha=0.5,
                label="optimum r = 1.0")
    ax.set_xlabel("gradient step"); ax.set_ylabel("batch-average reward")
    ax.set_title("Average reward during training")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # Panel 3: policy density at start, middle, end.
    ax = axes[2]
    a_grid = np.linspace(-3, 5, 500)
    snapshots = [0, n_steps // 4, n_steps - 1]
    labels = ["initial", "mid-training", "final"]
    colors = ["#888888", "#dd8452", "#3a7ebf"]
    for k, lab, c in zip(snapshots, labels, colors):
        mu_k = mus[k]; sig_k = sigmas[k]
        pdf = np.exp(-0.5 * ((a_grid - mu_k) / sig_k) ** 2) / (
            np.sqrt(2 * np.pi) * sig_k)
        ax.plot(a_grid, pdf, color=c, linewidth=2,
                 label=f"{lab} (μ={mu_k:.2f}, σ={sig_k:.2f})")
    ax.plot(a_grid, [reward(a) for a in a_grid], color="#c44e52",
            linestyle=":", linewidth=1.5, label="reward $r(a)$")
    ax.axvline(2.0, color="#c44e52", linestyle="--", alpha=0.4)
    ax.set_xlabel("action $a$"); ax.set_ylabel("density / reward")
    ax.set_title("Policy density evolves toward optimum")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle(
        "REINFORCE on a continuous action space — the regime where Q-learning "
        "can't easily play. Policy mean tracks the optimum, std contracts.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path("figures/02a_sol3_continuous_action.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
