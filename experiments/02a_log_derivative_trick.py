"""Figure: visualizing the policy gradient on a 1D continuous bandit.

Setup: action a ∈ R. Policy is Gaussian: pi(a; mu) = N(a; mu, sigma^2).
Reward: r(a) = -(a - 2)^2 (peaks at a = 2).

The policy gradient is
    grad_mu E[r(a)] = E_{a ~ pi}[r(a) * (a - mu) / sigma^2]
                    = ∫ pi(a; mu) * r(a) * (a - mu)/sigma^2 da

We plot:
  - Left: the policy density and reward function.
  - Center: the integrand of the gradient (the "weighted score").
  - Right: the resulting trajectory of mu under gradient ascent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.policy_gradient import GaussianPolicy


def reward(a):
    """Reward: 1 - (a - 2)^2. Positive in (1, 3), negative outside."""
    return 1.0 - (a - 2.0) ** 2


def main() -> None:
    sigma = 1.0
    a_grid = np.linspace(-3, 5, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    # ----- Panel 1: policy and reward -----
    ax = axes[0]
    mu = 0.0
    p = np.exp(-0.5 * ((a_grid - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    r = reward(a_grid)
    ax.plot(a_grid, p, color="#3a7ebf", linewidth=2, label=rf"$\pi(a; \mu={mu})$")
    ax2 = ax.twinx()
    ax2.plot(a_grid, r, color="#c44e52", linewidth=2,
              label=r"$r(a) = 1 - (a-2)^2$")
    ax.set_xlabel("action $a$")
    ax.set_ylabel("policy density", color="#3a7ebf")
    ax2.set_ylabel("reward", color="#c44e52")
    ax.tick_params(axis="y", labelcolor="#3a7ebf")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax.set_title("Policy and reward")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # ----- Panel 2: gradient integrand -----
    ax = axes[1]
    score = (a_grid - mu) / sigma ** 2          # d/dmu log pi(a; mu)
    integrand = p * r * score
    ax.plot(a_grid, integrand, color="#222", linewidth=2)
    ax.fill_between(a_grid, integrand, where=integrand > 0,
                     color="#55a467", alpha=0.4, label="positive contribution")
    ax.fill_between(a_grid, integrand, where=integrand < 0,
                     color="#c44e52", alpha=0.4, label="negative contribution")
    grad_value = float(np.trapezoid(integrand, a_grid))
    ax.axhline(0, color="#222", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("action $a$")
    ax.set_ylabel(r"$\pi(a) \cdot r(a) \cdot (a - \mu)/\sigma^2$")
    ax.set_title(
        f"Integrand of gradient at μ = {mu}\n"
        rf"$\nabla_\mu J = \int \pi(a)\,r(a)\,(a-\mu)/\sigma^2\,da \approx {grad_value:.3f}$",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # ----- Panel 3: gradient ascent trajectory -----
    ax = axes[2]
    rng = np.random.default_rng(0)
    policy = GaussianPolicy(mu_init=0.0, sigma=sigma)
    mus = [policy.mu]
    n_steps = 60
    n_samples_per_step = 30
    lr = 0.05

    for step in range(n_steps):
        # Sample n_samples_per_step actions and take a stochastic gradient step.
        grad_estimate = 0.0
        for _ in range(n_samples_per_step):
            a = policy.sample(rng)
            grad_estimate += reward(a) * policy.grad_log_prob_mu(a)
        grad_estimate /= n_samples_per_step
        policy.mu += lr * grad_estimate
        mus.append(policy.mu)

    ax.plot(mus, color="#3a7ebf", linewidth=2, marker="o", markersize=4,
             markevery=5, label="REINFORCE")
    ax.axhline(2.0, color="#c44e52", linestyle="--", linewidth=1.5,
                label="optimal $\\mu^\\star = 2$")
    ax.set_xlabel("gradient step")
    ax.set_ylabel(r"$\mu$ (policy mean)")
    ax.set_title(
        f"Gradient ascent\n"
        f"lr = {lr}, batch = {n_samples_per_step}, σ = {sigma}",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.2, 2.4)

    fig.suptitle(
        "Policy gradient on a 1D continuous bandit. Each sampled action contributes "
        r"$r(a)\cdot(a-\mu)/\sigma^2$ to the gradient — high-magnitude rewards far from μ dominate.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02a_log_derivative_trick.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
