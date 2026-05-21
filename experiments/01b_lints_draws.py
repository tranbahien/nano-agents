"""Figure: Linear Thompson Sampling — each posterior draw is a hypothesis.

Visualizes the central idea of LinTS: at each step, sample one theta_tilde
from the posterior and commit to it for the decision. As the posterior
contracts, draws agree more, and the algorithm transitions from exploration
to exploitation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def posterior(X, y, sigma=0.5, lam=1.0):
    d = X.shape[1]
    A = lam * np.eye(d) + (X.T @ X) / sigma ** 2
    cov = np.linalg.inv(A)
    mu = cov @ (X.T @ y) / sigma ** 2
    return mu, cov


def main() -> None:
    rng = np.random.default_rng(2)
    sigma = 0.5
    lam = 1.0
    true_theta = np.array([0.3, 1.0])
    n_draws = 10

    # Stream of data
    T_max = 200
    z_all = rng.uniform(-2, 2, T_max)
    X_all = np.column_stack([np.ones(T_max), z_all])
    y_all = X_all @ true_theta + rng.normal(0, sigma, T_max)

    T_targets = [10, 200]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    z_grid = np.linspace(-2.5, 2.5, 200)
    X_grid = np.column_stack([np.ones_like(z_grid), z_grid])

    sample_rng = np.random.default_rng(42)

    for ax, n in zip(axes, T_targets):
        X, y = X_all[:n], y_all[:n]
        mu, cov = posterior(X, y, sigma=sigma, lam=lam)

        # Draw n_draws samples from the posterior; each is one "hypothesis".
        thetas_tilde = sample_rng.multivariate_normal(mu, cov, size=n_draws)

        # Each hypothesis as a faint line
        for theta_t in thetas_tilde:
            f_t = X_grid @ theta_t
            ax.plot(z_grid, f_t, color="#3a7ebf", alpha=0.35, linewidth=1.2)

        # True function and posterior mean for reference
        ax.plot(z_grid, X_grid @ true_theta, color="#222", linewidth=2,
                linestyle="--", label=r"true $f(z) = 0.3 + z$", zorder=4)
        ax.plot(z_grid, X_grid @ mu, color="#c44e52", linewidth=2,
                label=r"posterior mean", zorder=5)

        # Data
        ax.scatter(z_all[:n], y_all[:n], s=14, color="#666",
                   alpha=0.6, edgecolor="none", zorder=3,
                   label=f"observed data (n={n})")

        ax.set_title(f"n = {n}    ({n_draws} posterior draws shown in blue)",
                     fontsize=12)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.5, 4.5)
        ax.set_xlabel("z")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("reward")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Linear Thompson Sampling: each step samples one $\\tilde\\theta$ "
        "from the posterior and acts as if it were truth.\n"
        "Wide posterior → diverse hypotheses → exploration. "
        "Tight posterior → similar hypotheses → exploitation.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()

    out = Path("figures/01b_lints_draws.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
