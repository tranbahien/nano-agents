"""Figure: Bayesian linear regression posterior evolution in 2D.

Shows the Gaussian posterior over theta in R^2 contracting toward the true
parameter as data accumulates. Analogue of the Beta posterior figure for
multi-armed bandits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_posterior(X, y, sigma=0.3, lam=1.0):
    """Return mean and covariance of theta given (X, y) under
    N(0, lam^{-1} I) prior and N(0, sigma^2) noise."""
    d = X.shape[1]
    A = lam * np.eye(d) + (X.T @ X) / sigma ** 2
    cov = np.linalg.inv(A)
    mu = cov @ (X.T @ y) / sigma ** 2
    return mu, cov


def main() -> None:
    rng = np.random.default_rng(0)
    sigma = 0.7  # higher noise → slower contraction → more visual signal
    lam = 1.0
    true_theta = np.array([1.2, -0.6])

    # Generate a stream of (x, y) pairs.
    T = 500
    X_all = rng.standard_normal((T, 2))
    X_all /= np.linalg.norm(X_all, axis=1, keepdims=True)  # unit-norm contexts
    y_all = X_all @ true_theta + rng.normal(0, sigma, T)

    T_targets = [0, 10, 100, 500]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)

    # Build a grid for contouring
    grid_size = 250
    g = np.linspace(-2.5, 2.5, grid_size)
    G1, G2 = np.meshgrid(g, g)
    grid = np.stack([G1.ravel(), G2.ravel()], axis=1)

    for ax, n in zip(axes, T_targets):
        if n == 0:
            mu = np.zeros(2)
            cov = np.eye(2) / lam
        else:
            mu, cov = gaussian_posterior(X_all[:n], y_all[:n], sigma=sigma, lam=lam)

        # Posterior density on the grid (normalized to peak = 1 for visualization)
        diff = grid - mu
        cov_inv = np.linalg.inv(cov)
        quad = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        density = np.exp(-0.5 * quad).reshape(grid_size, grid_size)

        # Fill the posterior with a soft gradient
        ax.contourf(G1, G2, density, levels=20, cmap="Blues", alpha=0.7,
                    vmin=0, vmax=1)

        # 68% and 95% confidence ellipses (chi-square levels for 2 DoF)
        # density level = exp(-0.5 * chi2) where chi2(2 DoF) = 2.30 (68%), 5.99 (95%)
        ax.contour(G1, G2, density, levels=[np.exp(-5.99 / 2), np.exp(-2.30 / 2)],
                   colors=["#1f4d7a", "#1f4d7a"], linewidths=[1.0, 1.6],
                   linestyles=["dashed", "solid"])

        # True parameter
        ax.plot(*true_theta, "*", color="#c44e52", markersize=20,
                markeredgecolor="white", markeredgewidth=1.5,
                label=r"true $\theta^\star$" if n == 0 else None, zorder=5)
        # Posterior mean
        ax.plot(*mu, "o", color="#222", markersize=8,
                markeredgecolor="white", markeredgewidth=1.5,
                label=r"posterior mean $\mu_n$" if n == 0 else None, zorder=5)

        ax.set_title(f"n = {n}", fontsize=12)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel(r"$\theta_1$")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    axes[0].set_ylabel(r"$\theta_2$")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Add a custom legend for the confidence-region lines on the last panel
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1f4d7a", lw=1.6, label="68% region"),
        Line2D([0], [0], color="#1f4d7a", lw=1.0, linestyle="dashed", label="95% region"),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper left",
                     fontsize=9, framealpha=0.95)

    fig.suptitle(
        r"Gaussian posterior over $\theta \in \mathbb{R}^2$ under "
        r"$\mathcal{N}(0, \lambda^{-1}I)$ prior with $\sigma=0.7$ noise. "
        r"The 68% and 95% regions both contract at rate $1/\sqrt{n}$.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()

    out = Path("figures/01b_blr_posterior_evolution.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
