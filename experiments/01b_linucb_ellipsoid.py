"""Figure: LinUCB predictive uncertainty and confidence ellipsoid.

Visualizes the LinUCB exploration mechanism: the algorithm picks contexts/arms
where the upper confidence bound of the predictive distribution is highest.
The bound shrinks toward the true function as data accumulates.

Uses d=2 features [1, z] so the regression is a line over the scalar z.
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
    return mu, cov, A


def main() -> None:
    rng = np.random.default_rng(1)
    sigma = 0.4
    lam = 1.0
    true_theta = np.array([0.3, 1.0])  # f(z) = 0.3 + z
    alpha = 1.5  # LinUCB exploration parameter

    # Sample z from a non-uniform distribution so some regions are under-explored.
    # Heavily favor positive z to create asymmetric uncertainty.
    T_max = 200
    z_all = np.concatenate([
        rng.uniform(0.5, 2.0, int(T_max * 0.75)),
        rng.uniform(-2.0, 0.5, int(T_max * 0.25)),
    ])
    rng.shuffle(z_all)
    X_all = np.column_stack([np.ones(len(z_all)), z_all])
    y_all = X_all @ true_theta + rng.normal(0, sigma, len(z_all))

    T_targets = [20, 200]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    z_grid = np.linspace(-2.5, 2.5, 300)
    X_grid = np.column_stack([np.ones_like(z_grid), z_grid])

    for ax, n in zip(axes, T_targets):
        X, y = X_all[:n], y_all[:n]
        mu, cov, A = posterior(X, y, sigma=sigma, lam=lam)

        # Predictive mean and std
        f_mean = X_grid @ mu
        f_var = np.einsum("ni,ij,nj->n", X_grid, cov, X_grid)
        f_std = np.sqrt(np.maximum(f_var, 0))

        # 95% predictive band (mean ± 2σ over θ; noise σ NOT added for clarity)
        f_lo, f_hi = f_mean - 2 * f_std, f_mean + 2 * f_std
        # LinUCB upper bound
        ucb = f_mean + alpha * f_std

        # True function
        f_true = X_grid @ true_theta

        # Plot
        ax.fill_between(z_grid, f_lo, f_hi, color="#3a7ebf", alpha=0.18,
                         label=r"95% credible band: $\hat f \pm 2\sqrt{x^\top A^{-1} x}$")
        ax.plot(z_grid, f_mean, color="#3a7ebf", linewidth=2,
                label=r"posterior mean $x^\top \hat\theta$")
        ax.plot(z_grid, ucb, color="#f4a261", linewidth=2,
                label=rf"LinUCB index ($\alpha={alpha}$)")
        ax.plot(z_grid, f_true, color="#222", linewidth=1.5, linestyle="--",
                label=r"true $f(z) = 0.3 + z$")

        # Data
        ax.scatter(z_all[:n], y_all[:n], s=14, color="#c44e52",
                   alpha=0.65, edgecolor="none", zorder=4,
                   label=f"observed data (n={n})")

        ax.set_title(f"n = {n}", fontsize=12)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.5, 4.5)
        ax.set_xlabel("z")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("reward")
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.95, ncol=1)
    fig.suptitle(
        "LinUCB picks the action with the highest upper confidence index "
        "(orange). As data accumulates, the band tightens around the true "
        "function and the bonus shrinks.",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()

    out = Path("figures/01b_linucb_ellipsoid.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
