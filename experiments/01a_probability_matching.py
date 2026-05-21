"""Figure: Probability matching property of Thompson Sampling.

Empirically verifies that the probability TS picks arm k equals the
posterior probability that arm k is optimal. Side-by-side bars: empirical
pick frequencies vs Monte Carlo estimate of P(arm k is best | posterior).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Fix a set of posteriors that are partially overlapping — interesting case.
    # Arm 1: Beta(8, 12)  → mean 0.40, narrow-ish
    # Arm 2: Beta(15, 10) → mean 0.60, narrow-ish
    # Arm 3: Beta(5, 4)   → mean 0.55, wide (less data)
    # Arm 4: Beta(3, 2)   → mean 0.60, very wide (little data)
    alpha = np.array([8.0, 15.0, 5.0, 3.0])
    beta_p = np.array([12.0, 10.0, 4.0, 2.0])
    K = len(alpha)

    rng = np.random.default_rng(0)
    n_trials = 100_000

    # Empirical: run TS selection many times under this fixed posterior.
    samples = rng.beta(alpha[:, None], beta_p[:, None], size=(K, n_trials))
    picks = np.argmax(samples, axis=0)
    empirical = np.bincount(picks, minlength=K) / n_trials

    # Analytical (Monte Carlo) estimate of P(arm k is best | posterior).
    # This is exactly what TS samples from — so it should match empirical above.
    # We use a separate large sample purely to ground-truth.
    gt_samples = rng.beta(alpha[:, None], beta_p[:, None], size=(K, 1_000_000))
    gt_picks = np.argmax(gt_samples, axis=0)
    analytical = np.bincount(gt_picks, minlength=K) / gt_picks.size

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # Left panel: the posteriors themselves
    from scipy.stats import beta as beta_dist
    x = np.linspace(0, 1, 500)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(K):
        pdf = beta_dist.pdf(x, alpha[k], beta_p[k])
        ax1.plot(x, pdf, color=colors[k], linewidth=2,
                 label=f"Arm {k+1}: Beta({int(alpha[k])}, {int(beta_p[k])})")
        ax1.fill_between(x, pdf, alpha=0.15, color=colors[k])
    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel("posterior density")
    ax1.set_title("Four posteriors with overlapping support", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, 1)

    # Right panel: empirical vs analytical
    arms = np.arange(K)
    width = 0.38
    ax2.bar(arms - width / 2, analytical, width, color="#888",
            edgecolor="white", label=r"$P(\mu_k = \max_j \mu_j \mid \mathcal{D})$")
    ax2.bar(arms + width / 2, empirical, width, color="#3a7ebf",
            edgecolor="white", label="TS pick frequency")

    for k in range(K):
        ax2.text(k - width / 2, analytical[k] + 0.01, f"{analytical[k]:.3f}",
                 ha="center", fontsize=9, color="#444")
        ax2.text(k + width / 2, empirical[k] + 0.01, f"{empirical[k]:.3f}",
                 ha="center", fontsize=9, color="#1f4d7a")

    ax2.set_xticks(arms)
    ax2.set_xticklabels([f"Arm {k+1}" for k in range(K)])
    ax2.set_ylabel("probability")
    ax2.set_title("Probability matching: pick frequency = P(optimal)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(empirical.max(), analytical.max()) * 1.25)

    fig.suptitle(
        "Thompson Sampling picks arm $k$ exactly as often as $k$ is the optimal arm under the posterior.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out = Path("figures/01a_probability_matching.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
