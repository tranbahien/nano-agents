"""Figure: the PPO clipped surrogate L^CLIP.

Visualize, as a function of the importance ratio r, the unclipped objective
r*A, the clipped objective clip(r, 1-ε, 1+ε)*A, and the min of the two.
Two panels: one for positive advantage (good action), one for negative.

The min asymmetry is the key insight: for A>0, the clip caps the upside
(don't make a "good" action too much more probable in one step). For A<0,
the clip caps the downside (don't make a "bad" action too much less
probable in one step). Both prevent the new policy from moving too far
from the data-generating policy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    eps = 0.2
    r = np.linspace(0.0, 2.0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              sharey=False)

    for ax, A in zip(axes, [+1.0, -1.0]):
        unclipped = r * A
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        L_ppo = np.minimum(unclipped, clipped)

        ax.plot(r, unclipped, color="#888", linewidth=2, linestyle="--",
                 label=r"$r \cdot A$  (unclipped)")
        ax.plot(r, clipped, color="#3a7ebf", linewidth=2,
                 linestyle=":", label=rf"$\mathrm{{clip}}(r, 1-\epsilon, 1+\epsilon) \cdot A$")
        ax.plot(r, L_ppo, color="#c44e52", linewidth=3,
                 label=r"$L^{\mathrm{CLIP}} = \min(\cdot)$")

        # Shade the clipped regions (where gradient is zero).
        if A > 0:
            ax.axvspan(1 + eps, 2.0, color="#ffe5e5", alpha=0.5,
                        label="zero gradient region")
        else:
            ax.axvspan(0, 1 - eps, color="#ffe5e5", alpha=0.5,
                        label="zero gradient region")

        ax.axvline(1.0, color="#222", linewidth=0.7, alpha=0.4)
        ax.axvline(1 - eps, color="#888", linewidth=0.7, alpha=0.5,
                    linestyle=":")
        ax.axvline(1 + eps, color="#888", linewidth=0.7, alpha=0.5,
                    linestyle=":")
        ax.text(1 - eps, ax.get_ylim()[0] if A > 0 else ax.get_ylim()[1],
                r"$1-\epsilon$",
                fontsize=10, ha="center", va="bottom" if A > 0 else "top")
        ax.text(1 + eps, ax.get_ylim()[0] if A > 0 else ax.get_ylim()[1],
                r"$1+\epsilon$",
                fontsize=10, ha="center", va="bottom" if A > 0 else "top")

        ax.set_xlabel(r"importance ratio $r(\theta) = \pi_\theta(a|s) / \pi_{\theta_{\rm old}}(a|s)$")
        ax.set_ylabel(r"surrogate objective")
        ax.set_title(f"Advantage  A = {A}", fontsize=12)
        ax.legend(loc="upper left" if A > 0 else "lower left", fontsize=9)
        ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle(
        r"PPO clipped surrogate $L^{\mathrm{CLIP}}$. "
        "When the new policy strays past the clip boundary, the gradient is zero — preventing further drift.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02c_clipped_surrogate.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
