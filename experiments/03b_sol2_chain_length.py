"""Solution 3b.2: error compounding over multi-hop chains.

A noisy agent picks the wrong relation at each step with probability p.
For a chain of length L, the probability of getting *every* step right is
(1 - p)^L — exponential decay in chain length. This is the central
challenge of long-horizon agents: small per-step error rates compound
catastrophically.

We sweep chain length and per-step error rate, comparing the empirical
success rate to the (1 - p)^L prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.tools import build_example_graph, make_lookup, run_react_noisy


def main() -> None:
    chain_lengths = list(range(1, 11))
    error_rates = [0.0, 0.05, 0.1, 0.2]
    n_tasks = 400

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(error_rates)))

    # Left: empirical success vs chain length, several error rates.
    ax = axes[0]
    for p_err, color in zip(error_rates, colors):
        success = np.zeros(len(chain_lengths))
        for j, cl in enumerate(chain_lengths):
            n_correct = 0
            for t in range(n_tasks):
                kg, task = build_example_graph(
                    np.random.default_rng(t), chain_len=cl, seed=t)
                lookup = make_lookup(kg)
                trace = run_react_noisy(
                    kg, task, lookup, rng=np.random.default_rng(t + 5000),
                    wrong_relation_prob=p_err, max_steps=20)
                n_correct += int(trace.correct)
            success[j] = n_correct / n_tasks
        ax.plot(chain_lengths, success, "o-", color=color, linewidth=2,
                markersize=7, label=f"per-step error = {p_err}")
        # Overlay (1-p)^L prediction (the chain has cl+1 lookups: cl
        # relations plus the value hop, but the value hop is never wrong
        # in our noise model, so effective length is cl).
        pred = [(1 - p_err) ** cl for cl in chain_lengths]
        ax.plot(chain_lengths, pred, "--", color=color, alpha=0.5, linewidth=1.5)

    ax.set_xlabel("chain length L")
    ax.set_ylabel("task success rate")
    ax.set_title("Success decays with L (dashed $(1-p)^L$ is a lower bound)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(chain_lengths)

    # Right: the same data as a heatmap of success rate.
    ax = axes[1]
    fine_errors = np.linspace(0, 0.3, 16)
    grid = np.zeros((len(fine_errors), len(chain_lengths)))
    for i, p_err in enumerate(fine_errors):
        for j, cl in enumerate(chain_lengths):
            grid[i, j] = (1 - p_err) ** cl
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn",
                    extent=[chain_lengths[0] - 0.5, chain_lengths[-1] + 0.5,
                            fine_errors[0], fine_errors[-1]],
                    vmin=0, vmax=1)
    ax.set_xlabel("chain length L")
    ax.set_ylabel("per-step error rate p")
    ax.set_title(r"Predicted success $(1-p)^L$")
    # Contour lines at 0.5 and 0.9 success.
    X, Y = np.meshgrid(chain_lengths, fine_errors)
    cs = ax.contour(X, Y, grid, levels=[0.5, 0.9], colors="black",
                     linewidths=1.5)
    ax.clabel(cs, fmt="%.1f")
    fig.colorbar(im, ax=ax, label="success rate")

    fig.suptitle(
        "Error compounding: even a 10% per-step error rate collapses to "
        "~35% success on a 10-hop chain. Long-horizon reliability is hard.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path("figures/03b_sol2_chain_length.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
