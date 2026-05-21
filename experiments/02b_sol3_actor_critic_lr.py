"""Solution 2b.3: actor vs critic learning rates.

A2C has two separate learning rates: lr_actor (for theta) and lr_critic
(for V_phi). Classical wisdom says the critic should be faster than the
actor — a good baseline needs to be accurate before the actor can benefit.
We sweep both and visualize the resulting return as a 2D heatmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_a2c,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    n_episodes = 1500
    n_seeds = 3

    lr_actors = [0.01, 0.03, 0.1, 0.3]
    lr_critics = [0.05, 0.15, 0.4, 0.8]

    results = np.zeros((len(lr_actors), len(lr_critics)))
    for i, la in enumerate(lr_actors):
        for j, lc in enumerate(lr_critics):
            finals = np.zeros(n_seeds)
            for seed in range(n_seeds):
                policy = SoftmaxPolicy(env.nS, env.nA)
                baseline = TabularBaseline(env.nS)
                hist = train_a2c(
                    env, policy, baseline,
                    n_episodes=n_episodes,
                    lr_actor=la, lr_critic=lc,
                    gamma=gamma, lam=0.95,
                    rng=np.random.default_rng(seed),
                )
                finals[seed] = float(np.mean(hist["returns"][-100:]))
            results[i, j] = float(np.mean(finals))
            print(f"  lr_a={la}, lr_c={lc}: {results[i,j]:.2f}")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(results, cmap="viridis", aspect="auto", origin="lower",
                    vmin=results.min(), vmax=results.max())
    ax.set_xticks(range(len(lr_critics)))
    ax.set_xticklabels([f"{lc}" for lc in lr_critics])
    ax.set_yticks(range(len(lr_actors)))
    ax.set_yticklabels([f"{la}" for la in lr_actors])
    ax.set_xlabel(r"critic learning rate $\alpha_\phi$")
    ax.set_ylabel(r"actor learning rate $\alpha_\theta$")
    ax.set_title(
        f"A2C+GAE final return vs the two learning rates\n"
        f"avg over {n_seeds} seeds, {n_episodes} episodes each",
        fontsize=11,
    )

    # Annotate each cell.
    for i in range(len(lr_actors)):
        for j in range(len(lr_critics)):
            val = results[i, j]
            color = ("white" if val < (results.min() + results.max()) / 2
                     else "black")
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("final return (last 100 episodes)")

    # Mark the best cell.
    best_i, best_j = np.unravel_index(np.argmax(results), results.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                 facecolor="none", edgecolor="#c44e52",
                                 linewidth=3.5))

    fig.tight_layout()
    out = Path("figures/02b_sol3_actor_critic_lr.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
