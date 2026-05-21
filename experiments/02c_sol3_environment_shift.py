"""Solution 2c.3: PPO vs REINFORCE under environment shift.

Mirror of 1d's solution 4 (Q-learning under non-stationarity), now for
policy gradient methods. The +1 and -1 terminals swap positions partway
through training.

The clipping/trust region adds STABILITY but reduces ADAPTIVITY. So we
expect PPO to recover from the shift more slowly than REINFORCE, but
without the catastrophic dip — its policy doesn't run far away from
the data each batch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import GridWorld
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_ppo,
    train_reinforce,
)


def main() -> None:
    gamma = 0.95
    env_p1 = GridWorld(rows=5, cols=5,
                        terminals={(4, 4): 1.0, (0, 4): -1.0},
                        step_reward=-0.04, slip=0.0)
    env_p2 = GridWorld(rows=5, cols=5,
                        terminals={(0, 4): 1.0, (4, 4): -1.0},
                        step_reward=-0.04, slip=0.0)

    n_episodes_per_phase = 1500

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    # REINFORCE
    policy = SoftmaxPolicy(env_p1.nS, env_p1.nA)
    rng = np.random.default_rng(0)
    h1 = train_reinforce(env_p1, policy, n_episodes=n_episodes_per_phase,
                          lr=0.1, gamma=gamma, baseline="mean", rng=rng)
    h2 = train_reinforce(env_p2, policy, n_episodes=n_episodes_per_phase,
                          lr=0.1, gamma=gamma, baseline="mean", rng=rng)
    returns_r = np.concatenate([h1["returns"], h2["returns"]])
    w = 50
    if len(returns_r) >= w:
        ma_r = np.convolve(returns_r, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(ma_r)) + w // 2, ma_r,
                color="#c44e52", linewidth=2,
                label=f"REINFORCE  (per-episode update, lr=0.1)")

    # PPO
    policy = SoftmaxPolicy(env_p1.nS, env_p1.nA)
    baseline = TabularBaseline(env_p1.nS)
    rng = np.random.default_rng(0)
    n_iter_per_phase = n_episodes_per_phase // 16
    h1 = train_ppo(env_p1, policy, baseline, n_iterations=n_iter_per_phase,
                    n_trajectories_per_iter=16, n_epochs=10,
                    lr_actor=0.05, lr_critic=0.2,
                    gamma=gamma, lam=0.95, eps=0.2, rng=rng)
    h2 = train_ppo(env_p2, policy, baseline, n_iterations=n_iter_per_phase,
                    n_trajectories_per_iter=16, n_epochs=10,
                    lr_actor=0.05, lr_critic=0.2,
                    gamma=gamma, lam=0.95, eps=0.2, rng=rng)
    # Expand PPO returns to per-episode resolution.
    returns_p_per_iter = np.concatenate([h1["returns"], h2["returns"]])
    returns_p = np.repeat(returns_p_per_iter, 16)
    if len(returns_p) >= w:
        ma_p = np.convolve(returns_p, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(ma_p)) + w // 2, ma_p,
                color="#3a7ebf", linewidth=2,
                label="PPO  (10 epochs/batch, ε=0.2)")

    ax.axvline(n_episodes_per_phase, color="#222", linestyle="--",
                linewidth=1.5, alpha=0.6)
    ax.text(n_episodes_per_phase, ax.get_ylim()[1] * 0.95, " ← world flips",
            fontsize=10, color="#222", ha="left", va="top")

    ax.set_xlabel("episode (approx.)")
    ax.set_ylabel(f"return ({w}-ep moving average)")
    ax.set_title(
        f"Policy-gradient methods under a hard environment switch.\n"
        f"REINFORCE adapts faster post-shift; PPO trades adaptation for stability.",
        fontsize=11,
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02c_sol3_environment_shift.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
