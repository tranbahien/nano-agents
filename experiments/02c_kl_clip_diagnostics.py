"""Figure: KL divergence and clip fraction over PPO training.

Track:
  - Mean importance ratio (should stay near 1.0 if policy doesn't drift)
  - KL divergence between old and new policy (the implicit trust region)
  - Fraction of samples that hit the clip (an indicator of saturation)

These three diagnostics are essential in practice — if KL grows unbounded,
training is unstable; if clip fraction is too high, you're not benefiting
from the data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_ppo,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    n_iterations = 150

    configs = [
        (r"PPO  $\epsilon = 0.1$ (tight clip)", 0.1, "#3a7ebf"),
        (r"PPO  $\epsilon = 0.2$ (default)",   0.2, "#dd8452"),
        (r"PPO  $\epsilon = 0.5$ (loose clip)", 0.5, "#c44e52"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label, eps, color in configs:
        policy = SoftmaxPolicy(env.nS, env.nA)
        baseline = TabularBaseline(env.nS)
        hist = train_ppo(
            env, policy, baseline,
            n_iterations=n_iterations,
            n_trajectories_per_iter=16,
            n_epochs=10,
            lr_actor=0.05, lr_critic=0.2,
            gamma=gamma, lam=0.95, eps=eps,
            rng=np.random.default_rng(0),
        )

        axes[0].plot(hist["returns"], color=color, linewidth=2, label=label)
        axes[1].plot(hist["kl"], color=color, linewidth=2, label=label)
        axes[2].plot(hist["clip_fraction"], color=color, linewidth=2,
                      label=label)

    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("mean return")
    axes[0].set_title("Returns")
    axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"KL$(\pi_{\rm old}\,\|\,\pi_\theta)$  (per iteration)")
    axes[1].set_title("Implicit KL — bounded by the clip")
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("clip fraction")
    axes[2].set_title("Fraction of samples in the clipped (zero-gradient) region")
    axes[2].grid(alpha=0.3); axes[2].set_axisbelow(True)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "PPO diagnostics. Smaller ε ⇒ tighter trust region (smaller KL, higher clip fraction). "
        "All three converge to similar returns.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02c_kl_clip_diagnostics.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
