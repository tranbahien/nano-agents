"""Figure: the reference-KL penalty prevents reward hacking.

We construct a situation where the *learned* reward model is mildly wrong
(slightly overestimates one completion's reward). With no reference KL,
the policy will exploit this misestimate and concentrate on the wrong
completion, sacrificing true reward.

With a reference KL penalty, the policy stays close enough to the (uniform)
reference that it doesn't fully exploit the error. Trades some peak reward
for resilience to reward-model error.

This is a small but informative reproduction of the "reward hacking"
phenomenon in real RLHF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf import PreferenceTask, TabularRewardModel, train_grpo


class CorruptedRewardModel:
    """Reward model that adds a positive bias to one specific completion.

    Used to simulate reward-model misestimation (the typical source of
    reward hacking in real RLHF).
    """
    def __init__(self, base_rewards: np.ndarray, hack_context: int,
                  hack_completion: int, bias: float):
        self.r = base_rewards.copy()
        self.r[hack_context, hack_completion] += bias

    def predict(self, c: int, a: int) -> float:
        return float(self.r[c, a])


def main() -> None:
    task = PreferenceTask(n_contexts=4, n_completions=4, reward_scale=2.0, seed=0)

    # Pick a context and a completion that is NOT the true best, and inject a
    # reward bias that makes the corrupted reward model rank it #1.
    true_best = task.true_rewards.argmax(axis=1)
    hack_context = 0
    # Choose a completion that is not the true best.
    others = [a for a in range(task.n_completions) if a != true_best[hack_context]]
    hack_completion = int(others[0])
    bias_needed = (task.true_rewards[hack_context, true_best[hack_context]]
                   - task.true_rewards[hack_context, hack_completion]
                   + 2.0)
    corrupted_rm = CorruptedRewardModel(
        task.true_rewards, hack_context, hack_completion, bias_needed)

    # Reference policy: uniform.
    ref_policy = SoftmaxPolicy(task.n_contexts, task.n_completions)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    n_iterations = 200
    kl_coefs = [0.0, 1.0, 5.0]
    colors = ["#c44e52", "#dd8452", "#3a7ebf"]

    for kl_coef, color in zip(kl_coefs, colors):
        policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
        hist = train_grpo(
            task, policy,
            n_iterations=n_iterations, group_size=8, n_prompts_per_iter=2,
            n_epochs=4, lr=0.2, eps=0.2,
            use_true_reward=False, reward_model=corrupted_rm,
            track_kl_against_ref=True,
            ref_policy=ref_policy, kl_coef=kl_coef,
            rng=np.random.default_rng(0))

        # Left: true reward over training (what we'd actually deploy).
        true_returns = np.zeros(n_iterations)
        # We need to re-evaluate the policy's expected TRUE reward at each
        # iteration; the history only has the *corrupted* reward signal.
        # The simplest approach: rerun training while logging.
        # Actually train_grpo already records expected_return as TRUE reward.
        true_returns = hist["expected_return"]
        axes[0].plot(true_returns, color=color, linewidth=2,
                      label=rf"$\beta_{{\rm KL}} = {kl_coef}$")

        # Right: KL to reference.
        axes[1].plot(hist["kl_to_ref"], color=color, linewidth=2,
                      label=rf"$\beta_{{\rm KL}} = {kl_coef}$")

    optimal_true = task.expected_reward_under_policy(
        np.eye(task.n_completions)[task.greedy_policy()])
    axes[0].axhline(optimal_true, color="#222", linestyle="--", linewidth=1,
                     alpha=0.5,
                     label=f"true optimum = {optimal_true:.2f}")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("expected TRUE reward")
    axes[0].set_title("True reward under corrupted reward model.\n"
                        "No KL → policy chases the wrong target.")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].set_xlabel("iteration"); axes[1].set_ylabel(r"KL$(\pi \,\|\, \pi_{\rm ref})$")
    axes[1].set_title("KL drift from reference policy")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

    fig.suptitle(
        "Reward hacking under corrupted reward model. The KL penalty "
        "(right) trades some headline reward for resilience to RM error (left).",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02d_kl_penalty_reward_hacking.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
