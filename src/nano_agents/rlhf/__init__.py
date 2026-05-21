"""RLHF, DPO, and GRPO on a small synthetic preference task.

Companion to: posts/02d-rlhf-dpo-grpo.qmd

The subpackage is built around a single tiny contextual-completion task
where the true reward function is known. This lets us:

  - Sample synthetic preference pairs via Bradley-Terry.
  - Train a reward model from those preferences.
  - Train a policy with PPO using the learned reward model (RLHF).
  - Train a policy with DPO directly from preferences (no reward model).
  - Train a policy with GRPO using a group-relative baseline.

All three target the same underlying task, so they're directly comparable.
"""

from .dpo import dpo_loss, dpo_loss_for_pair, train_dpo
from .grpo import train_grpo
from .reward_model import TabularRewardModel
from .task import PreferenceTask

__all__ = [
    "PreferenceTask",
    "TabularRewardModel",
    "train_dpo",
    "dpo_loss",
    "dpo_loss_for_pair",
    "train_grpo",
]
