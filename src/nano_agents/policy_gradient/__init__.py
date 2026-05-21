"""Policy gradient methods.

Companion to: posts/02a-reinforce-and-policy-gradient.qmd
                posts/02b-actor-critic.qmd
"""

from .actor_critic import (
    entropy_grad,
    policy_entropy,
    train_a2c,
)
from .advantage import (
    discounted_returns_from_advantages,
    gae,
    n_step_returns,
)
from .baselines import TabularBaseline
from .policies import GaussianPolicy, SoftmaxPolicy
from .ppo import (
    collect_batch,
    compute_batch_advantages,
    ppo_update_step,
    train_ppo,
)
from .reinforce import (
    collect_trajectory,
    compute_returns,
    reinforce_step,
    train_reinforce,
)

__all__ = [
    "SoftmaxPolicy",
    "GaussianPolicy",
    "TabularBaseline",
    "collect_trajectory",
    "compute_returns",
    "reinforce_step",
    "train_reinforce",
    "train_a2c",
    "train_ppo",
    "collect_batch",
    "compute_batch_advantages",
    "ppo_update_step",
    "policy_entropy",
    "entropy_grad",
    "gae",
    "n_step_returns",
    "discounted_returns_from_advantages",
]
