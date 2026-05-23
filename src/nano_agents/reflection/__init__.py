"""Self-reflection as Monte Carlo.

Companion to posts/04b-self-reflection.qmd.
"""

from .analysis import (
    break_term,
    fix_term,
    one_round_accuracy,
    reflection_delta,
)
from .environment import ReflectiveQA
from .loops import (
    accuracy,
    best_of_n,
    reflect,
    reflect_with_oracle,
    self_consistency,
    single_shot,
)

__all__ = [
    "ReflectiveQA",
    "single_shot",
    "reflect",
    "reflect_with_oracle",
    "best_of_n",
    "self_consistency",
    "accuracy",
    "one_round_accuracy",
    "fix_term",
    "break_term",
    "reflection_delta",
]
