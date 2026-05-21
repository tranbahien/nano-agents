"""Contextual bandits.

Companion to: posts/01b-contextual-bandits.qmd
"""

from .agents import (
    ContextFreeUCB,
    ContextualEpsilonGreedy,
    LinearThompsonSampling,
    LinUCB,
)
from .environments import LinearContextualBandit, NonlinearContextualBandit
from .experiments import run_contextual, run_many_contextual

__all__ = [
    "LinearContextualBandit",
    "NonlinearContextualBandit",
    "LinUCB",
    "LinearThompsonSampling",
    "ContextualEpsilonGreedy",
    "ContextFreeUCB",
    "run_contextual",
    "run_many_contextual",
]
