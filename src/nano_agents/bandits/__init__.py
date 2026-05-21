"""Multi-armed bandits.

Companion to: posts/01a-multi-armed-bandits.qmd
"""

from .agents import (
    EpsilonGreedy,
    GaussianThompsonSampling,
    GaussianUCB,
    ThompsonSampling,
    UCB1,
)
from .environments import BernoulliBandit, DriftingBernoulliBandit, GaussianBandit
from .experiments import run, run_many

__all__ = [
    "BernoulliBandit",
    "DriftingBernoulliBandit",
    "GaussianBandit",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonSampling",
    "GaussianUCB",
    "GaussianThompsonSampling",
    "run",
    "run_many",
]
