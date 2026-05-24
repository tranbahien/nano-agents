"""Multi-agent systems: voting, debate, and pipelines.

Companion to posts/05a-multi-agent-systems.qmd.
"""

from .committee import (
    Committee,
    committee_accuracy,
    condorcet_accuracy,
    correlated_vote_ceiling,
    majority_vote,
)
from .debate import debate, debate_accuracy, pipeline_accuracy

__all__ = [
    "Committee",
    "condorcet_accuracy",
    "correlated_vote_ceiling",
    "majority_vote",
    "committee_accuracy",
    "debate",
    "debate_accuracy",
    "pipeline_accuracy",
]
