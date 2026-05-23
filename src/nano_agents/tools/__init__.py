"""Tool use as actions: turning an LM into an agent.

Companion to posts/03b-tool-use.qmd.
"""

from .model import (
    ArithmeticProblem,
    NoisyArithmeticModel,
    sample_problem,
)
from .react import (
    MultiHopTask,
    ReActTrace,
    build_example_graph,
    run_react_noisy,
    run_react_oracle,
)
from .tools import (
    KnowledgeGraph,
    Tool,
    make_calculator,
    make_lookup,
)

__all__ = [
    "Tool",
    "KnowledgeGraph",
    "make_calculator",
    "make_lookup",
    "ArithmeticProblem",
    "NoisyArithmeticModel",
    "sample_problem",
    "MultiHopTask",
    "ReActTrace",
    "build_example_graph",
    "run_react_oracle",
    "run_react_noisy",
]
