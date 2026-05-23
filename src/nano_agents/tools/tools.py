"""Tool abstractions for an LLM agent.

A Tool is anything with a name and a callable interface that maps an input
(string or structured args) to an observation. The agent's action space is
augmented with these tools: at each step it may emit a token, call a tool,
or produce a final answer.

We provide two synthetic tools used throughout Post 3b:
  - Calculator: exact arithmetic (the agent's internal arithmetic is noisy).
  - Lookup: single-hop traversal of a small knowledge graph.

These are deliberately simple. The point is the *control structure* — when
to call a tool, how to incorporate its output — not the tools themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    """A named tool with a callable interface.

    fn maps an input to (observation, ok) where ok=False signals a tool
    error (bad arguments, missing entity, etc.).
    """
    name: str
    description: str
    fn: Callable
    call_count: int = 0

    def __call__(self, arg):
        self.call_count += 1
        return self.fn(arg)


def make_calculator(error_rate: float = 0.0, rng=None) -> Tool:
    """A calculator tool. With error_rate > 0, occasionally returns a wrong
    result (models an unreliable external API).
    """
    def calc(expr: tuple):
        a, op, b = expr
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        else:
            return (None, False)  # unsupported op -> tool error
        if rng is not None and error_rate > 0 and rng.random() < error_rate:
            # Corrupt the result.
            result = result + int(rng.choice([-2, -1, 1, 2]))
        return (result, True)

    return Tool(name="calculator",
                description="Computes a (op) b for op in {+, -, *}.",
                fn=calc)


@dataclass
class KnowledgeGraph:
    """A tiny knowledge graph: entity -> {relation -> value}.

    Values may be other entities (for multi-hop traversal) or numbers
    (terminal attributes).
    """
    edges: dict = field(default_factory=dict)

    def add(self, entity: str, relation: str, value):
        self.edges.setdefault(entity, {})[relation] = value

    def query(self, entity: str, relation: str):
        if entity not in self.edges or relation not in self.edges[entity]:
            return (None, False)
        return (self.edges[entity][relation], True)


def make_lookup(kg: KnowledgeGraph) -> Tool:
    """A lookup tool over a knowledge graph. Input: (entity, relation)."""
    def lookup(arg: tuple):
        entity, relation = arg
        return kg.query(entity, relation)

    return Tool(name="lookup",
                description="Looks up (entity, relation) in the knowledge graph.",
                fn=lookup)
