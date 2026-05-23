"""The ReAct loop: interleaved Reasoning and Acting (Yao et al. 2022).

The agent alternates between:
  - THINK: an internal reasoning step (no environment effect).
  - ACT:   call a tool, receive an observation.
  - ANSWER: emit a final answer, ending the episode.

This is exactly an episode in a POMDP (cf. Post 1d): the world state is
hidden, tool calls produce observations, and the agent must accumulate
enough observations to answer correctly. The trace below is the agent's
'belief' being built up one observation at a time.

We provide a multi-hop knowledge-graph environment where answering requires
chaining several lookups, plus a scripted oracle policy and a noisy policy
for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nano_agents.tools.tools import KnowledgeGraph, Tool, make_lookup


@dataclass
class MultiHopTask:
    """A question requiring a chain of lookups through the graph.

    start: the starting entity.
    chain: list of relations to follow in order.
    answer: the final value after following the whole chain (precomputed).
    """
    start: str
    chain: list
    answer: object


@dataclass
class ReActTrace:
    """Record of one ReAct episode."""
    steps: list = field(default_factory=list)  # list of (kind, content)
    final_answer: object = None
    correct: bool = False
    n_tool_calls: int = 0

    def add(self, kind: str, content):
        self.steps.append((kind, content))


def build_example_graph(rng, n_entities: int = 12, chain_len: int = 3,
                          seed: int = 0) -> tuple:
    """Construct a small knowledge graph and a task that needs chain_len hops.

    Returns (KnowledgeGraph, MultiHopTask).
    """
    rng = np.random.default_rng(seed)
    kg = KnowledgeGraph()
    entities = [f"E{i}" for i in range(n_entities)]
    relations = ["next", "parent", "linked"]

    # Build a random graph where every entity has each relation pointing
    # somewhere (to keep traversal well-defined).
    for e in entities:
        for r in relations:
            target = entities[int(rng.integers(n_entities))]
            kg.add(e, r, target)

    # Terminal numeric attribute on every entity.
    values = {e: int(rng.integers(1, 100)) for e in entities}
    for e in entities:
        kg.add(e, "value", values[e])

    # Build a chain_len task: start -> r1 -> r2 -> ... -> value.
    start = entities[int(rng.integers(n_entities))]
    chain_relations = [str(rng.choice(relations)) for _ in range(chain_len)]
    chain = chain_relations + ["value"]

    # Compute the true answer by traversing.
    cur = start
    for r in chain_relations:
        cur, ok = kg.query(cur, r)
        assert ok
    answer, ok = kg.query(cur, "value")
    assert ok

    return kg, MultiHopTask(start=start, chain=chain, answer=answer)


def run_react_oracle(kg: KnowledgeGraph, task: MultiHopTask,
                      lookup_tool: Tool) -> ReActTrace:
    """An oracle ReAct policy that knows the relation chain and follows it.

    Demonstrates the ideal trace: think, lookup, observe, repeat, answer.
    """
    trace = ReActTrace()
    cur = task.start
    trace.add("think", f"I need to follow the chain {task.chain} from {cur}.")
    for r in task.chain:
        trace.add("act", ("lookup", (cur, r)))
        result, ok = lookup_tool((cur, r))
        trace.add("observe", result if ok else "ERROR")
        trace.n_tool_calls += 1
        if not ok:
            trace.final_answer = None
            trace.correct = False
            return trace
        cur = result
    trace.final_answer = cur
    trace.correct = (cur == task.answer)
    trace.add("answer", cur)
    return trace


def run_react_noisy(kg: KnowledgeGraph, task: MultiHopTask, lookup_tool: Tool,
                     rng, wrong_relation_prob: float = 0.0,
                     max_steps: int = 10) -> ReActTrace:
    """A noisy ReAct policy that may pick the wrong relation at each step.

    Models an imperfect agent: with probability wrong_relation_prob it
    follows an incorrect relation, derailing the chain.
    """
    trace = ReActTrace()
    cur = task.start
    relations = ["next", "parent", "linked", "value"]
    for i, correct_r in enumerate(task.chain):
        if i >= max_steps:
            break
        # Decide which relation to follow.
        if rng.random() < wrong_relation_prob:
            r = str(rng.choice(relations))
        else:
            r = correct_r
        trace.add("act", ("lookup", (cur, r)))
        result, ok = lookup_tool((cur, r))
        trace.add("observe", result if ok else "ERROR")
        trace.n_tool_calls += 1
        if not ok:
            trace.correct = False
            return trace
        cur = result
    trace.final_answer = cur
    trace.correct = (cur == task.answer)
    trace.add("answer", cur)
    return trace
