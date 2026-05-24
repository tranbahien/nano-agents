"""nanoAgent: a Bayes-consistent controller that orchestrates capabilities.

Companion to posts/05b-nanoagent.qmd.

This is the reference implementation the whole curriculum builds toward. The
agent maintains a single *belief* -- a calibrated probability that its current
answer is correct -- and at each step makes a value-of-information decision:
for every capability (call a tool, retrieve, reflect, vote), it estimates the
expected accuracy *after* that action, subtracts the action's cost, and takes
the best option only if that net gain beats simply answering now. When no
action's expected gain outweighs its cost, it stops and commits.

The expected-accuracy models the controller uses are exactly the analytic
results derived earlier in the series, imported and reused:

  - reflection.one_round_accuracy  (Post 4b) -- value of a reflect step;
  - multiagent.condorcet_accuracy  (Post 5a) -- value of voting;
  - calibration.sigmoid / logit    (Post 4c) -- calibrating the belief.

Calibration is what makes the whole thing work: the controller's decisions are
only as good as its belief is honest. An overconfident belief makes every
"answer now" look attractive, so the agent stops too early and never gathers
the evidence that would have fixed a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nano_agents.calibration.model import logit, sigmoid
from nano_agents.multiagent import condorcet_accuracy
from nano_agents.reflection import one_round_accuracy

ANSWER, TOOL, RETRIEVE, REFLECT, VOTE = "answer", "tool", "retrieve", "reflect", "vote"


@dataclass
class SelfModel:
    """What the agent believes about its own capabilities (its world-model)."""
    tool_accuracy: float = 0.97          # tool on its home type (calculation)
    retrieval_accuracy: float = 0.90     # retrieval on its home type (knowledge)
    detect: float = 0.75                 # reflection critic detection (Post 4b)
    false_alarm: float = 0.20            # reflection critic false-alarm
    revise: float = 0.60                 # accuracy of a revised answer
    vote_n: int = 4                      # extra samples drawn per vote action
    home_type = {TOOL: "calculation", RETRIEVE: "knowledge"}


@dataclass
class Costs:
    """Per-action cost in accuracy-equivalent units (latency, tokens, money)."""
    tool: float = 0.06
    retrieve: float = 0.06
    reflect: float = 0.05
    vote: float = 0.04                   # per sampled answer


@dataclass
class Trace:
    answer: int = -1
    correct: bool = False
    actions: list = field(default_factory=list)   # ordered actions taken
    beliefs: list = field(default_factory=list)    # belief after each step
    cost: float = 0.0


class NanoAgent:
    """A value-of-information controller over generate/tool/retrieve/reflect/vote."""

    def __init__(self, self_model: SelfModel | None = None,
                 costs: Costs | None = None, calibration_T: float = 1.0,
                 max_steps: int = 3) -> None:
        self.m = self_model or SelfModel()
        self.c = costs or Costs()
        self.calibration_T = float(calibration_T)
        self.max_steps = int(max_steps)

    # -- belief -----------------------------------------------------------

    def _calibrate(self, raw_conf: float) -> float:
        """Apply temperature scaling to the raw confidence (Post 4c)."""
        return float(sigmoid(logit(raw_conf) / self.calibration_T))

    # -- value of information --------------------------------------------

    def _expected_after(self, action: str, belief: float, qtype: str) -> float:
        """Expected accuracy of the current answer *after* taking `action`."""
        if action == TOOL:
            return self.m.tool_accuracy if qtype == self.m.home_type[TOOL] else belief
        if action == RETRIEVE:
            return self.m.retrieval_accuracy if qtype == self.m.home_type[RETRIEVE] else belief
        if action == REFLECT:
            return one_round_accuracy(belief, self.m.detect, self.m.false_alarm,
                                      self.m.revise)
        if action == VOTE:
            return condorcet_accuracy(belief, 1 + self.m.vote_n)
        raise ValueError(action)

    def _cost(self, action: str) -> float:
        return {TOOL: self.c.tool, RETRIEVE: self.c.retrieve,
                REFLECT: self.c.reflect,
                VOTE: self.c.vote * self.m.vote_n}[action]

    def _best_action(self, belief: float, qtype: str):
        """The action with the greatest expected gain net of cost, or ANSWER."""
        best, best_gain = ANSWER, 0.0
        for action in (TOOL, RETRIEVE, REFLECT, VOTE):
            gain = (self._expected_after(action, belief, qtype) - belief) \
                - self._cost(action)
            if gain > best_gain:
                best, best_gain = action, gain
        return best, best_gain

    # -- execution --------------------------------------------------------

    def _execute(self, action, world, qid, answer, belief, qtype, rng):
        """Run the chosen action against the world; return (answer, belief)."""
        if action == TOOL:
            return world.use_tool(qid, rng), self._expected_after(TOOL, belief, qtype)
        if action == RETRIEVE:
            return world.retrieve(qid, rng), self._expected_after(RETRIEVE, belief, qtype)
        if action == REFLECT:
            _, new = world.reflect(qid, answer, rng)
            return new, self._expected_after(REFLECT, belief, qtype)
        if action == VOTE:
            return world.vote(qid, 1 + self.m.vote_n, rng), \
                self._expected_after(VOTE, belief, qtype)
        raise ValueError(action)

    # -- the control loop -------------------------------------------------

    def solve(self, world, qid: int, rng) -> Trace:
        """Answer one question, choosing actions by value of information."""
        qtype = world.observe(qid)["type"]
        answer, raw_conf = world.generate(qid, rng)
        belief = self._calibrate(raw_conf)

        tr = Trace(actions=[], beliefs=[belief])
        for _ in range(self.max_steps):
            action, gain = self._best_action(belief, qtype)
            if action == ANSWER:
                break
            answer, belief = self._execute(action, world, qid, answer, belief,
                                            qtype, rng)
            tr.actions.append(action)
            tr.beliefs.append(belief)
            tr.cost += self._cost(action)

        tr.answer = int(answer)
        tr.correct = world.is_correct(qid, answer)
        return tr


def run_agent(world, agent: NanoAgent, seed: int = 0):
    """Solve every question in the world; return (accuracy, mean_cost, traces)."""
    rng = np.random.default_rng(seed)
    traces = [agent.solve(world, qid, rng) for qid in range(world.n_questions)]
    acc = float(np.mean([t.correct for t in traces]))
    cost = float(np.mean([t.cost for t in traces]))
    return acc, cost, traces
