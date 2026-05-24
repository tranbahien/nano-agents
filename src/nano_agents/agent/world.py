"""A simulated world for exercising the nanoAgent controller.

Companion to posts/05b-nanoagent.qmd.

The nanoAgent's *controller* is the real artifact (controller.py); to run it
end to end without a live LLM we plug it into this faithful simulation, the
same way the rest of the curriculum used toy models that preserve the
mechanism. Each question has a *type* and a latent difficulty, and each
capability mirrors an earlier post:

  - generate  : the base model (Post 3a) -- correct with a type/difficulty
                dependent probability, reporting a (possibly miscalibrated)
                confidence (Post 4c);
  - use_tool  : a calculator-like tool (Post 3b) -- near-perfect on
                `calculation` questions, useless elsewhere;
  - retrieve  : retrieval (Post 4a) -- strong on `knowledge` questions,
                noisy elsewhere;
  - reflect   : a self-critic (Post 4b) -- can fix `reasoning` questions when
                it discriminates, at the risk of over-correction.

The world knows the ground truth (for scoring); the agent never sees it.
"""

from __future__ import annotations

import numpy as np

from nano_agents.calibration.model import logit, sigmoid

QUESTION_TYPES = ("knowledge", "calculation", "reasoning")

# Per-type base generator accuracy (at average difficulty). The base model is
# weak where an external capability shines, so routing matters.
_BASE_ACCURACY = {"knowledge": 0.45, "calculation": 0.40, "reasoning": 0.60}

# Capability accuracies on their "home" type.
TOOL_ACCURACY = 0.97          # calculator on calculation questions
RETRIEVAL_ACCURACY = 0.90     # retrieval on knowledge questions
# Reflection self-critic (Post 4b): detection, false alarm, revised accuracy.
REFLECT_DETECT = 0.75
REFLECT_FALSE_ALARM = 0.20
REFLECT_REVISE = 0.60


class SimulatedWorld:
    """A bank of typed questions plus the capabilities an agent can invoke."""

    def __init__(self, n_questions: int = 2000, n_answers: int = 5,
                 overconfidence: float = 1.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n_questions = int(n_questions)
        self.n_answers = int(n_answers)
        self.overconfidence = float(overconfidence)
        self.types = rng.choice(QUESTION_TYPES, size=n_questions)
        self.difficulty = rng.normal(0.0, 1.0, size=n_questions)   # latent logit shift
        self.correct = rng.integers(0, n_answers, size=n_questions)

    # -- ground truth helpers (agent never calls these) ------------------

    def _gen_accuracy(self, qid: int) -> float:
        base = _BASE_ACCURACY[self.types[qid]]
        return float(np.clip(sigmoid(logit(base) - 0.6 * self.difficulty[qid]),
                             0.02, 0.98))

    def _wrong(self, rng, truth):
        w = int(rng.integers(0, self.n_answers - 1))
        return w if w < truth else w + 1

    def _sample(self, rng, truth, p):
        return int(truth) if rng.random() < p else self._wrong(rng, truth)

    # -- capabilities the controller orchestrates ------------------------

    def observe(self, qid: int):
        """What the agent sees before acting: the question type (not the answer)."""
        return {"type": self.types[qid]}

    def generate(self, qid: int, rng):
        """Base-model answer with a reported confidence (Post 3a + 4c)."""
        p = self._gen_accuracy(qid)
        ans = self._sample(rng, self.correct[qid], p)
        conf = float(sigmoid(self.overconfidence * logit(p)))   # may be miscalibrated
        return ans, conf

    def use_tool(self, qid: int, rng):
        """Calculator: near-perfect on calculation, useless elsewhere (Post 3b)."""
        p = TOOL_ACCURACY if self.types[qid] == "calculation" else self._gen_accuracy(qid)
        return self._sample(rng, self.correct[qid], p)

    def retrieve(self, qid: int, rng):
        """Retrieval: strong on knowledge, noisy elsewhere (Post 4a)."""
        p = RETRIEVAL_ACCURACY if self.types[qid] == "knowledge" else self._gen_accuracy(qid)
        return self._sample(rng, self.correct[qid], p)

    def reflect(self, qid: int, answer: int, rng):
        """Self-critic + revise (Post 4b). Returns (flagged, new_answer)."""
        is_correct = answer == self.correct[qid]
        if is_correct:
            flagged = rng.random() < REFLECT_FALSE_ALARM
        else:
            flagged = rng.random() < REFLECT_DETECT
        if not flagged:
            return False, answer
        return True, self._sample(rng, self.correct[qid], REFLECT_REVISE)

    def vote(self, qid: int, n: int, rng):
        """Sample n base-model answers and take the plurality (Post 3a/5a)."""
        p = self._gen_accuracy(qid)
        cands = [self._sample(rng, self.correct[qid], p) for _ in range(n)]
        vals, counts = np.unique(cands, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def is_correct(self, qid: int, answer: int) -> bool:
        return int(answer) == int(self.correct[qid])
