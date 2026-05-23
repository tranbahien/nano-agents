"""Synthetic environment for self-reflection as Monte Carlo.

Companion to posts/04b-self-reflection.qmd.

A bank of multiple-choice problems. Three components stand in for the
pieces of a real reflective agent:

  - a **generator** that proposes an answer (correct with some accuracy),
  - a noisy **self-critic** that judges whether an answer is wrong, with a
    detection rate (catching genuine errors) and a false-alarm rate
    (flagging correct answers), and
  - a **reviser** that proposes a fresh answer when the critic complains,
    optionally helped by the critique (an accuracy gain).

The whole point is that the critic is *imperfect* and is the model judging
its own work — the same network that generated the answer. Whether
reflection helps turns entirely on whether that self-judgement
discriminates right from wrong.
"""

from __future__ import annotations

import numpy as np


class ReflectiveQA:
    """A multiple-choice problem bank with a generator, self-critic, reviser.

    Parameters
    ----------
    n_problems : int
        Number of problems in the bank.
    n_answers : int
        Number of candidate answers per problem (chance accuracy = 1/n).
    gen_accuracy : float
        P(the generator proposes the correct answer) in [0, 1].
    detect_rate : float
        P(critic flags as wrong | the answer IS wrong). Sensitivity.
    false_alarm : float
        P(critic flags as wrong | the answer is CORRECT). 1 - specificity.
    revise_gain : float
        Added to gen_accuracy when producing a revision, modelling an
        *informative* critique that localizes the error. 0.0 = blind
        resampling (revision is just another independent draw).
    seed : int
    """

    def __init__(
        self,
        n_problems: int = 400,
        n_answers: int = 5,
        gen_accuracy: float = 0.4,
        detect_rate: float = 0.7,
        false_alarm: float = 0.2,
        revise_gain: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_problems = n_problems
        self.n_answers = n_answers
        self.gen_accuracy = float(gen_accuracy)
        self.detect_rate = float(detect_rate)
        self.false_alarm = float(false_alarm)
        self.revise_gain = float(revise_gain)
        rng = np.random.default_rng(seed)
        self.correct = rng.integers(0, n_answers, size=n_problems)

    # -- components -------------------------------------------------------

    def generate(self, problem: int, rng, accuracy: float | None = None) -> int:
        """Propose an answer: correct w.p. `accuracy`, else a uniform wrong one."""
        a = self.gen_accuracy if accuracy is None else accuracy
        truth = int(self.correct[problem])
        if rng.random() < a:
            return truth
        # pick uniformly among the wrong answers
        wrong = rng.integers(0, self.n_answers - 1)
        return wrong if wrong < truth else wrong + 1

    def revise(self, problem: int, rng) -> int:
        """Propose a fresh answer after a critique (accuracy = gen + gain)."""
        a = np.clip(self.gen_accuracy + self.revise_gain, 0.0, 1.0)
        return self.generate(problem, rng, accuracy=a)

    def critic(self, problem: int, answer: int, rng) -> bool:
        """Return True if the self-critic flags `answer` as WRONG (revise it)."""
        is_correct = answer == self.correct[problem]
        if is_correct:
            return rng.random() < self.false_alarm        # false alarm
        return rng.random() < self.detect_rate             # true detection

    def is_correct(self, problem: int, answer: int) -> bool:
        return answer == int(self.correct[problem])

    @property
    def discrimination(self) -> float:
        """Critic informedness (Youden's J): detect_rate - false_alarm."""
        return self.detect_rate - self.false_alarm
