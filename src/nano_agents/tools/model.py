"""Synthetic agent model and tasks.

We don't have a real LLM here, so we model the relevant behavior: an agent
whose *internal* arithmetic is reliable for small operands and degrades for
large ones — exactly the failure mode that motivates giving LLMs a
calculator. A tool call returns the exact answer (modulo tool errors).

This lets us study the control question precisely: given a noisy internal
solver and an exact-but-costly tool, when should the agent call the tool?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArithmeticProblem:
    a: int
    op: str
    b: int

    @property
    def answer(self) -> int:
        if self.op == "+":
            return self.a + self.b
        if self.op == "-":
            return self.a - self.b
        if self.op == "*":
            return self.a * self.b
        raise ValueError(self.op)

    @property
    def difficulty(self) -> float:
        """A scalar difficulty proxy: total digit count of the operands,
        with multiplication weighted more heavily.
        """
        digits = len(str(abs(self.a))) + len(str(abs(self.b)))
        weight = 2.0 if self.op == "*" else 1.0
        return weight * digits

    def as_tuple(self) -> tuple:
        return (self.a, self.op, self.b)


def sample_problem(rng, max_digits: int = 4, ops=("+", "-", "*")) -> ArithmeticProblem:
    """Sample a random arithmetic problem with operands up to max_digits long."""
    d_a = int(rng.integers(1, max_digits + 1))
    d_b = int(rng.integers(1, max_digits + 1))
    a = int(rng.integers(10 ** (d_a - 1), 10 ** d_a))
    b = int(rng.integers(10 ** (d_b - 1), 10 ** d_b))
    op = str(rng.choice(ops))
    return ArithmeticProblem(a, op, b)


class NoisyArithmeticModel:
    """Models internal arithmetic that degrades with difficulty.

    P(correct internally) = base_accuracy * exp(-decay * (difficulty - d0)),
    clipped to [0, base_accuracy]. Small problems are nearly always right;
    big multiplications are nearly always wrong.
    """

    def __init__(self, base_accuracy: float = 0.99, decay: float = 0.45,
                 d0: float = 2.0):
        self.base_accuracy = base_accuracy
        self.decay = decay
        self.d0 = d0

    def internal_accuracy(self, difficulty: float) -> float:
        p = self.base_accuracy * np.exp(-self.decay * max(difficulty - self.d0, 0.0))
        return float(np.clip(p, 0.0, self.base_accuracy))

    def solve_internally(self, problem: ArithmeticProblem, rng) -> tuple:
        """Returns (answer, is_correct). Wrong answers are plausible but off."""
        p = self.internal_accuracy(problem.difficulty)
        if rng.random() < p:
            return (problem.answer, True)
        # Produce a plausible wrong answer (off by a small multiple of a
        # power of ten, the way digit-level mistakes look).
        true = problem.answer
        magnitude = 10 ** int(rng.integers(0, max(len(str(abs(true))), 1)))
        wrong = true + int(rng.choice([-3, -2, -1, 1, 2, 3])) * magnitude
        if wrong == true:
            wrong = true + 1
        return (wrong, False)
