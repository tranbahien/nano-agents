"""Closed-form analysis of one round of self-reflection.

Let a = generator accuracy, d = critic detection rate, f = critic
false-alarm rate, a' = revision accuracy. After one critique-and-revise
round (revise only flagged answers), the probability the final answer is
correct is

    A1 = a * [(1 - f) + f * a']        # started correct
       + (1 - a) * [d * a']             # started wrong, detected, revised

so the change relative to no reflection (A0 = a) is

    delta = A1 - a
          = (1 - a) * d * a'            # FIX: catch a wrong answer, revise it
          - a * f * (1 - a')            # BREAK: false-alarm a right answer, botch it

Reflection helps iff the fix term exceeds the break term. With *blind*
revision (a' = a) this collapses to the clean

    delta = a * (1 - a) * (d - f),

i.e. reflection helps iff the critic discriminates (d > f, Youden's J > 0),
with the largest gain at a = 1/2.
"""

from __future__ import annotations


def one_round_accuracy(a: float, d: float, f: float, a_prime: float) -> float:
    """Expected accuracy after one critique-and-revise round."""
    return a * ((1.0 - f) + f * a_prime) + (1.0 - a) * (d * a_prime)


def fix_term(a: float, d: float, a_prime: float) -> float:
    """Expected rate of fixing a wrong answer."""
    return (1.0 - a) * d * a_prime


def break_term(a: float, f: float, a_prime: float) -> float:
    """Expected rate of breaking a correct answer."""
    return a * f * (1.0 - a_prime)


def reflection_delta(a: float, d: float, f: float, a_prime: float) -> float:
    """Net accuracy change from one reflection round (fix minus break)."""
    return fix_term(a, d, a_prime) - break_term(a, f, a_prime)
