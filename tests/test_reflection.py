"""Tests for the reflection subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.reflection import (
    ReflectiveQA,
    accuracy,
    best_of_n,
    one_round_accuracy,
    reflect,
    reflect_with_oracle,
    reflection_delta,
    self_consistency,
    single_shot,
)


def test_env_shapes_and_chance():
    env = ReflectiveQA(n_problems=200, n_answers=5, seed=0)
    assert env.correct.shape == (200,)
    assert env.correct.min() >= 0 and env.correct.max() < 5


def test_generator_accuracy_matches_parameter():
    env = ReflectiveQA(n_problems=2000, n_answers=5, gen_accuracy=0.4, seed=1)
    acc = accuracy(env, single_shot, n_trials=3, seed=2)
    assert abs(acc - 0.4) < 0.03


def test_critic_rates():
    # A critic with detect=0.8 should flag ~80% of wrong answers, ~10% of right.
    env = ReflectiveQA(n_problems=4000, detect_rate=0.8, false_alarm=0.1, seed=0)
    rng = np.random.default_rng(0)
    flagged_wrong = flagged_right = nwrong = nright = 0
    for p in range(env.n_problems):
        truth = int(env.correct[p])
        wrong = (truth + 1) % env.n_answers
        if env.critic(p, truth, rng):
            flagged_right += 1
        nright += 1
        if env.critic(p, wrong, rng):
            flagged_wrong += 1
        nwrong += 1
    assert abs(flagged_wrong / nwrong - 0.8) < 0.04
    assert abs(flagged_right / nright - 0.1) < 0.04


def test_discrimination_property():
    env = ReflectiveQA(detect_rate=0.7, false_alarm=0.2)
    assert abs(env.discrimination - 0.5) < 1e-9


def test_analytic_matches_simulation_blind():
    # Blind revision (revise_gain=0): simulated one-round reflect should match
    # the closed-form one_round_accuracy with a'=a, within Monte Carlo noise.
    a, d, f = 0.4, 0.7, 0.2
    env = ReflectiveQA(n_problems=4000, gen_accuracy=a, detect_rate=d,
                       false_alarm=f, revise_gain=0.0, seed=0)
    sim = accuracy(env, reflect, n_trials=2, seed=3, max_rounds=1)
    analytic = one_round_accuracy(a, d, f, a_prime=a)
    assert abs(sim - analytic) < 0.03


def test_reflection_helps_iff_discriminating():
    # Good critic (d>f): reflection beats single-shot.
    good = ReflectiveQA(n_problems=3000, gen_accuracy=0.4, detect_rate=0.8,
                        false_alarm=0.15, seed=0)
    base = accuracy(good, single_shot, n_trials=2, seed=1)
    refl = accuracy(good, reflect, n_trials=2, seed=1, max_rounds=3)
    assert refl > base + 0.02

    # Anti-discriminating critic (f>d): reflection hurts.
    bad = ReflectiveQA(n_problems=3000, gen_accuracy=0.6, detect_rate=0.2,
                       false_alarm=0.6, seed=0)
    base_b = accuracy(bad, single_shot, n_trials=2, seed=1)
    refl_b = accuracy(bad, reflect, n_trials=2, seed=1, max_rounds=3)
    assert refl_b < base_b - 0.02


def test_oracle_reflection_is_monotonic_and_strong():
    env = ReflectiveQA(n_problems=3000, gen_accuracy=0.4, seed=0)
    base = accuracy(env, single_shot, n_trials=2, seed=1)
    oracle = accuracy(env, reflect_with_oracle, n_trials=2, seed=1, max_rounds=4)
    # A perfect verifier never breaks a correct answer; with several rounds it
    # should approach 1 - (1-a)^(rounds+1) >> base.
    assert oracle > base + 0.2


def test_delta_sign_matches_helps():
    assert reflection_delta(a=0.4, d=0.8, f=0.15, a_prime=0.4) > 0
    assert reflection_delta(a=0.6, d=0.2, f=0.6, a_prime=0.6) < 0


def test_best_of_n_and_self_consistency_run():
    env = ReflectiveQA(n_problems=500, gen_accuracy=0.4, detect_rate=0.75,
                       false_alarm=0.2, seed=0)
    bon = accuracy(env, best_of_n, n_trials=1, seed=0, n=4)
    sc = accuracy(env, self_consistency, n_trials=1, seed=0, n=5)
    assert 0.0 <= bon <= 1.0 and 0.0 <= sc <= 1.0
    # Self-consistency should beat a single sample when a > chance.
    base = accuracy(env, single_shot, n_trials=1, seed=0)
    assert sc >= base - 0.05
