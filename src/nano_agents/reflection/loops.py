"""Inference strategies built on a ReflectiveQA environment.

Five strategies, all spending model calls in different ways:

  - `single_shot`        : generate once, accept.
  - `reflect`            : generate, then critique-and-revise until the critic
                           accepts or a round budget is hit (Reflexion-style).
  - `reflect_with_oracle`: same loop but with a *perfect* external verifier
                           (a tool / unit test) instead of self-critique.
  - `best_of_n`          : sample n candidates, keep the one the critic rates
                           best (noisy verifier reranking, cf. Post 3a).
  - `self_consistency`   : sample n candidates, majority-vote the answer.

All take an env, a problem index, and an rng, and return the chosen answer.
"""

from __future__ import annotations

import numpy as np

from .environment import ReflectiveQA


def single_shot(env: ReflectiveQA, problem: int, rng) -> int:
    return env.generate(problem, rng)


def reflect(
    env: ReflectiveQA,
    problem: int,
    rng,
    max_rounds: int = 3,
    use_oracle: bool = False,
) -> int:
    """Generate, then critique-and-revise until accepted or budget exhausted.

    With `use_oracle=True`, the critic is replaced by a perfect external
    verifier: it flags iff the answer is genuinely wrong (detect=1, false
    alarm=0). This models reflection with tool/test feedback rather than
    pure self-judgement.
    """
    answer = env.generate(problem, rng)
    for _ in range(max_rounds):
        if use_oracle:
            flagged = not env.is_correct(problem, answer)
        else:
            flagged = env.critic(problem, answer, rng)
        if not flagged:
            break                      # critic accepts -> stop
        answer = env.revise(problem, rng)
    return answer


def reflect_with_oracle(env, problem, rng, max_rounds: int = 3) -> int:
    return reflect(env, problem, rng, max_rounds=max_rounds, use_oracle=True)


def _critic_score(env, problem, answer, rng) -> float:
    """A noisy goodness score the critic assigns to a candidate (higher=better).

    Built from the same detect/false-alarm rates: a candidate the critic
    would flag as wrong gets a low score. Ties broken by noise.
    """
    flagged = env.critic(problem, answer, rng)
    return rng.random() * 0.5 + (0.0 if flagged else 1.0)


def best_of_n(env, problem, rng, n: int = 4) -> int:
    """Sample n candidates; return the one the critic scores highest."""
    cands = [env.generate(problem, rng) for _ in range(n)]
    scores = [_critic_score(env, problem, c, rng) for c in cands]
    return cands[int(np.argmax(scores))]


def self_consistency(env, problem, rng, n: int = 4) -> int:
    """Sample n candidates; return the majority-vote answer (no critic)."""
    cands = [env.generate(problem, rng) for _ in range(n)]
    vals, counts = np.unique(cands, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def accuracy(env, strategy, n_trials: int = 1, seed: int = 0, **kwargs) -> float:
    """Mean accuracy of a strategy over the problem bank, averaged over trials."""
    rng = np.random.default_rng(seed)
    correct = total = 0
    for _ in range(n_trials):
        for p in range(env.n_problems):
            ans = strategy(env, p, rng, **kwargs)
            correct += int(env.is_correct(p, ans))
            total += 1
    return correct / total
