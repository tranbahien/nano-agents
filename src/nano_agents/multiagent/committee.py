"""Committees of agents: voting, correlation, and the Condorcet jury theorem.

Companion to posts/05a-multi-agent-systems.qmd.

A committee of `n_agents` answers multiple-choice items. Each agent is correct
with marginal probability `accuracy` p. The crucial knob is `correlation` rho:
the fraction of items on which the agents fail (or succeed) *in lockstep* --
all giving the same answer -- because they share blind spots (same base model,
same training data, same tempting distractor). On the remaining (1 - rho)
fraction, agents answer independently.

This single knob spans the two regimes that decide whether voting helps:

  - rho = 0 : independent errors. Majority vote obeys the Condorcet jury
              theorem -- accuracy -> 1 as the committee grows (if p > 1/2).
  - rho > 0 : correlated errors. Majority-vote accuracy is capped at
              1 - rho*(1 - p), no matter how many agents you add.

Diversity (low rho) is the resource that makes a committee worth more than
its members; correlation is what destroys it.
"""

from __future__ import annotations

from math import comb

import numpy as np


def condorcet_accuracy(p: float, n: int) -> float:
    """Majority-vote accuracy of n independent agents each correct w.p. p.

    Binary (correct / wrong-and-aligned) jury model; ties split by a coin flip.
    """
    n = int(n)
    acc = 0.0
    for k in range(n + 1):
        w = comb(n, k) * p ** k * (1 - p) ** (n - k)
        if k > n / 2:
            acc += w
        elif k == n / 2:
            acc += 0.5 * w
    return float(acc)


def correlated_vote_ceiling(p: float, rho: float) -> float:
    """Large-committee majority-vote accuracy under lockstep correlation rho.

    As n -> infinity: independent items are decided correctly (prob 1 when
    p>1/2), lockstep items are correct only with prob p. So the ceiling is
    rho*p + (1-rho)*1 = 1 - rho*(1-p).
    """
    return 1.0 - rho * (1.0 - p)


class Committee:
    """A committee of correlated multiple-choice agents.

    Parameters
    ----------
    n_agents : int
    n_answers : int
        Choices per item (chance accuracy = 1/n_answers).
    accuracy : float
        Marginal per-agent accuracy p.
    correlation : float
        Lockstep fraction rho in [0, 1]: probability an item is answered in
        unison by all agents (shared correctness and, when wrong, a shared
        wrong answer).
    seed : int
    """

    def __init__(
        self,
        n_agents: int = 5,
        n_answers: int = 4,
        accuracy: float = 0.6,
        correlation: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_agents = int(n_agents)
        self.n_answers = int(n_answers)
        self.accuracy = float(accuracy)
        self.correlation = float(correlation)
        self.seed = int(seed)

    def _wrong_answer(self, rng, truth, size=None):
        """Sample wrong answer(s) uniformly from the non-truth choices."""
        w = rng.integers(0, self.n_answers - 1, size=size)
        return np.where(w < truth, w, w + 1)

    def answer_matrix(self, n_items: int, rng):
        """Return (truth[n_items], answers[n_items, n_agents])."""
        truth = rng.integers(0, self.n_answers, size=n_items)
        ans = np.empty((n_items, self.n_agents), dtype=int)
        lockstep = rng.random(n_items) < self.correlation
        for j in range(n_items):
            t = int(truth[j])
            if lockstep[j]:
                # one shared answer for the whole committee
                shared = t if rng.random() < self.accuracy else int(
                    self._wrong_answer(rng, t))
                ans[j, :] = shared
            else:
                correct = rng.random(self.n_agents) < self.accuracy
                wrong = self._wrong_answer(rng, t, size=self.n_agents)
                ans[j, :] = np.where(correct, t, wrong)
        return truth, ans


def majority_vote(answers, n_answers: int, rng=None):
    """Plurality vote across axis 1; ties broken randomly."""
    answers = np.asarray(answers)
    n_items = answers.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.empty(n_items, dtype=int)
    for j in range(n_items):
        counts = np.bincount(answers[j], minlength=n_answers)
        top = np.flatnonzero(counts == counts.max())
        out[j] = top[rng.integers(len(top))]
    return out


def committee_accuracy(committee: Committee, n_items: int = 2000,
                       n_trials: int = 1, seed: int = 0) -> float:
    """Mean majority-vote accuracy of a committee over items and trials."""
    rng = np.random.default_rng(seed)
    correct = total = 0
    for _ in range(n_trials):
        truth, ans = committee.answer_matrix(n_items, rng)
        voted = majority_vote(ans, committee.n_answers, rng)
        correct += int((voted == truth).sum())
        total += n_items
    return correct / total
