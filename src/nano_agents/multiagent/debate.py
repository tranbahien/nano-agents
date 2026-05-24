"""Debate dynamics and sequential pipelines.

Two more multi-agent patterns beyond independent voting:

  - `debate`: agents see each other's current answers and revise. Each round
    an agent either *conforms* (adopts the current plurality) with probability
    `conformity`, or *reconsiders* by drawing a fresh independent opinion.
    Conformity converges the committee to a consensus -- which helps when the
    plurality is right and *herds* the group onto a wrong answer when it is
    not. Stubborn agents (who never conform) model overconfident participants.

  - `pipeline_accuracy`: a chain of agents where each must succeed for the
    whole task to succeed. Reliability is the product of per-stage
    reliabilities, p**L -- the same error-compounding cliff as long-horizon
    tool use (Post 3b).
"""

from __future__ import annotations

import numpy as np

from .committee import Committee, majority_vote


def _plurality(row, n_answers, rng):
    counts = np.bincount(row, minlength=n_answers)
    top = np.flatnonzero(counts == counts.max())
    return int(top[rng.integers(len(top))])


def debate(
    committee: Committee,
    n_items: int,
    rng,
    rounds: int = 4,
    conformity: float = 0.5,
    stubborn_frac: float = 0.0,
):
    """Simulate multi-round debate; return accuracy per round (incl. round 0).

    Round 0 is the initial independent answers (one-shot majority vote
    baseline). Each later round, non-stubborn agents conform to the plurality
    with probability `conformity`, otherwise redraw an independent opinion.
    Stubborn agents keep their round-0 answer forever.
    """
    truth, ans = committee.answer_matrix(n_items, rng)
    n_ag = committee.n_agents
    n_stub = int(round(stubborn_frac * n_ag))
    stubborn = np.zeros(n_ag, dtype=bool)
    stubborn[:n_stub] = True
    initial = ans.copy()

    accs = [_vote_acc(ans, truth, committee.n_answers, rng)]
    for _ in range(rounds):
        new = ans.copy()
        for j in range(n_items):
            plur = _plurality(ans[j], committee.n_answers, rng)
            for i in range(n_ag):
                if stubborn[i]:
                    new[j, i] = initial[j, i]
                elif rng.random() < conformity:
                    new[j, i] = plur
                else:
                    # reconsider: fresh independent opinion
                    t = int(truth[j])
                    if rng.random() < committee.accuracy:
                        new[j, i] = t
                    else:
                        w = rng.integers(0, committee.n_answers - 1)
                        new[j, i] = w if w < t else w + 1
        ans = new
        accs.append(_vote_acc(ans, truth, committee.n_answers, rng))
    return np.array(accs)


def _vote_acc(ans, truth, n_answers, rng):
    voted = majority_vote(ans, n_answers, rng)
    return float((voted == truth).mean())


def debate_accuracy(committee, n_items=1500, rounds=4, conformity=0.5,
                    stubborn_frac=0.0, seed=0):
    """Final-round debate accuracy."""
    rng = np.random.default_rng(seed)
    return float(debate(committee, n_items, rng, rounds, conformity,
                        stubborn_frac)[-1])


def pipeline_accuracy(stage_accuracy: float, n_stages: int) -> float:
    """Reliability of a sequential pipeline: every stage must succeed.

    p**L -- the multiplicative error-compounding of a chain of agents.
    """
    return float(stage_accuracy ** n_stages)
