"""Beam search and self-consistency decoding strategies.

Together with the temperature/top-k/top-p sampling in `tinylm.py`, this
covers the full spectrum of "what to do with a fixed LM at inference time"
that Post 3a discusses.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from nano_agents.decoding.tinylm import TinyMarkovLM


def beam_search(lm: TinyMarkovLM, beam_width: int) -> tuple[int, ...]:
    """Standard beam search. Returns the highest-log-prob complete sequence.

    Each beam is (running_logprob, current_token, sequence_so_far). We
    expand each beam by all possible next tokens, score, and keep the
    top `beam_width` by log probability.
    """
    beams: list[tuple[float, int, tuple[int, ...]]] = [(0.0, lm.BOS, ())]
    for step in range(lm.L):
        next_beams: list[tuple[float, int, tuple[int, ...]]] = []
        for lp, cur, seq in beams:
            probs = lm.conditional_probs(cur, T=1.0)
            for t in range(lm.V):
                if probs[t] <= 0:
                    continue
                next_beams.append(
                    (lp + float(np.log(probs[t])), t, seq + (t,)))
        # Keep top beam_width by log prob.
        next_beams.sort(key=lambda b: b[0], reverse=True)
        beams = next_beams[:beam_width]
    return beams[0][2]


def all_beams(lm: TinyMarkovLM, beam_width: int
                ) -> list[tuple[float, tuple[int, ...]]]:
    """Same as beam_search, but returns the full final beam (lp, seq)."""
    beams: list[tuple[float, int, tuple[int, ...]]] = [(0.0, lm.BOS, ())]
    for step in range(lm.L):
        next_beams: list[tuple[float, int, tuple[int, ...]]] = []
        for lp, cur, seq in beams:
            probs = lm.conditional_probs(cur, T=1.0)
            for t in range(lm.V):
                if probs[t] <= 0:
                    continue
                next_beams.append(
                    (lp + float(np.log(probs[t])), t, seq + (t,)))
        next_beams.sort(key=lambda b: b[0], reverse=True)
        beams = next_beams[:beam_width]
    return [(lp, seq) for (lp, _, seq) in beams]


def self_consistency(
    lm: TinyMarkovLM, n_samples: int, rng,
    T: float = 1.0, top_k: int | None = None, top_p: float | None = None,
    answer_extractor: Callable | None = None,
) -> tuple:
    """Sample many sequences and take the most common 'answer'.

    By default the 'answer' is the entire sequence. For tasks where multiple
    sequences encode the same answer (e.g., different reasoning paths to
    the same number), pass an `answer_extractor(sequence) -> hashable`.
    Returns (winning_answer, vote_counts).
    """
    if answer_extractor is None:
        answer_extractor = lambda s: s
    answers = []
    for _ in range(n_samples):
        seq = lm.sample_sequence(rng, T=T, top_k=top_k, top_p=top_p)
        answers.append(answer_extractor(seq))
    counts = Counter(answers)
    winner = counts.most_common(1)[0][0]
    return winner, counts
