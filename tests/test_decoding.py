"""Tests for the decoding subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.decoding import (
    TinyMarkovLM,
    all_beams,
    beam_search,
    self_consistency,
)
from nano_agents.decoding.tinylm import _apply_top_k_top_p


def test_tinylm_basic_structure():
    lm = TinyMarkovLM(vocab_size=8, seq_length=3, seed=0)
    # Good sequences are actually good.
    for seq in lm.good_sequences:
        assert lm.reward(seq) == 1.0


def test_tinylm_sequence_log_prob_normalizes():
    """Summing exp(logprob) over all sequences (then EOS) should give ~1."""
    lm = TinyMarkovLM(vocab_size=6, seq_length=3, seed=0)
    seqs = lm.all_sequences()
    probs = np.array([np.exp(lm.sequence_log_prob(s)) for s in seqs])
    assert abs(probs.sum() - 1.0) < 1e-6, f"Got {probs.sum()}, expected 1.0"


def test_top_k_truncation():
    p = np.array([0.4, 0.3, 0.15, 0.10, 0.05])
    p_k2 = _apply_top_k_top_p(p, top_k=2, top_p=None)
    # Only top 2 should be nonzero.
    assert np.count_nonzero(p_k2) == 2
    # And they should sum to 1.
    assert abs(p_k2.sum() - 1.0) < 1e-9


def test_top_p_truncation():
    p = np.array([0.4, 0.3, 0.15, 0.10, 0.05])
    # Cumulative: 0.4, 0.7, 0.85, 0.95, 1.0
    # top_p = 0.8 keeps {0.4, 0.3} (cum 0.7 <= 0.8) but also adds the next
    # one to cross the 0.8 boundary; conventions vary. Our impl keeps tokens
    # whose cumulative-from-top is <= top_p AND always keeps the top token.
    p_p = _apply_top_k_top_p(p, top_k=None, top_p=0.7)
    assert abs(p_p.sum() - 1.0) < 1e-9
    # The top token is always kept.
    assert p_p[0] > 0


def test_beam_search_returns_high_prob_sequence():
    """Beam search with beam_width = 1 should match greedy argmax."""
    lm = TinyMarkovLM(vocab_size=6, seq_length=3, seed=0)
    beam_seq = beam_search(lm, beam_width=1)
    # Verify by enumeration: this should be the highest-prob sequence
    # among all those reachable greedily.
    cur = lm.BOS
    greedy = []
    for _ in range(lm.L):
        p = lm.conditional_probs(cur, T=1.0)
        t = int(p.argmax())
        greedy.append(t)
        cur = t
    assert beam_seq == tuple(greedy)


def test_beam_search_beats_or_equals_greedy_on_full_logprob():
    """Beam search with width > 1 should never have a *lower* log-prob
    than width 1."""
    lm = TinyMarkovLM(vocab_size=6, seq_length=4, seed=0)
    seq1 = beam_search(lm, beam_width=1)
    seqK = beam_search(lm, beam_width=4)
    lp1 = lm.sequence_log_prob(seq1)
    lpK = lm.sequence_log_prob(seqK)
    assert lpK >= lp1 - 1e-9


def test_self_consistency_picks_argmax_at_low_T():
    lm = TinyMarkovLM(vocab_size=6, seq_length=3, seed=0)
    rng = np.random.default_rng(0)
    # At low T, the winner should dominate the vote.
    winner, counts = self_consistency(lm, n_samples=20, rng=rng, T=0.05)
    assert counts[winner] >= 15  # solid majority
