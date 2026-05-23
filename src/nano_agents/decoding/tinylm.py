"""A small autoregressive language model used to study decoding strategies.

The model is a Markov chain over a tiny vocabulary, with transition logits
designed so that a small number of "good" sequences receive high reward and
the model has been (imperfectly) trained toward them. The space is small
enough to enumerate all sequences, which lets us compute *exact* properties
of any decoding strategy — no Monte Carlo error in the analytics.

Why a synthetic model? Because the post is about *decoding-time inference*,
not language modeling. We need a generative process with known structure
where we can ask precise questions about MAP, sampling, beam search, and
self-consistency. A toy Markov LM is the cleanest substrate for that.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class TinyMarkovLM:
    """Markov-bigram language model over a small vocabulary.

    Generates sequences of length L by sampling token-by-token from
    P(t_{n+1} | t_n). A 'reward' function — known to us but hidden from
    the LM — scores complete sequences. The LM is trained to favor
    high-reward sequences but not perfectly.

    Vocabulary layout:
        0: <BOS>     (always the starting token; never re-emitted)
        1..V-2:      content tokens
        V-1: <EOS>   (terminates a sequence; only allowed at final step)
    """

    def __init__(self, vocab_size: int = 8, seq_length: int = 4,
                 noise: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V = vocab_size
        self.L = seq_length  # number of content tokens generated
        self.BOS = 0

        # Reward structure: a small number of "correct" sequences score
        # high. We sample a few correct content-token sequences and give
        # them positive reward.
        n_good = 3
        good_set: list[tuple[int, ...]] = []
        while len(good_set) < n_good:
            seq = tuple(int(rng.integers(1, vocab_size))
                        for _ in range(seq_length))
            if seq not in good_set:
                good_set.append(seq)
        self.good_sequences = good_set
        # "Almost good" sequences (1-token edits of good ones) score
        # somewhere between 0 and the good reward — these are how MAP
        # collapse can happen.
        self.partial_reward = 0.2

        # Transition logits: each row is logits over the next token.
        # We initialize random, then upweight transitions that appear in
        # at least one good sequence — this is the "training" step.
        raw = noise * rng.normal(0, 1, (vocab_size, vocab_size))
        # Forbid emitting BOS again at any step.
        raw[:, self.BOS] = -1e9
        for seq in good_set:
            # Pump up p(seq[0] | BOS)
            raw[self.BOS, seq[0]] += 3.0
            for i in range(len(seq) - 1):
                raw[seq[i], seq[i + 1]] += 2.0

        self.logits = raw  # shape (V, V)

    # ---- distributions ----

    def conditional_logits(self, current: int) -> np.ndarray:
        """Logits for the next token given current token.

        BOS is always blocked from being re-emitted.
        """
        return self.logits[current]

    def conditional_probs(self, current: int, T: float = 1.0) -> np.ndarray:
        return softmax(self.conditional_logits(current) / max(T, 1e-8))

    # ---- exact enumeration ----

    def all_sequences(self) -> list[tuple[int, ...]]:
        """All content-token sequences of length L over the non-BOS vocab."""
        seqs: list[tuple[int, ...]] = [()]
        for _ in range(self.L):
            seqs = [s + (t,) for s in seqs
                    for t in range(1, self.V)]
        return seqs

    def sequence_log_prob(self, seq: tuple[int, ...]) -> float:
        """Joint log probability of generating this content sequence."""
        cur = self.BOS
        lp = 0.0
        for t in seq:
            p = self.conditional_probs(cur, T=1.0)
            lp += float(np.log(p[t] + 1e-30))
            cur = t
        return lp

    def reward(self, seq: tuple[int, ...]) -> float:
        if tuple(seq) in self.good_sequences:
            return 1.0
        # Partial credit for sequences that share at least L-1 tokens with
        # any good sequence — this is what creates the MAP-collapse pitfall.
        for g in self.good_sequences:
            matches = sum(1 for a, b in zip(seq, g) if a == b)
            if matches >= self.L - 1:
                return self.partial_reward
        return 0.0

    # ---- sampling ----

    def sample_sequence(self, rng, T: float = 1.0,
                          top_k: int | None = None,
                          top_p: float | None = None
                          ) -> tuple[int, ...]:
        """Generate one sequence using the given decoding parameters."""
        cur = self.BOS
        out: list[int] = []
        for _ in range(self.L):
            p = self.conditional_probs(cur, T=T)
            p = _apply_top_k_top_p(p, top_k=top_k, top_p=top_p)
            t = int(rng.choice(self.V, p=p))
            out.append(t)
            cur = t
        return tuple(out)


def _apply_top_k_top_p(p: np.ndarray, top_k: int | None,
                         top_p: float | None) -> np.ndarray:
    """Standard top-k / top-p truncation, then renormalize."""
    if top_k is None and top_p is None:
        return p
    p = p.copy()
    if top_k is not None and top_k > 0:
        # Keep the highest k.
        if top_k < len(p):
            cutoff = np.partition(p, -top_k)[-top_k]
            p[p < cutoff] = 0.0
    if top_p is not None and 0 < top_p < 1:
        order = np.argsort(p)[::-1]
        cumsum = np.cumsum(p[order])
        # Tokens whose cumulative mass exceeds top_p (and aren't the first
        # to reach it) get cut.
        keep_in_sorted = cumsum <= top_p
        # Always keep at least the top token.
        keep_in_sorted[0] = True
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep_in_sorted]] = True
        p = np.where(mask, p, 0.0)
    s = p.sum()
    return p / s if s > 0 else p
