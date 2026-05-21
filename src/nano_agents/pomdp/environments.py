"""POMDP environments.

The Tiger problem (Cassandra, Kaelbling & Littman 1994) is the canonical
small POMDP example. Two states (tiger behind left/right door), three
actions (open-left, open-right, listen), two observations (growl-left,
growl-right). Listening is informative but costly; opening the wrong door
is catastrophic. The optimal policy depends on the *belief*, not the
hidden state.
"""

from __future__ import annotations

import numpy as np


class TigerPOMDP:
    """The two-door Tiger problem.

    States:        TL (0) = tiger behind left, TR (1) = tiger behind right.
    Actions:       OL (0) = open left, OR (1) = open right, Listen (2).
    Observations:  GL (0) = growl from left, GR (1) = growl from right.

    Opening a door 'resets' the world (next state uniform). Listening keeps
    the state and produces a noisy observation with the given accuracy.
    """

    def __init__(
        self,
        listen_accuracy: float = 0.85,
        listen_cost: float = -1.0,
        open_correct: float = 10.0,
        open_wrong: float = -100.0,
    ):
        self.listen_accuracy = float(listen_accuracy)
        self.listen_cost = float(listen_cost)
        self.open_correct = float(open_correct)
        self.open_wrong = float(open_wrong)

        self.nS, self.nA, self.nO = 2, 3, 2

        # Transition tensor P[s, a, s'].
        self.P = np.zeros((2, 3, 2))
        # Listen keeps state.
        self.P[0, 2, 0] = 1.0
        self.P[1, 2, 1] = 1.0
        # Opening either door resets to uniform.
        for a in (0, 1):
            self.P[:, a, 0] = 0.5
            self.P[:, a, 1] = 0.5

        # Observation tensor Z[s', a, o] = P(o | s', a).
        self.Z = np.zeros((2, 3, 2))
        # Listen: accuracy on each side.
        self.Z[0, 2, 0] = listen_accuracy        # in TL, Listen → GL w.p. acc
        self.Z[0, 2, 1] = 1.0 - listen_accuracy
        self.Z[1, 2, 0] = 1.0 - listen_accuracy
        self.Z[1, 2, 1] = listen_accuracy
        # Opening: uninformative observations.
        for a in (0, 1):
            self.Z[:, a, :] = 0.5

        # Reward tensor R[s, a].
        self.R = np.zeros((2, 3))
        self.R[:, 2] = listen_cost            # listening
        self.R[0, 0] = open_wrong             # TL, open left = tiger!
        self.R[0, 1] = open_correct           # TL, open right = gold
        self.R[1, 0] = open_correct           # TR, open left = gold
        self.R[1, 1] = open_wrong             # TR, open right = tiger!

    def sample_obs(self, s_next: int, a: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.nO, p=self.Z[s_next, a]))

    def step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, int, float]:
        """Sample (s', o, r) for a single step."""
        s_next = int(rng.choice(self.nS, p=self.P[s, a]))
        r = float(self.R[s, a])
        o = self.sample_obs(s_next, a, rng)
        return s_next, o, r
