"""Gridworld MDPs.

Two flavors:
  - GridWorld: a general grid with start, terminals, walls, and step reward.
  - TwoGoalGridWorld: a fixed 5x5 layout with a small near-by reward and a
    bigger far-away reward, designed to show how the discount factor changes
    the optimal policy.

States are 2D cells (row, col). Actions are 0=Up, 1=Right, 2=Down, 3=Left.
With slip probability p, the action goes uniformly at random over the
three *non*-intended directions with probability p, and as intended with
probability 1-p. p=0 is deterministic.
"""

from __future__ import annotations

import numpy as np

# Action conventions: 0=Up, 1=Right, 2=Down, 3=Left.
ACTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
ACTION_NAMES = ["Up", "Right", "Down", "Left"]


class GridWorld:
    """Discrete gridworld MDP.

    Attributes
    ----------
    rows, cols : int
    walls       : set of (r, c) tuples. Walls cannot be entered; an action
                  that would land on a wall keeps the agent in place.
    terminals   : dict from (r, c) to terminal reward. Terminal states are
                  absorbing (no further transitions, no further rewards).
    step_reward : float reward per non-terminal step.
    slip        : in [0, 1]. Probability that the action slips to a random
                  non-intended direction.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        terminals: dict[tuple[int, int], float],
        walls: set[tuple[int, int]] | None = None,
        step_reward: float = -0.04,
        slip: float = 0.0,
    ):
        self.rows = rows
        self.cols = cols
        self.terminals = dict(terminals)
        self.walls = set(walls) if walls else set()
        self.step_reward = float(step_reward)
        self.slip = float(slip)
        self.nA = 4

        # All non-wall cells are states; terminals are also states (absorbing).
        self.states = [(r, c) for r in range(rows) for c in range(cols)
                        if (r, c) not in self.walls]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.nS = len(self.states)

    def is_terminal(self, s: tuple[int, int]) -> bool:
        return s in self.terminals

    def step_outcomes(self, s: tuple[int, int], a: int):
        """Return list of (next_state, prob, reward) for taking action a in state s."""
        if self.is_terminal(s):
            return [(s, 1.0, 0.0)]

        outcomes = []  # list of (s', prob, reward)
        # Probability mass on intended vs slipped directions.
        for a_taken in range(self.nA):
            if a_taken == a:
                prob = 1.0 - self.slip
            else:
                prob = self.slip / 3.0
            if prob == 0:
                continue
            dr, dc = ACTIONS[a_taken]
            nr, nc = s[0] + dr, s[1] + dc
            ns = (nr, nc)
            # If next cell is off-grid or a wall, stay in place.
            if (nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols
                    or ns in self.walls):
                ns = s
            reward = self.terminals[ns] if ns in self.terminals else self.step_reward
            outcomes.append((ns, prob, reward))
        return outcomes

    def transition_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (P, R) where:
            P[s, a, s'] = transition probability,
            R[s, a]     = expected immediate reward.
        Indexed by self.state_to_idx.
        """
        P = np.zeros((self.nS, self.nA, self.nS))
        R = np.zeros((self.nS, self.nA))
        for s in self.states:
            si = self.state_to_idx[s]
            for a in range(self.nA):
                for ns, prob, r in self.step_outcomes(s, a):
                    nsi = self.state_to_idx[ns]
                    P[si, a, nsi] += prob
                    R[si, a] += prob * r
        return P, R


class TwoGoalGridWorld(GridWorld):
    """A 5x5 grid with a small near-by reward and a bigger far reward.

    Layout (S = start, +1 = small goal, +10 = big goal, # = wall):
        . . . . +1
        . # # . .
        S . . . .
        . # # . .
        . . . . +10

    The big goal is 4-7 steps from start; the small one is 4 steps. With
    high discount the big reward wins; with low discount the small does.
    """

    def __init__(self, slip: float = 0.0, step_reward: float = -0.04):
        terminals = {(0, 4): 1.0, (4, 4): 10.0}
        walls = {(1, 1), (1, 2), (3, 1), (3, 2)}
        super().__init__(
            rows=5, cols=5, terminals=terminals, walls=walls,
            step_reward=step_reward, slip=slip,
        )
        self.start = (2, 0)
