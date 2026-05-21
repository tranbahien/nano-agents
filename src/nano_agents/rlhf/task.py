"""A synthetic preference-learning task.

The setting mirrors RLHF for language models, scaled down to a few contexts
and completions:

  - n_contexts contexts (think: prompts).
  - n_completions completions per context (think: candidate responses).
  - A "true" reward function r*(context, completion) ∈ R, known to us
    but unknown to the algorithms — they only see preferences.

Preference labels are sampled from the Bradley-Terry model:

    P(y_w ≻ y_l | x) = sigmoid(r*(x, y_w) - r*(x, y_l)).

This is the standard generative model assumed by RLHF reward-modeling.
The temperature is fixed at 1 here; in practice it's wrapped into the
reward scale.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class PreferenceTask:
    """Tabular preference task: tiny number of contexts × completions.

    Attributes:
        n_contexts:    number of contexts (prompts)
        n_completions: number of completions per context
        true_rewards:  shape (n_contexts, n_completions). Hidden from algorithms.
    """

    def __init__(self, n_contexts: int = 4, n_completions: int = 4,
                 reward_scale: float = 2.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_contexts = n_contexts
        self.n_completions = n_completions
        self.true_rewards = rng.uniform(
            -reward_scale, reward_scale, (n_contexts, n_completions))

    def reward(self, context: int, completion: int) -> float:
        return float(self.true_rewards[context, completion])

    def expected_reward_under_policy(self, policy_probs: np.ndarray) -> float:
        """E_{x ~ uniform, y ~ pi(.|x)} [ r*(x, y) ].

        policy_probs: shape (n_contexts, n_completions), each row sums to 1.
        """
        return float(np.mean(np.sum(policy_probs * self.true_rewards, axis=1)))

    def sample_preference_pair(self, context: int, rng) -> tuple:
        """Pick two distinct completions, sample preference, return
        (winner_idx, loser_idx).
        """
        a, b = rng.choice(self.n_completions, size=2, replace=False)
        r_a = self.true_rewards[context, a]
        r_b = self.true_rewards[context, b]
        p_a_over_b = _sigmoid(r_a - r_b)
        if rng.random() < p_a_over_b:
            return int(a), int(b)  # a wins
        return int(b), int(a)

    def collect_preferences(self, n_samples: int, rng) -> list[tuple]:
        """Sample n preference triples (context, winner, loser)."""
        prefs = []
        for _ in range(n_samples):
            c = int(rng.integers(self.n_contexts))
            winner, loser = self.sample_preference_pair(c, rng)
            prefs.append((c, winner, loser))
        return prefs

    def optimal_policy_probs(self, temperature: float = 1.0) -> np.ndarray:
        """Soft-optimal policy under the true reward and temperature."""
        z = self.true_rewards / temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def greedy_policy(self) -> np.ndarray:
        return self.true_rewards.argmax(axis=1)
