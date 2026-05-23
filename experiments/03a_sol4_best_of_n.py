"""Solution 3a.4: best-of-N (BoN) sampling with a verifier.

Instead of picking the most-frequent answer (self-consistency), suppose we
have access to an external 'verifier' that scores each candidate sequence.
The verifier can be:
  - A noisy oracle (correct most of the time but not always),
  - A learned reward model,
  - An actual unit test / executable check,
  - Another LLM scoring the candidate.

BoN: sample N candidates, return the one the verifier ranks highest.
We compare two verifiers:
  1. Perfect oracle (reward function itself)
  2. Noisy oracle (correct with probability 1 - flip_rate)

The latter is realistic: real reward models are imperfect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM


def main() -> None:
    lm = TinyMarkovLM(vocab_size=8, seq_length=5, noise=0.6, seed=4)
    correct = {seq[-1] for seq in lm.good_sequences}

    def is_correct(seq):
        return seq[-1] in correct

    def perfect_verifier(seq):
        return 1.0 if is_correct(seq) else 0.0

    def noisy_verifier(seq, flip_rate, rng):
        true_v = 1.0 if is_correct(seq) else 0.0
        if rng.random() < flip_rate:
            return 1.0 - true_v
        return true_v

    N_values = [1, 2, 4, 8, 16, 32, 64]
    n_trials = 300

    verifier_configs = [
        ("perfect verifier",              0.0,  "#3a7ebf"),
        ("noisy verifier (10% flip)",     0.1,  "#55a467"),
        ("noisy verifier (30% flip)",     0.3,  "#dd8452"),
        ("self-consistency (no verifier)", None, "#c44e52"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, flip, color in verifier_configs:
        acc = np.zeros(len(N_values))
        for i, N in enumerate(N_values):
            for trial in range(n_trials):
                rng = np.random.default_rng(trial * 100 + i)
                candidates = [
                    lm.sample_sequence(rng, T=1.0) for _ in range(N)]
                if flip is None:
                    # Self-consistency: majority vote on answer.
                    from collections import Counter
                    votes = Counter(s[-1] for s in candidates)
                    winner_ans = max(votes.keys(),
                                       key=lambda k: votes[k])
                    if winner_ans in correct:
                        acc[i] += 1
                else:
                    # BoN: score each candidate, pick highest.
                    if flip == 0.0:
                        scores = [perfect_verifier(s) for s in candidates]
                    else:
                        srng = np.random.default_rng(
                            trial * 100 + i + 999999)
                        scores = [noisy_verifier(s, flip, srng)
                                  for s in candidates]
                    best = candidates[int(np.argmax(scores))]
                    if is_correct(best):
                        acc[i] += 1
            acc[i] /= n_trials
        ax.plot(N_values, acc, "o-", color=color, linewidth=2,
                 markersize=9, label=label)
        print(f"  {label}: acc at N=1 = {acc[0]:.2f}, "
              f"N=64 = {acc[-1]:.2f}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_values); ax.set_xticklabels([str(n) for n in N_values])
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Best-of-N vs self-consistency. Verifier quality matters — "
        "a noisy verifier (30% flip) is barely better than no verifier.",
        fontsize=11,
    )
    ax.legend(loc="lower right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0.4, 1.05)

    fig.tight_layout()
    out = Path("figures/03a_sol4_best_of_n.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
