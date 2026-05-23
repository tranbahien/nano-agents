"""Solution 3a.3: top-k / top-p interaction with self-consistency.

We've seen that:
  - High T helps diversity (-> better self-consistency upper bound)
  - But high T degrades single-sample accuracy

Top-k and top-p offer a different way to trade these off: keep T at a
moderate value but TRUNCATE the long tail of unlikely (probably wrong)
tokens. Compare:
  - Pure temperature: T = 1.0, full vocab
  - Top-k = 3 at T = 1.0
  - Top-p = 0.9 at T = 1.0
  - Top-p = 0.7 at T = 1.0
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM


def main() -> None:
    lm = TinyMarkovLM(vocab_size=8, seq_length=5, noise=0.6, seed=4)
    correct = {seq[-1] for seq in lm.good_sequences}

    def answer_of(seq):
        return seq[-1]

    configs = [
        ("T=1.0, full vocab",         dict(T=1.0)),
        ("T=1.0, top-k = 3",           dict(T=1.0, top_k=3)),
        ("T=1.0, top-p = 0.9",         dict(T=1.0, top_p=0.9)),
        ("T=1.0, top-p = 0.7",         dict(T=1.0, top_p=0.7)),
    ]
    colors = ["#888888", "#3a7ebf", "#55a467", "#c44e52"]

    N_values = [1, 2, 4, 8, 16, 32, 64]
    n_trials = 250

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (label, kw), c in zip(configs, colors):
        acc = np.zeros(len(N_values))
        for i, N in enumerate(N_values):
            for trial in range(n_trials):
                rng = np.random.default_rng(trial * 100 + i)
                votes = Counter(
                    answer_of(lm.sample_sequence(rng, **kw))
                    for _ in range(N))
                w = max(votes.keys(), key=lambda k: votes[k])
                if w in correct:
                    acc[i] += 1
            acc[i] /= n_trials
        ax.plot(N_values, acc, "o-", color=c, linewidth=2,
                 markersize=9, label=label)
        print(f"  {label}: acc at N=1 = {acc[0]:.2f}, "
              f"N=64 = {acc[-1]:.2f}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_values); ax.set_xticklabels([str(n) for n in N_values])
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("self-consistency accuracy")
    ax.set_title(
        "Top-p truncates the long tail without sacrificing diversity within "
        "the head — best both at N=1 and at higher N.",
        fontsize=11,
    )
    ax.legend(loc="lower right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0.4, 1.05)

    fig.tight_layout()
    out = Path("figures/03a_sol3_top_k_top_p.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
