"""Figure: self-consistency as Monte Carlo, done correctly.

The earlier toy setup voted over entire sequences, so voting just converged
to MAP. The realistic case is: many distinct chains of reasoning lead to
the same final answer. We model that by defining the "answer" as the
last token only, with reward depending on whether the answer is "correct"
(in a small set of correct answers).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM


def main() -> None:
    lm = TinyMarkovLM(vocab_size=8, seq_length=5, noise=0.6, seed=4)
    correct_answers = {seq[-1] for seq in lm.good_sequences}

    def answer_of(seq):
        return seq[-1]

    def is_correct(seq):
        return answer_of(seq) in correct_answers

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # ---- Left: histogram of answers from N=50 samples at T=1 ----
    ax = axes[0]
    N = 50
    rng = np.random.default_rng(0)
    answers = [answer_of(lm.sample_sequence(rng, T=1.0)) for _ in range(N)]
    counts = Counter(answers)
    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]
    colors_per = ["#55a467" if k in correct_answers else "#aaaaaa"
                  for k in labels]
    bars = ax.bar([str(k) for k in labels], values, color=colors_per,
                  edgecolor="white", linewidth=1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha="center", fontsize=10)
    ax.set_xlabel("answer (last token)")
    ax.set_ylabel(f"count out of N = {N} samples")
    ax.set_title("Vote over answers at T = 1")
    winner = max(counts.keys(), key=lambda k: counts[k])
    ax.text(0.02, 0.95,
            f"winner: token {winner} -> "
            f"{'CORRECT' if winner in correct_answers else 'WRONG'}",
            transform=ax.transAxes, fontsize=11,
            color="#55a467" if winner in correct_answers else "#c44e52",
            fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#55a467", label="correct answer"),
                       Patch(facecolor="#aaaaaa", label="wrong answer")]
    ax.legend(handles=legend_elements, loc="upper right")

    # ---- Right: accuracy vs N at several temperatures ----
    ax = axes[1]
    N_values = [1, 2, 4, 8, 16, 32, 64, 128]
    n_runs = 300
    Ts = [0.7, 1.0, 1.5]
    colors_T = ["#3a7ebf", "#55a467", "#dd8452"]

    for T, c in zip(Ts, colors_T):
        acc = np.zeros(len(N_values))
        for i, N in enumerate(N_values):
            for trial in range(n_runs):
                rng = np.random.default_rng(trial * 1000 + i)
                votes = Counter(
                    answer_of(lm.sample_sequence(rng, T=T))
                    for _ in range(N))
                w = max(votes.keys(), key=lambda k: votes[k])
                if w in correct_answers:
                    acc[i] += 1
            acc[i] /= n_runs
        ax.plot(N_values, acc, "o-", color=c, linewidth=2,
                markersize=9, label=f"T = {T}")

    # Reference: single-sample accuracy at T=1.
    p_single = float(np.mean(
        [is_correct(lm.sample_sequence(np.random.default_rng(k), T=1.0))
         for k in range(3000)]))
    ax.axhline(p_single, color="#888", linestyle=":", linewidth=1.5,
               label=f"single sample @ T=1: accuracy = {p_single:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_values); ax.set_xticklabels([str(n) for n in N_values])
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("P(majority-vote answer is correct)")
    ax.set_title(f"Voting accuracy vs N ({n_runs} trials per point)")
    ax.legend(loc="lower right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Self-consistency over answers. Test-time compute -> higher accuracy.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/03a_self_consistency.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
