"""Solution 3a.1: temperature sweep on a synthetic reasoning task.

Trade-off: single-sample mean reward decreases with T (lower T = follow
the MAP more closely). BUT self-consistency accuracy is maximized at a
*non-zero* T because zero-T sampling gives no diversity to vote over.

Show both curves on the same axes — the gap is the value of test-time
compute.
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

    Ts = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
    n_trials = 500
    N_self_consistency = 16

    single_acc = np.zeros(len(Ts))
    sc_acc = np.zeros(len(Ts))

    for i, T in enumerate(Ts):
        for trial in range(n_trials):
            rng = np.random.default_rng(trial + i * 1000)
            # Single sample.
            s = lm.sample_sequence(rng, T=float(T))
            if s[-1] in correct:
                single_acc[i] += 1
            # Self-consistency (N samples).
            rng2 = np.random.default_rng(trial + i * 1000 + 50000)
            votes = Counter(
                lm.sample_sequence(rng2, T=float(T))[-1]
                for _ in range(N_self_consistency))
            w = max(votes.keys(), key=lambda k: votes[k])
            if w in correct:
                sc_acc[i] += 1
        single_acc[i] /= n_trials
        sc_acc[i] /= n_trials
        print(f"  T = {T:.2f}: single = {single_acc[i]:.2f}, "
              f"SC@{N_self_consistency} = {sc_acc[i]:.2f}")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(Ts, single_acc, "o-", color="#dd8452", linewidth=2,
            markersize=9, label="single sample")
    ax.plot(Ts, sc_acc, "s-", color="#3a7ebf", linewidth=2,
            markersize=9, label=f"self-consistency (N = {N_self_consistency})")
    ax.fill_between(Ts, single_acc, sc_acc, color="#3a7ebf", alpha=0.10,
                     label="benefit of self-consistency")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("accuracy")
    ax.set_title(
        f"Temperature sweep: single sample vs self-consistency.\n"
        f"Self-consistency benefits from T > 0; single sample peaks at T → 0.",
        fontsize=11,
    )
    ax.legend(loc="best"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = Path("figures/03a_sol1_temperature_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
