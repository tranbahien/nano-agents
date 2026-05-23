"""Figure: temperature controls the next-token distribution.

Same logits, four different temperatures:
  - T = 0.5: sharpens (concentration on top)
  - T = 1.0: the model's own distribution
  - T = 2.0: flattens
  - T = 5.0: nearly uniform

Right panel: sample 100 sequences at each temperature, plot the distribution
over the resulting "answer" (last token), and overlay the mean reward.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM


def main() -> None:
    lm = TinyMarkovLM(vocab_size=8, seq_length=4, seed=0)

    # Use the conditional distribution from the BOS as our running example.
    cur = lm.BOS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: distribution at different temperatures.
    ax = axes[0]
    Ts = [0.5, 1.0, 2.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(Ts)))
    width = 0.18
    x = np.arange(lm.V)
    for i, (T, c) in enumerate(zip(Ts, colors)):
        p = lm.conditional_probs(cur, T=T)
        ax.bar(x + i * width - 1.5 * width, p, width=width,
                color=c, edgecolor="white", linewidth=0.5,
                label=f"T = {T}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{i}" for i in range(lm.V)])
    ax.set_xlabel("token")
    ax.set_ylabel("$P(\\mathrm{next\\ token}\\,\\mid\\,\\mathrm{BOS}, T)$")
    ax.set_title("Next-token distribution at four temperatures")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_axisbelow(True)

    # Right: mean reward of sampled sequences vs T (logspace).
    ax = axes[1]
    Ts_sweep = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    n_samples = 600
    mean_rewards = []
    for T in Ts_sweep:
        rng = np.random.default_rng(0)
        rs = [lm.reward(lm.sample_sequence(rng, T=T)) for _ in range(n_samples)]
        mean_rewards.append(float(np.mean(rs)))
    # Reference: greedy reward (T=0).
    rng = np.random.default_rng(0)
    greedy = lm.sample_sequence(rng, T=0.001)
    greedy_r = lm.reward(greedy)
    # Reference: oracle expected reward of distribution at T=1 (computed exactly).
    seqs = lm.all_sequences()
    lps = np.array([lm.sequence_log_prob(s) for s in seqs])
    probs = np.exp(lps)
    rewards = np.array([lm.reward(s) for s in seqs])
    exact_T1 = float((probs * rewards).sum())

    ax.plot(Ts_sweep, mean_rewards, "o-", color="#3a7ebf", linewidth=2,
             markersize=8)
    ax.axhline(greedy_r, color="#c44e52", linestyle="--",
                linewidth=1.5, label=f"greedy (T→0): reward = {greedy_r}")
    ax.axhline(exact_T1, color="#888", linestyle=":",
                linewidth=1.5, label=f"exact E[r] at T=1: {exact_T1:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("mean reward over samples")
    ax.set_title(f"Mean reward vs T ({n_samples} samples per point)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle(
        "Temperature controls a tradeoff. Low T = high mean reward but low "
        "diversity. High T = explore at the cost of average quality.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/03a_temperature.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
