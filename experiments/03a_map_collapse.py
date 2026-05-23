"""Figure: the MAP-collapse phenomenon.

Beam search finds the sequence with the highest joint log-probability.
On most LMs this is NOT the sequence with the highest reward — often
it's a "bland but locally-safe" path that gets partial credit.

We compare the top-K beams against a random sample of sequences and their
rewards, to make the collapse visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM, all_beams


def main() -> None:
    lm = TinyMarkovLM(vocab_size=8, seq_length=4, noise=0.8, seed=2)
    seqs = lm.all_sequences()
    lps = np.array([lm.sequence_log_prob(s) for s in seqs])
    probs = np.exp(lps)
    rewards = np.array([lm.reward(s) for s in seqs])

    # Beam search results: get top 5 beams.
    top_beams = all_beams(lm, beam_width=5)
    beam_seqs = [seq for (_, seq) in top_beams]
    beam_lps = [lp for (lp, _) in top_beams]
    beam_rs = [lm.reward(seq) for seq in beam_seqs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # ---- Left: scatter of log-prob vs reward over ALL sequences ----
    ax = axes[0]
    # Color by reward for visual clarity.
    colors_all = ["#aaaaaa" if r == 0 else "#dd8452" if r < 0.5 else "#55a467"
                   for r in rewards]
    ax.scatter(lps, rewards + np.random.default_rng(0).normal(0, 0.02, len(lps)),
                s=22, c=colors_all, alpha=0.5,
                edgecolor="white", linewidth=0.4,
                label="all sequences (jittered)")

    # Highlight top beams.
    ax.scatter(beam_lps, beam_rs, s=180, c="#c44e52", marker="D",
                zorder=5, edgecolor="white", linewidth=2,
                label="top-5 beams")
    for (lp, r, seq) in zip(beam_lps, beam_rs, beam_seqs):
        ax.annotate(f"  {seq}", (lp, r), fontsize=8, color="#c44e52",
                     ha="left", va="center")

    # Mark the true reward-1 sequences.
    for g in lm.good_sequences:
        # Find its index
        idx = seqs.index(g)
        ax.scatter([lps[idx]], [rewards[idx]], s=180, c="#3a7ebf",
                    marker="*", zorder=6, edgecolor="white", linewidth=1.5,
                    label="reward = 1 sequences" if g == lm.good_sequences[0] else None)
        ax.annotate(f"  {g}", (lps[idx], rewards[idx]), fontsize=8,
                     color="#3a7ebf", ha="left", va="center")

    ax.set_xlabel("sequence log probability")
    ax.set_ylabel("reward (with vertical jitter)")
    ax.set_title("All sequences in the space, by log-prob and reward")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # ---- Right: beam search vs sampling vs greedy ----
    ax = axes[1]
    n_samples = 1000
    rng = np.random.default_rng(0)
    rewards_sampled = np.array(
        [lm.reward(lm.sample_sequence(rng, T=1.0)) for _ in range(n_samples)])
    rng = np.random.default_rng(0)
    rewards_greedy = np.array(
        [lm.reward(lm.sample_sequence(rng, T=0.01)) for _ in range(n_samples)])

    methods = ["greedy\n(T→0)", "sample\n(T=1)", "beam-1\n(=greedy)",
                "beam-3", "beam-10", "beam-50"]
    mean_r = [float(np.mean(rewards_greedy)),
              float(np.mean(rewards_sampled)),
              float(lm.reward(all_beams(lm, beam_width=1)[0][1])),
              float(lm.reward(all_beams(lm, beam_width=3)[0][1])),
              float(lm.reward(all_beams(lm, beam_width=10)[0][1])),
              float(lm.reward(all_beams(lm, beam_width=50)[0][1]))]
    colors_bar = ["#dd8452", "#3a7ebf", "#c44e52", "#c44e52", "#c44e52", "#c44e52"]
    bars = ax.bar(methods, mean_r, color=colors_bar,
                    edgecolor="white", linewidth=1.5, alpha=0.85)
    for bar, v in zip(bars, mean_r):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=10)
    # Reference: best reachable reward.
    ax.axhline(1.0, color="#222", linestyle="--", linewidth=1,
                alpha=0.5, label="best possible reward")
    ax.set_ylabel("reward")
    ax.set_title("All methods stuck at the MAP — but sampling lets us aggregate")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle(
        "The MAP collapse. High-log-prob sequences are not necessarily "
        "high-reward — beam search can miss the reward-1 sequences entirely.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/03a_map_collapse.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
