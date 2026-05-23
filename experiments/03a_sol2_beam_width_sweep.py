"""Solution 3a.2: beam width sweep across LMs of varying mis-calibration.

When the LM is well-calibrated (MAP = high reward), beam search helps.
When it's mis-calibrated (MAP has wrong reward), beam search just locks
in the bad answer harder.

Sweep beam width B from 1 to 50 across three different LMs (different
random seeds, different mis-calibration levels) and plot reward of the
top-B beam.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM, all_beams


def main() -> None:
    # Three LMs from different seeds — they have different MAP/reward alignment.
    configs = [
        ("seed=0 (MAP = good)",       0, 0.5),
        ("seed=2 (MAP collapses)",    2, 0.8),
        ("seed=4 (well-calibrated)",  4, 0.6),
    ]
    Bs = [1, 2, 5, 10, 20, 50, 200]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = ["#3a7ebf", "#c44e52", "#55a467"]

    for (label, seed, noise), color in zip(configs, colors):
        lm = TinyMarkovLM(vocab_size=8, seq_length=4, noise=noise, seed=seed)
        # Top-1 reward (MAP) — what you'd actually use.
        # Best-in-beam reward — what reranking by reward could find.
        top1_r = []
        best_r = []
        for B in Bs:
            beams = all_beams(lm, beam_width=B)
            top1_r.append(lm.reward(beams[0][1]))
            best_r.append(max(lm.reward(seq) for (_, seq) in beams))
        axes[0].plot(Bs, top1_r, "o-", color=color, linewidth=2,
                      markersize=9, label=label)
        axes[1].plot(Bs, best_r, "o-", color=color, linewidth=2,
                      markersize=9, label=label)
        best_reachable = max(lm.reward(s) for s in lm.all_sequences())
        print(f"  {label}: best reachable = {best_reachable}, "
              f"MAP (B=1) = {top1_r[0]}, best-in-beam-200 = {best_r[-1]}")

    for ax, title in zip(axes,
                          ["Top-1 reward (the MAP)",
                           "Best reward found anywhere in beam"]):
        ax.set_xscale("log")
        ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
        ax.set_xlabel("beam width B")
        ax.set_ylabel("reward")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)
        ax.set_ylim(-0.05, 1.10)

    fig.suptitle(
        "Beam width affects MAP only when the LM is well-calibrated; reranking "
        "by reward (right) lets a wider beam find good sequences regardless.",
        fontsize=12, y=1.02,
    )

    fig.tight_layout()
    out = Path("figures/03a_sol2_beam_width_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
