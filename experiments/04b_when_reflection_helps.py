"""Figure: when does self-reflection help? Accuracy vs generator accuracy.

Reflection helps most when the model is moderately wrong, and only when the
critic discriminates. A good critic (high detection, low false alarm) lifts
accuracy across the board; an anti-discriminating critic (false alarm >
detection) drags it *below* the single-shot baseline — the model talks
itself out of correct answers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import ReflectiveQA, accuracy, reflect, single_shot
from nano_agents.reflection import one_round_accuracy


def main() -> None:
    accs = np.linspace(0.05, 0.95, 19)
    critics = [
        (0.85, 0.10, "#55a467", "good critic (d=0.85, f=0.10)"),
        (0.55, 0.45, "#dd8452", "weak critic (d=0.55, f=0.45)"),
        (0.30, 0.60, "#c44e52", "anti-discriminating (d=0.30, f=0.60)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(accs, accs, "--", color="#888", linewidth=2,
            label="single-shot (no reflection)")

    for d, f, color, label in critics:
        sim = []
        for a in accs:
            env = ReflectiveQA(n_problems=600, n_answers=5, gen_accuracy=float(a),
                               detect_rate=d, false_alarm=f, revise_gain=0.0,
                               seed=0)
            sim.append(accuracy(env, reflect, n_trials=4, seed=1, max_rounds=3))
        ax.plot(accs, sim, "o-", color=color, linewidth=2, markersize=5,
                label=label)

    ax.set_xlabel("generator accuracy $a$ (single-shot)")
    ax.set_ylabel("accuracy after reflection (3 rounds)")
    ax.set_title(
        "When self-reflection helps. A discriminating critic lifts accuracy;\n"
        "an anti-discriminating one drags it below the single-shot diagonal.",
        fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = Path("figures/04b_when_reflection_helps.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
