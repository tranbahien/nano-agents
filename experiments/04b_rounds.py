"""Figure: more reflection is not always better.

Accuracy as a function of the number of critique-revise rounds. With a good
critic, accuracy rises and plateaus (diminishing returns — each round only
helps the answers still flagged). With a high-false-alarm critic, accuracy
*declines* with more rounds: the critic keeps flagging correct answers and
the reviser keeps replacing them, so the agent talks itself out of right
answers. Over-iteration is a real failure mode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import ReflectiveQA, accuracy, reflect


def main() -> None:
    rounds = list(range(0, 9))
    configs = [
        (0.85, 0.10, "#55a467", "good critic (d=0.85, f=0.10)"),
        (0.70, 0.30, "#3a7ebf", "decent critic (d=0.70, f=0.30)"),
        (0.50, 0.50, "#dd8452", "uninformative (d=0.50, f=0.50)"),
        (0.40, 0.55, "#c44e52", "over-critical (d=0.40, f=0.55)"),
    ]
    a = 0.4

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for d, f, color, label in configs:
        env = ReflectiveQA(n_problems=1200, n_answers=5, gen_accuracy=a,
                           detect_rate=d, false_alarm=f, revise_gain=0.0, seed=0)
        ys = [accuracy(env, reflect, n_trials=5, seed=1, max_rounds=r)
              for r in rounds]
        ax.plot(rounds, ys, "o-", color=color, linewidth=2, markersize=6,
                label=label)

    ax.axhline(a, color="#888", linestyle="--", linewidth=1.5,
               label=f"single-shot baseline ({a})")
    ax.set_xlabel("number of critique–revise rounds")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "More reflection is not always better. A good critic plateaus;\n"
        "an over-critical one degrades — the model revises correct answers away.",
        fontsize=11)
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(rounds)

    fig.tight_layout()
    out = Path("figures/04b_rounds.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
