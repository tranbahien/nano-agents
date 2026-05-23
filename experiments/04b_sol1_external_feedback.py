"""Solution 4b.1: external feedback vs pure self-critique.

The cleanest way to make reflection reliable is to replace the noisy
self-critic with an *external* verifier — a unit test, a compiler, a search
result, a calculator. We compare self-critique against a perfect external
verifier (detect=1, false-alarm=0) as the round budget grows. The external
verifier never breaks a correct answer and reliably catches wrong ones, so
its accuracy climbs monotonically toward 1; pure self-critique plateaus far
below, capped by its own discrimination. This is why the strongest
"self-correcting" systems are really tool-grounded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import (
    ReflectiveQA,
    accuracy,
    reflect,
    reflect_with_oracle,
)


def main() -> None:
    rounds = list(range(0, 9))
    a = 0.35
    env = ReflectiveQA(n_problems=1500, n_answers=5, gen_accuracy=a,
                       detect_rate=0.7, false_alarm=0.25, revise_gain=0.0, seed=0)

    self_crit = [accuracy(env, reflect, n_trials=5, seed=1, max_rounds=r)
                 for r in rounds]
    oracle = [accuracy(env, reflect_with_oracle, n_trials=5, seed=1, max_rounds=r)
              for r in rounds]
    # Theoretical oracle ceiling: 1 - (1-a)^(r+1) with blind revision.
    oracle_theory = [1 - (1 - a) ** (r + 1) for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(rounds, oracle, "o-", color="#3a7ebf", linewidth=2, markersize=6,
            label="external verifier (tool / test)")
    ax.plot(rounds, oracle_theory, ":", color="#3a7ebf", linewidth=1.5,
            label="oracle theory  $1-(1-a)^{r+1}$")
    ax.plot(rounds, self_crit, "s-", color="#c44e52", linewidth=2, markersize=6,
            label="pure self-critique (d=0.70, f=0.25)")
    ax.axhline(a, color="#888", linestyle="--", linewidth=1.5,
               label=f"single-shot ({a})")

    ax.set_xlabel("round budget")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "External feedback makes reflection reliable; self-critique plateaus.\n"
        "A verifier that never false-alarms climbs toward 1; self-judgement is capped.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(rounds); ax.set_ylim(0, 1.02)

    print(f"  self-critique plateau ~{self_crit[-1]:.2f}, oracle ~{oracle[-1]:.2f}")
    fig.tight_layout()
    out = Path("figures/04b_sol1_external_feedback.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
