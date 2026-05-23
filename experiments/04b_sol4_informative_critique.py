"""Solution 4b.4: does the critique actually inform the revision?

A subtle question: when reflection helps, is it because the critic *catches*
errors (and a blind resample fixes some), or because the critique *localizes*
the error so the revision is genuinely better? We sweep `revise_gain` — the
accuracy boost a revision gets from an informative critique — and compare
reflection against a best-of-N baseline that uses the same critic only to
rerank (no informative revision).

At revise_gain = 0 (blind resampling) reflection's advantage over best-of-N
is small; as the critique becomes informative, sequential reflection pulls
clearly ahead, because each revision is targeted rather than a fresh guess.
The lesson: the value of reflection lives in the *content* of the critique,
not merely in the act of trying again.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import (
    ReflectiveQA,
    accuracy,
    best_of_n,
    reflect,
    single_shot,
)


def main() -> None:
    gains = np.linspace(0.0, 0.5, 11)
    a = 0.35
    refl, bon = [], []
    base = None
    for g in gains:
        env = ReflectiveQA(n_problems=1500, n_answers=5, gen_accuracy=a,
                           detect_rate=0.75, false_alarm=0.2,
                           revise_gain=float(g), seed=0)
        if base is None:
            base = accuracy(env, single_shot, n_trials=5, seed=1)
        refl.append(accuracy(env, reflect, n_trials=5, seed=1, max_rounds=4))
        # best-of-N reranks but does not benefit from informative revision
        bon.append(accuracy(env, best_of_n, n_trials=5, seed=1, n=5))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(gains, refl, "o-", color="#c44e52", linewidth=2, markersize=7,
            label="reflection (uses critique to revise)")
    ax.plot(gains, bon, "s-", color="#dd8452", linewidth=2, markersize=7,
            label="best-of-5 (reranks only, no informative revision)")
    ax.axhline(base, color="#888", linestyle="--", linewidth=1.5,
               label=f"single-shot ({base:.2f})")

    ax.set_xlabel("revise_gain  (how much an informative critique boosts the revision)")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "The value of reflection is in the content of the critique.\n"
        "Blind resampling (gain 0) barely beats reranking; informative critiques pull ahead.",
        fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    print(f"  reflect {refl[0]:.2f}->{refl[-1]:.2f}, bon ~{np.mean(bon):.2f}")
    fig.tight_layout()
    out = Path("figures/04b_sol4_informative_critique.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
