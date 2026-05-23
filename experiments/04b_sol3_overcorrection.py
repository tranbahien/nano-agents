"""Solution 4b.3: the over-correction failure mode, traced.

With an over-critical critic (false alarm > detection), reflection erodes
initially-correct answers faster than it fixes wrong ones. We track, round
by round, the fraction of problems that are correct, split by whether the
*first* answer was correct or wrong. The "started correct" curve falls — the
critic keeps flagging good answers and the reviser keeps replacing them —
while the "started wrong" curve barely rises. The net is a loss: the model
is talking itself out of answers it had right.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import ReflectiveQA


def trace(env, max_rounds, n_trials, seed):
    """Run reflection, tracking correctness over rounds split by initial state."""
    rng = np.random.default_rng(seed)
    # accumulate P(correct at round r | started correct / started wrong)
    started_correct = np.zeros(max_rounds + 1)
    started_wrong = np.zeros(max_rounds + 1)
    n_sc = n_sw = 0
    for _ in range(n_trials):
        for p in range(env.n_problems):
            ans = env.generate(p, rng)
            init_correct = env.is_correct(p, ans)
            traj = []
            cur = ans
            accepted = False
            for r in range(max_rounds + 1):
                traj.append(env.is_correct(p, cur))
                if r == max_rounds:
                    break
                if accepted:
                    continue
                if env.critic(p, cur, rng):       # flagged -> revise
                    cur = env.revise(p, rng)
                else:
                    accepted = True
            traj = np.array(traj, dtype=float)
            if init_correct:
                started_correct += traj; n_sc += 1
            else:
                started_wrong += traj; n_sw += 1
    return started_correct / max(n_sc, 1), started_wrong / max(n_sw, 1)


def main() -> None:
    max_rounds = 8
    a = 0.4
    env = ReflectiveQA(n_problems=1200, n_answers=5, gen_accuracy=a,
                       detect_rate=0.40, false_alarm=0.55, revise_gain=0.0, seed=0)
    sc, sw = trace(env, max_rounds, n_trials=4, seed=1)
    rounds = np.arange(max_rounds + 1)
    overall = a * sc + (1 - a) * sw

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(rounds, sc, "o-", color="#55a467", linewidth=2, markersize=6,
            label="P(correct) | started correct")
    ax.plot(rounds, sw, "s-", color="#c44e52", linewidth=2, markersize=6,
            label="P(correct) | started wrong")
    ax.plot(rounds, overall, "^-", color="#3a7ebf", linewidth=2.5, markersize=6,
            label="overall accuracy")
    ax.axhline(a, color="#888", linestyle="--", linewidth=1.5,
               label=f"single-shot ({a})")

    ax.set_xlabel("critique–revise round")
    ax.set_ylabel("fraction correct")
    ax.set_title(
        "Over-correction (critic d=0.40, f=0.55). Initially-correct answers erode\n"
        "faster than wrong ones are fixed: the model revises right answers away.",
        fontsize=11)
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(rounds); ax.set_ylim(0, 1.02)

    print(f"  started-correct: {sc[0]:.2f} -> {sc[-1]:.2f}; overall {overall[0]:.2f} -> {overall[-1]:.2f}")
    fig.tight_layout()
    out = Path("figures/04b_sol3_overcorrection.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
