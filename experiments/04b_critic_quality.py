"""Figure: reflection gain vs critic discrimination (the verification gap).

Self-reflection's value hinges on a single quantity: can the model judge a
candidate better than it can generate one? We sweep the critic's
discrimination (detection minus false-alarm, Youden's J) and plot the
accuracy change from one round of reflection. Below zero discrimination the
critic is worse than a coin flip at telling right from wrong, and reflection
*loses* accuracy. The analytic prediction a(1-a)(d-f) (blind revision) is
overlaid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import (
    ReflectiveQA,
    accuracy,
    reflect,
    reflection_delta,
    single_shot,
)


def main() -> None:
    a = 0.4
    # Vary discrimination by sweeping detection while holding false alarm fixed.
    f = 0.2
    detects = np.linspace(0.0, 1.0, 21)
    disc = detects - f

    sim_delta, analytic_delta = [], []
    base = None
    for d in detects:
        env = ReflectiveQA(n_problems=1500, n_answers=5, gen_accuracy=a,
                           detect_rate=float(d), false_alarm=f, revise_gain=0.0,
                           seed=0)
        if base is None:
            base = accuracy(env, single_shot, n_trials=4, seed=1)
        refl = accuracy(env, reflect, n_trials=4, seed=1, max_rounds=1)
        sim_delta.append(refl - base)
        analytic_delta.append(reflection_delta(a, float(d), f, a_prime=a))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axhline(0, color="#888", linewidth=1.2)
    ax.axvline(0, color="#888", linestyle=":", linewidth=1.5,
               label="zero discrimination (critic = chance)")
    ax.plot(disc, analytic_delta, "-", color="#3a7ebf", linewidth=2,
            label="analytic  $a(1{-}a)(d{-}f)$")
    ax.plot(disc, sim_delta, "o", color="#c44e52", markersize=6,
            label="simulated (1 round)")
    ax.fill_between(disc, 0, analytic_delta, where=(np.array(analytic_delta) > 0),
                    color="#55a467", alpha=0.12)
    ax.fill_between(disc, 0, analytic_delta, where=(np.array(analytic_delta) < 0),
                    color="#c44e52", alpha=0.12)

    ax.annotate("reflection helps", (0.45, 0.02), fontsize=10, color="#2e7d4f")
    ax.annotate("reflection hurts", (-0.18, -0.03), fontsize=10, color="#a03a3e",
                ha="right")
    ax.set_xlabel("critic discrimination  $d - f$  (detection − false alarm)")
    ax.set_ylabel("accuracy change from reflection")
    ax.set_title(
        "Reflection pays off only across the generation–verification gap.\n"
        "If self-judgement can't beat chance (d \u2264 f), reflection loses accuracy.",
        fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/04b_critic_quality.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
