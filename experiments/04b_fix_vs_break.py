"""Figure: reflection = fixing wrong answers minus breaking right ones.

The net effect of one reflection round decomposes into two competing terms:

    fix   = (1 - a) * d * a'     # detect a wrong answer and revise it correctly
    break = a * f * (1 - a')     # false-alarm a correct answer and botch it

We plot both against the generator accuracy a (fixed critic). When the model
is mostly wrong (low a) the fix term dominates and reflection is a clear win;
when the model is mostly right (high a) the break term can overtake it. The
net curve is the accuracy change — positive in the middle-left, shrinking and
eventually negative as a -> 1 if the critic ever false-alarms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import break_term, fix_term, reflection_delta


def main() -> None:
    accs = np.linspace(0.0, 1.0, 101)
    d, f = 0.75, 0.30      # a decent but imperfect critic
    a_prime = accs          # blind revision: revision accuracy == generator a

    fixes = np.array([fix_term(a, d, ap) for a, ap in zip(accs, a_prime)])
    breaks = np.array([break_term(a, f, ap) for a, ap in zip(accs, a_prime)])
    net = fixes - breaks

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(accs, fixes, "-", color="#55a467", linewidth=2,
            label="fix rate  $(1{-}a)\\,d\\,a'$  (wrong → right)")
    ax.plot(accs, breaks, "-", color="#c44e52", linewidth=2,
            label="break rate  $a\\,f\\,(1{-}a')$  (right → wrong)")
    ax.plot(accs, net, "-", color="#3a7ebf", linewidth=2.5,
            label="net change (fix − break)")
    ax.axhline(0, color="#888", linewidth=1)

    # Mark where net crosses zero (reflection stops helping).
    cross = accs[np.argmin(np.abs(net[1:-1]))]
    ax.fill_between(accs, 0, net, where=(net > 0), color="#55a467", alpha=0.12)
    ax.fill_between(accs, 0, net, where=(net < 0), color="#c44e52", alpha=0.12)

    ax.set_xlabel("generator accuracy $a$  (= revision accuracy here)")
    ax.set_ylabel("rate / accuracy change")
    ax.set_title(
        "Reflection = fixing wrong answers − breaking right ones (critic d=0.75, f=0.30).\n"
        "The win is largest when the model is mostly wrong; it shrinks as a → 1.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    out = Path("figures/04b_fix_vs_break.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}  (net crosses ~{cross:.2f})")


if __name__ == "__main__":
    main()
