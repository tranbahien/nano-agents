"""Solution 5a.3: when does specialization beat a generalist?

A task with L subtasks. A generalist handles every subtask at accuracy g, for
end-to-end g**L. A team of specialists handles each subtask better (accuracy
s > g) but must hand off between stages, and each of the L-1 handoffs succeeds
with probability h (lost context, format mismatches, miscommunication):

    specialist pipeline = s**L * h**(L-1).

We sweep the specialist skill advantage s for a fixed handoff reliability and
mark where specialization overtakes the generalist. Division of labor pays
only when the per-subtask skill gain outruns the coordination tax -- which is
why naive "add more specialized agents" often underperforms one capable
generalist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    L = 4
    g = 0.80                       # generalist per-subtask accuracy
    s_vals = np.linspace(0.80, 0.99, 40)
    generalist = g ** L

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axhline(generalist, color="#888", linestyle="--", linewidth=2,
               label=f"generalist  $g^L$ (g={g}, L={L}) = {generalist:.2f}")
    for h, color in [(1.00, "#55a467"), (0.95, "#3a7ebf"), (0.85, "#c44e52")]:
        spec = s_vals ** L * h ** (L - 1)
        ax.plot(s_vals, spec, "-", color=color, linewidth=2,
                label=f"specialists, handoff h={h}")
        # crossover
        cross = s_vals[np.argmin(np.abs(spec - generalist))]
        if spec.max() > generalist > spec.min():
            ax.plot(cross, generalist, "o", color=color, markersize=8)

    ax.set_xlabel("specialist per-subtask accuracy  s")
    ax.set_ylabel("end-to-end task success")
    ax.set_title(
        "When specialization beats a generalist (4 subtasks). Specialists win\n"
        "only when their skill edge outruns the handoff tax $h^{L-1}$.",
        fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/05a_sol3_specialization.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
