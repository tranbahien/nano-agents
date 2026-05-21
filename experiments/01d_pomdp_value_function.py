"""Figure: Optimal value function V*(b) and policy π*(b) for the Tiger POMDP.

V*(b) is piecewise linear and convex in belief (Sondik 1971). With a fine
belief grid the discretized solver recovers this shape clearly.

We plot:
  - V*(b) as a function of b(TL), with the three action-conditional Q values.
  - The optimal action as a function of belief, highlighting the three regions:
    open right (b small), listen (b moderate), open left (b large).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.pomdp import TigerPOMDP, discretized_pomdp_value_iteration


def main() -> None:
    pomdp = TigerPOMDP(listen_accuracy=0.85)
    gamma = 0.95
    V, pi, b_grid, hist = discretized_pomdp_value_iteration(
        pomdp, n_belief=401, gamma=gamma, tol=1e-9, max_iters=10000)

    # Re-derive Q(b, a) from V for the figure.
    # Need to recompute Rb and T inside this function — easier to redo here.
    P, Z, R = pomdp.P, pomdp.Z, pomdp.R
    N = len(b_grid)
    Rb = np.zeros((N, pomdp.nA))
    T = np.zeros((N, pomdp.nA, N))
    for i, b0 in enumerate(b_grid):
        b = np.array([b0, 1 - b0])
        for a in range(pomdp.nA):
            Rb[i, a] = b @ R[:, a]
            pred = b @ P[:, a, :]
            for o in range(pomdp.nO):
                p_o = float((pred * Z[:, a, o]).sum())
                if p_o <= 0:
                    continue
                posterior = Z[:, a, o] * pred / p_o
                j = max(0, min(N - 1, int(round(posterior[0] * (N - 1)))))
                T[i, a, j] += p_o
    Q = Rb + gamma * (T @ V)

    action_names = ["Open Left", "Open Right", "Listen"]
    action_colors = {0: "#c44e52", 1: "#dd8452", 2: "#3a7ebf"}

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5),
                              gridspec_kw={"height_ratios": [3, 1]},
                              sharex=True)

    # Top: Q(b, a) and V*(b).
    ax = axes[0]
    for a in range(pomdp.nA):
        ax.plot(b_grid, Q[:, a], color=action_colors[a], linewidth=1.5,
                alpha=0.7, label=rf"$Q^\star(b, \text{{{action_names[a]}}})$")
    ax.plot(b_grid, V, color="#222", linewidth=2.5, linestyle="--",
            label=r"$V^\star(b) = \max_a Q^\star(b, a)$")
    ax.set_ylabel("value")
    ax.set_title(
        r"Tiger POMDP: optimal value $V^\star(b)$ is piecewise linear convex in belief"
        f"  (γ = {gamma}, listen accuracy = {pomdp.listen_accuracy})",
        fontsize=12,
    )
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # Bottom: optimal policy as a function of belief.
    ax = axes[1]
    # Make a color band for each region.
    pi_arr = np.array(pi)
    # Find transitions.
    cuts = np.where(np.diff(pi_arr) != 0)[0]
    edges = np.concatenate([[0], cuts + 1, [N]])
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        a = int(pi_arr[lo])
        ax.axvspan(b_grid[lo], b_grid[min(hi, N - 1)],
                    color=action_colors[a], alpha=0.35)
    # Annotate regions.
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        a = int(pi_arr[lo])
        mid = (b_grid[lo] + b_grid[min(hi, N - 1)]) / 2
        ax.text(mid, 0.5, action_names[a], ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(r"$b(\mathrm{TL}) = P(\mathrm{tiger\ is\ left})$")
    ax.set_title("Optimal policy by belief region", fontsize=11)

    fig.tight_layout()
    out = Path("figures/01d_pomdp_value_function.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
