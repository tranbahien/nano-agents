"""Figure: top-k vs top-p (nucleus) truncation.

Two-panel visualization:
  Left: a representative next-token distribution, with the cuts highlighted
        for top-k = 3 and top-p = 0.9.
  Right: how the effective distribution shape changes across a range of
        contexts (some peaked, some flatter), showing why top-p adapts
        better to varying entropy than fixed-k.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM
from nano_agents.decoding.tinylm import _apply_top_k_top_p


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: visualize cuts on a single distribution ----
    lm = TinyMarkovLM(vocab_size=10, seq_length=3, seed=2)
    # Pick the context where t2 -> ... has an interesting distribution.
    cur = 2
    p_full = lm.conditional_probs(cur, T=1.0)
    # Sort by probability descending for clean plotting.
    order = np.argsort(p_full)[::-1]
    p_sorted = p_full[order]

    ax = axes[0]
    x = np.arange(len(p_sorted))
    bars_full = ax.bar(x - 0.27, p_sorted, width=0.25,
                        color="#aaaaaa", edgecolor="white",
                        linewidth=0.5, label="original")

    # top-k = 3
    p_k = _apply_top_k_top_p(p_full, top_k=3, top_p=None)
    p_k_sorted = p_k[order]
    ax.bar(x, p_k_sorted, width=0.25, color="#3a7ebf",
            edgecolor="white", linewidth=0.5, label="top-k = 3")
    # top-p = 0.9
    p_p = _apply_top_k_top_p(p_full, top_k=None, top_p=0.9)
    p_p_sorted = p_p[order]
    ax.bar(x + 0.27, p_p_sorted, width=0.25, color="#c44e52",
            edgecolor="white", linewidth=0.5, label="top-p = 0.9")

    ax.set_xticks(x)
    ax.set_xticklabels([f"r{i}" for i in range(len(p_sorted))])
    ax.set_xlabel("token (sorted by probability)")
    ax.set_ylabel("probability after truncation + renormalize")
    ax.set_title("Truncation rules on one distribution")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_axisbelow(True)

    # ---- Right: effective entropy under top-k vs top-p across contexts ----
    ax = axes[1]
    n_contexts = 50
    rng = np.random.default_rng(0)
    # Build synthetic distributions over V=10 with varying entropy by
    # mixing a peaked Dirichlet with a flat one.
    V = 10
    Hs = np.zeros(n_contexts)
    H_k3 = np.zeros(n_contexts)
    H_p9 = np.zeros(n_contexts)
    for i in range(n_contexts):
        alpha = 0.1 + (i / n_contexts) * 1.5  # smaller alpha = more peaked
        p = rng.dirichlet(alpha * np.ones(V))
        Hs[i] = -np.sum(p * np.log(p + 1e-30))
        p_k = _apply_top_k_top_p(p, top_k=3, top_p=None)
        H_k3[i] = -np.sum(p_k * np.log(p_k + 1e-30))
        p_p = _apply_top_k_top_p(p, top_k=None, top_p=0.9)
        H_p9[i] = -np.sum(p_p * np.log(p_p + 1e-30))

    order_ctx = np.argsort(Hs)
    ax.plot(Hs[order_ctx], Hs[order_ctx], "-", color="#aaaaaa",
             linewidth=2, label="original entropy")
    ax.plot(Hs[order_ctx], H_k3[order_ctx], "o-", color="#3a7ebf",
             linewidth=1.5, markersize=5,
             label="entropy after top-k=3")
    ax.plot(Hs[order_ctx], H_p9[order_ctx], "s-", color="#c44e52",
             linewidth=1.5, markersize=5,
             label="entropy after top-p=0.9")
    ax.set_xlabel("original distribution entropy")
    ax.set_ylabel("entropy after truncation")
    ax.set_title("Top-p tracks original entropy; top-k clips it flat")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle(
        "Top-k uses a fixed cardinality; top-p adapts to the local distribution.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/03a_top_k_top_p.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
