"""Figure: retrieval quality — precision and recall.

Left: precision@k and recall@k vs k for a fixed retriever. Classic
tradeoff: small k = high precision, low recall; large k = the reverse.
Right: precision@3 as a function of the retriever's quality (the
`relevance` knob), showing the phase transition from useless to perfect
retrieval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: precision/recall vs k.
    env = RetrievalQA(n_queries=80, n_relevant=5, n_distractors=250,
                      relevance=0.7, embed_dim=16, seed=0)
    ks = [1, 2, 3, 5, 8, 12, 20, 30]
    prec = [np.mean([env.retrieval_precision(q, k) for q in range(env.n_queries)])
            for k in ks]
    rec = [np.mean([env.retrieval_recall(q, k) for q in range(env.n_queries)])
           for k in ks]

    ax = axes[0]
    ax.plot(ks, prec, "o-", color="#3a7ebf", linewidth=2, markersize=8,
            label="precision@k")
    ax.plot(ks, rec, "s-", color="#c44e52", linewidth=2, markersize=8,
            label="recall@k")
    ax.set_xlabel("k (documents retrieved)")
    ax.set_ylabel("rate")
    ax.set_title("Precision vs recall (5 relevant docs per query)")
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(); ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)

    # Right: precision@3 vs retriever quality.
    ax = axes[1]
    relevances = np.linspace(0.1, 0.95, 18)
    p3 = []
    for rel in relevances:
        e = RetrievalQA(n_queries=80, n_relevant=5, n_distractors=250,
                        relevance=float(rel), embed_dim=16, seed=0)
        p3.append(np.mean([e.retrieval_precision(q, 3)
                            for q in range(e.n_queries)]))
    ax.plot(relevances, p3, "o-", color="#55a467", linewidth=2, markersize=7)
    ax.set_xlabel("retriever quality (relevance blend)")
    ax.set_ylabel("precision@3")
    ax.set_title("Retrieval has a quality threshold")
    ax.grid(alpha=0.3); ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)
    ax.axhline(5 / (5 + 250), color="#888", linestyle=":", linewidth=1.5,
                label="random retrieval")
    ax.legend(loc="upper left")

    fig.suptitle(
        "Retrieval quality. Left: the precision-recall tradeoff in k. "
        "Right: below a quality threshold, the retriever is no better than random.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path("figures/04a_retrieval_quality.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
