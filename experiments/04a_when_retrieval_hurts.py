"""Figure: when retrieval HURTS.

Retrieval is not free insurance. If the model already knows the answer
(high parametric knowledge) and the retriever is poor, the retrieved
documents inject misleading evidence that overrides correct parametric
knowledge. We sweep retriever quality for a knowledgeable model and show
the crossover: below some retrieval quality, open-book is WORSE than
closed-book.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, answer_accuracy


def main() -> None:
    relevances = np.linspace(0.15, 0.9, 16)
    n_trials = 25
    knowledge_levels = [(0.9, "#3a7ebf", "knowledgeable model (90%)"),
                        (0.5, "#dd8452", "partial knowledge (50%)")]

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for kn, color, label in knowledge_levels:
        # Closed-book baseline (no retrieval) — independent of relevance.
        env0 = RetrievalQA(n_queries=80, n_answers=5, n_relevant=3,
                           n_distractors=250, relevance=0.7, embed_dim=16,
                           seed=0)
        closed = answer_accuracy(env0, knowledge=kn, k=0, n_trials=n_trials,
                                  use_retrieval=False, seed=0)
        ax.axhline(closed, color=color, linestyle="--", linewidth=1.8,
                    alpha=0.7)

        openbook = []
        for rel in relevances:
            env = RetrievalQA(n_queries=80, n_answers=5, n_relevant=3,
                              n_distractors=250, relevance=float(rel),
                              embed_dim=16, seed=0)
            openbook.append(answer_accuracy(env, knowledge=kn, k=3,
                                             evidence_strength=2.0,
                                             n_trials=n_trials, seed=0))
        openbook = np.array(openbook)
        ax.plot(relevances, openbook, "o-", color=color, linewidth=2,
                markersize=7, label=label)

        # Mark the crossover where open-book overtakes closed-book.
        crossings = np.where(np.diff(np.sign(openbook - closed)))[0]
        if len(crossings):
            ci = crossings[0]
            ax.scatter([relevances[ci + 1]], [closed], s=150, color=color,
                        marker="X", zorder=5, edgecolor="white", linewidth=1.5)

    ax.set_xlabel("retriever quality (relevance blend)")
    ax.set_ylabel("answer accuracy")
    ax.set_title(
        "Retrieval can hurt. Dashed = closed-book baseline. With a poor "
        "retriever,\nopen-book (solid) falls BELOW closed-book — bad evidence "
        "overrides good priors.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = Path("figures/04a_when_retrieval_hurts.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
