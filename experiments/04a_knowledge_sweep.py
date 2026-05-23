"""Figure: accuracy vs parametric knowledge, closed-book vs open-book.

The central result of the post. Sweep the model's parametric knowledge
(how much it has memorized) and compare:
  - closed-book: answer from the prior alone.
  - open-book (RAG): condition on retrieved documents.

Retrieval helps most when parametric knowledge is LOW — exactly the regime
where the prior is uninformative and evidence dominates the posterior.
When the model already knows everything, retrieval adds little.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, answer_accuracy


def main() -> None:
    env = RetrievalQA(n_queries=80, n_answers=5, n_relevant=3,
                      n_distractors=250, relevance=0.75, embed_dim=16, seed=0)

    knowledge_levels = np.linspace(0, 1, 11)
    n_trials = 20

    closed = np.zeros(len(knowledge_levels))
    openbook = np.zeros(len(knowledge_levels))
    for i, kn in enumerate(knowledge_levels):
        closed[i] = answer_accuracy(env, knowledge=kn, k=0, n_trials=n_trials,
                                     use_retrieval=False, seed=i)
        openbook[i] = answer_accuracy(env, knowledge=kn, k=3, n_trials=n_trials,
                                       evidence_strength=2.0, seed=i)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(knowledge_levels, closed, "o-", color="#dd8452", linewidth=2,
            markersize=9, label="closed-book (prior only)")
    ax.plot(knowledge_levels, openbook, "s-", color="#3a7ebf", linewidth=2,
            markersize=9, label="open-book / RAG (prior + retrieval)")
    ax.fill_between(knowledge_levels, closed, openbook, color="#3a7ebf",
                     alpha=0.10, label="value of retrieval")

    # Random-guess baseline.
    ax.axhline(1.0 / env.n_answers, color="#888", linestyle=":", linewidth=1.5,
                label=f"random guess (1/{env.n_answers})")

    ax.set_xlabel("parametric knowledge (fraction of queries memorized)")
    ax.set_ylabel("answer accuracy")
    ax.set_title(
        "Retrieval helps most when the model knows least.\n"
        "The gap (shaded) shrinks as parametric knowledge grows.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = Path("figures/04a_knowledge_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
