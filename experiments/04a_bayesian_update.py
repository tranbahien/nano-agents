"""Figure: RAG as Bayesian conditioning — prior x likelihood = posterior.

Three-panel bar chart for a single query where the model's parametric prior
is wrong (peaked on the wrong answer), but retrieved evidence corrects it.
This is the core conceptual picture of the post.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, posterior


def main() -> None:
    env = RetrievalQA(n_queries=30, n_answers=5, n_relevant=3,
                      n_distractors=150, relevance=0.8, embed_dim=16, seed=1)

    # Find a query where the parametric prior would be WRONG, so retrieval
    # has something to fix. We hand-craft an illustrative prior.
    q = 0
    correct = int(env.correct_answers[q])
    n = env.n_answers
    wrong = (correct + 2) % n

    # A confidently-wrong prior (the model "misremembers").
    prior = np.full(n, 0.05)
    prior[wrong] = 0.6
    prior[correct] = 0.15
    prior = prior / prior.sum()

    # Retrieve documents and form the posterior.
    docs = env.retrieve(q, k=3)
    post = posterior(env, prior, docs, evidence_strength=2.0)

    # Likelihood contribution: count of retrieved docs supporting each answer.
    support_counts = np.zeros(n)
    for d in docs:
        support_counts[env.doc_supports[d]] += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(n)
    labels = [f"A{i}" for i in range(n)]

    def style(ax, vals, title, color):
        bars = ax.bar(x, vals, color=color, edgecolor="white", linewidth=1)
        # Highlight the correct answer's bar with a border.
        bars[correct].set_edgecolor("#222")
        bars[correct].set_linewidth(2.5)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

    style(axes[0], prior, "Prior  $p_0(a\\,|\\,q)$\n(parametric memory: confidently wrong)",
          "#dd8452")
    axes[0].set_ylabel("probability")
    axes[0].annotate("model's\nguess", (wrong, prior[wrong] + 0.03),
                      ha="center", fontsize=9, color="#c44e52")

    # Middle: likelihood (normalized support counts for display).
    like_display = support_counts / max(support_counts.sum(), 1)
    style(axes[1], like_display,
          "Evidence  $\\prod_d L(d\\,|\\,a)$\n(retrieved docs vote)", "#3a7ebf")
    axes[1].annotate("all 3 docs\nsupport A%d" % correct,
                      (correct, 0.7), ha="center", fontsize=9,
                      color="white", fontweight="bold")

    style(axes[2], post,
          "Posterior  $p(a\\,|\\,q, \\mathrm{docs})$\n(corrected toward truth)",
          "#55a467")
    axes[2].annotate("now correct", (correct, post[correct] + 0.03),
                      ha="center", fontsize=9, color="#55a467")

    fig.suptitle(
        "RAG is Bayesian conditioning. A confidently-wrong prior (left) is "
        "corrected by retrieved evidence (middle) into a right posterior (right). "
        "Correct answer outlined in black.",
        fontsize=12, y=1.04)
    fig.tight_layout()
    out = Path("figures/04a_bayesian_update.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
