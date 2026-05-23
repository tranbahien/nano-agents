"""Solution 4a.4: why retrieval embeddings are high-dimensional.

Retrieval works by cosine similarity. In low dimensions, random distractor
vectors are fairly likely to land near the query by chance — there isn't
much "room," so relevant and irrelevant docs are hard to separate. In high
dimensions, random unit vectors are nearly orthogonal (concentration of
measure): a genuinely-relevant document stands out sharply from the noise.

We sweep embedding dimension and measure both retrieval precision and
end-to-end accuracy. The payoff: retrieval quality climbs with dimension
and then saturates — which is exactly why production text embeddings live
in hundreds or thousands of dimensions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, answer_accuracy


def main() -> None:
    dims = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    relevance = 0.45  # a fixed, modest blend so dimension is the only variable
    n_trials = 15

    prec3 = []
    acc = []
    for d in dims:
        env = RetrievalQA(n_queries=80, n_answers=5, n_relevant=3,
                          n_distractors=300, relevance=relevance,
                          embed_dim=d, seed=0)
        prec3.append(np.mean([env.retrieval_precision(q, 3)
                              for q in range(env.n_queries)]))
        acc.append(answer_accuracy(env, knowledge=0.0, k=3,
                                    evidence_strength=2.0, n_trials=n_trials,
                                    seed=0))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(dims, prec3, "o-", color="#3a7ebf", linewidth=2, markersize=8,
            label="retrieval precision@3")
    ax.plot(dims, acc, "s-", color="#55a467", linewidth=2, markersize=8,
            label="end-to-end accuracy (ignorant model)")
    ax.axhline(1.0 / 5, color="#888", linestyle=":", linewidth=1.5,
                label="random-guess accuracy (1/5)")
    ax.axhline(3 / (3 + 300), color="#bbb", linestyle=":", linewidth=1.5,
                label="random-retrieval precision")

    ax.set_xscale("log"); ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("rate")
    ax.set_title(
        "Why retrieval embeddings are high-dimensional. In high dimensions "
        "random\nvectors are near-orthogonal, so relevant docs stand out. "
        "Quality saturates.",
        fontsize=11)
    ax.legend(loc="center right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)

    print("  dim:  prec@3  acc")
    for d, p, a in zip(dims, prec3, acc):
        print(f"  {d:3d}:  {p:.2f}    {a:.2f}")

    fig.tight_layout()
    out = Path("figures/04a_sol4_embedding_dim.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
