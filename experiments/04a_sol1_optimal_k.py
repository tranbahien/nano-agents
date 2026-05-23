"""Solution 4a.1: the optimal retrieval budget k vs corpus difficulty.

The dilution effect (main figure 4) showed accuracy peaks near the number
of relevant documents. But the *location* of that peak depends on how
noisy the corpus is — how many distractors there are relative to relevant
docs. We sweep both k and the distractor count and find the optimal k.

Cleaner corpora (few distractors) tolerate larger k; noisy corpora demand
tight retrieval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, answer_accuracy


def main() -> None:
    ks = [1, 2, 3, 5, 8, 12, 20, 35, 50]
    distractor_counts = [(50, "#55a467", "clean (50 distractors)"),
                          (250, "#dd8452", "moderate (250)"),
                          (1000, "#c44e52", "noisy (1000)")]
    n_relevant = 5
    n_trials = 20

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for n_dist, color, label in distractor_counts:
        env = RetrievalQA(n_queries=60, n_answers=5, n_relevant=n_relevant,
                          n_distractors=n_dist, relevance=0.7, embed_dim=16,
                          seed=0)
        acc = []
        for k in ks:
            acc.append(answer_accuracy(env, knowledge=0.0, k=k,
                                        evidence_strength=1.5,
                                        n_trials=n_trials, seed=0))
        acc = np.array(acc)
        ax.plot(ks, acc, "o-", color=color, linewidth=2, markersize=8,
                label=label)
        best = int(np.argmax(acc))
        ax.scatter([ks[best]], [acc[best]], s=180, color=color, marker="*",
                    zorder=5, edgecolor="white", linewidth=1.3)
        print(f"  {label}: best k = {ks[best]}, acc = {acc[best]:.2f}")

    ax.axvline(n_relevant, color="#888", linestyle=":", linewidth=1.5,
                label=f"# relevant docs ({n_relevant})")
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (documents retrieved)")
    ax.set_ylabel("answer accuracy (ignorant model)")
    ax.set_title(
        "Optimal k is small (2-5) for any corpus. Surprisingly, the noisier\n"
        "corpus is MORE robust to over-retrieval — random votes average out.",
        fontsize=11)
    ax.legend(loc="lower left"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = Path("figures/04a_sol1_optimal_k.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
