"""Figure: the retrieval budget k and the dilution effect.

Conditioning on more documents isn't always better. Too few (k below the
number of relevant docs) and you miss evidence; too many and you pull in
distractors whose votes dilute the signal. We sweep k for a few retriever
qualities, with an ignorant model (knowledge=0) so accuracy reflects
retrieval alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, answer_accuracy


def main() -> None:
    ks = [1, 2, 3, 5, 8, 12, 20, 35, 50]
    relevances = [(0.55, "#c44e52"), (0.7, "#dd8452"), (0.85, "#55a467")]
    n_trials = 25

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for rel, color in relevances:
        env = RetrievalQA(n_queries=80, n_answers=5, n_relevant=3,
                          n_distractors=250, relevance=rel, embed_dim=16,
                          seed=0)
        acc = []
        for k in ks:
            acc.append(answer_accuracy(env, knowledge=0.0, k=k,
                                        evidence_strength=1.5,
                                        n_trials=n_trials, seed=0))
        ax.plot(ks, acc, "o-", color=color, linewidth=2, markersize=8,
                label=f"retriever quality = {rel}")
        # Mark the peak.
        best = int(np.argmax(acc))
        ax.scatter([ks[best]], [acc[best]], s=160, color=color, marker="*",
                    zorder=5, edgecolor="white", linewidth=1.2)

    ax.axvline(3, color="#888", linestyle=":", linewidth=1.5,
                label="# relevant docs (3)")
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (documents retrieved and conditioned on)")
    ax.set_ylabel("answer accuracy (ignorant model)")
    ax.set_title(
        "More retrieval isn't always better. Accuracy peaks near the number "
        "of\nrelevant docs, then declines as distractors dilute the evidence.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = Path("figures/04a_k_tradeoff.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
