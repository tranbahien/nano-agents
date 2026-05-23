"""Solution 4a.2: calibrating evidence strength.

The posterior update weights each document by `evidence_strength` — how
many nats of log-likelihood one document contributes to the answer it
supports. This is a calibration parameter: it should reflect how much we
trust a single retrieved document.

  - Too low: even relevant documents barely move the prior; retrieval is
    wasted, the model stays anchored to its (possibly wrong) prior.
  - Too high: a single retrieved document (possibly a distractor) can
    override everything, including a correct prior.

We sweep evidence_strength for models of differing parametric knowledge and
find the calibrated sweet spot. This is the same over/under-confidence
tradeoff that calibration (Post 4c) addresses for the model's own outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, answer_accuracy


def main() -> None:
    env = RetrievalQA(n_queries=80, n_answers=5, n_relevant=2,
                      n_distractors=400, relevance=0.4, embed_dim=16, seed=0)
    strengths = np.linspace(0.0, 5.0, 21)
    n_trials = 25
    knowledge_levels = [(0.2, "#c44e52", "low knowledge (0.2)"),
                        (0.5, "#dd8452", "medium (0.5)"),
                        (0.9, "#3a7ebf", "high knowledge (0.9)")]

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for kn, color, label in knowledge_levels:
        acc = []
        for s in strengths:
            acc.append(answer_accuracy(env, knowledge=kn, k=8,
                                        evidence_strength=float(s),
                                        n_trials=n_trials, seed=0))
        acc = np.array(acc)
        ax.plot(strengths, acc, "-", color=color, linewidth=2, label=label)
        best = int(np.argmax(acc))
        ax.scatter([strengths[best]], [acc[best]], s=150, color=color,
                    marker="*", zorder=5, edgecolor="white", linewidth=1.2)
        print(f"  {label}: best strength = {strengths[best]:.2f}, "
              f"acc = {acc[best]:.2f}")

    ax.set_xlabel("evidence strength (nats per document)")
    ax.set_ylabel("answer accuracy")
    ax.set_title(
        "Calibrating how much to trust each document. Too low: prior wins. "
        "Too high:\na single distractor can override a correct prior. "
        "Stars mark the optimum.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    ax.annotate("prior dominates", (0.15, 0.06), fontsize=9, color="#666")
    ax.annotate("evidence dominates", (3.6, 0.06), fontsize=9, color="#666")

    fig.tight_layout()
    out = Path("figures/04a_sol2_evidence_strength.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
