"""Solution 4a.3: confidence-gated retrieval.

Should the agent always retrieve? No. With an imperfect retriever, pulling
documents for a query the model already answers confidently risks
corrupting a correct prior (main figure 5, solution 2). The fix: gate
retrieval on the model's own confidence — retrieve only when the prior is
uncertain.

This is the RAG analogue of the tool-use threshold from Post 3b, and it
depends on the model being *calibrated*: it must know when it doesn't know.
A miscalibrated model gates at the wrong times. This directly motivates
Post 4c (calibration).

We compare three policies under a mediocre retriever:
  - never retrieve (closed-book)
  - always retrieve
  - confidence-gated (retrieve iff prior max-prob < threshold)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.retrieval import RetrievalQA, parametric_prior, posterior


def evaluate_gated(env, knowledge, k, evidence_strength, conf_threshold,
                    n_trials, seed):
    """Confidence-gated policy: retrieve only when prior max-prob < threshold."""
    rng = np.random.default_rng(seed)
    correct = total = 0
    retrieved_count = 0
    for _ in range(n_trials):
        for q in range(env.n_queries):
            prior = parametric_prior(env, q, knowledge, rng)
            if prior.max() < conf_threshold:
                docs = env.retrieve(q, k)
                post = posterior(env, prior, docs, evidence_strength)
                retrieved_count += 1
            else:
                post = prior
            correct += int(np.argmax(post) == env.correct_answers[q])
            total += 1
    return correct / total, retrieved_count / total


def evaluate_fixed(env, knowledge, k, evidence_strength, use_retrieval,
                    n_trials, seed):
    rng = np.random.default_rng(seed)
    correct = total = 0
    for _ in range(n_trials):
        for q in range(env.n_queries):
            prior = parametric_prior(env, q, knowledge, rng)
            if use_retrieval:
                docs = env.retrieve(q, k)
                post = posterior(env, prior, docs, evidence_strength)
            else:
                post = prior
            correct += int(np.argmax(post) == env.correct_answers[q])
            total += 1
    return correct / total


def main() -> None:
    # Mediocre retriever, so that indiscriminate retrieval can hurt.
    env = RetrievalQA(n_queries=80, n_answers=5, n_relevant=2,
                      n_distractors=400, relevance=0.45, embed_dim=16, seed=0)
    knowledge = 0.6  # the model knows a fair amount, so has priors worth keeping
    k = 6
    evidence_strength = 2.0
    n_trials = 30

    never = evaluate_fixed(env, knowledge, k, evidence_strength,
                            use_retrieval=False, n_trials=n_trials, seed=0)
    always = evaluate_fixed(env, knowledge, k, evidence_strength,
                             use_retrieval=True, n_trials=n_trials, seed=0)

    thresholds = np.linspace(0.2, 1.0, 17)
    gated_acc = []
    gated_rate = []
    for thr in thresholds:
        a, r = evaluate_gated(env, knowledge, k, evidence_strength,
                               float(thr), n_trials, seed=0)
        gated_acc.append(a)
        gated_rate.append(r)
    gated_acc = np.array(gated_acc)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(thresholds, gated_acc, "o-", color="#55a467", linewidth=2,
            markersize=7, label="confidence-gated retrieval")
    ax.axhline(never, color="#dd8452", linestyle="--", linewidth=2,
                label=f"never retrieve (closed-book): {never:.2f}")
    ax.axhline(always, color="#c44e52", linestyle="--", linewidth=2,
                label=f"always retrieve: {always:.2f}")
    best = int(np.argmax(gated_acc))
    ax.scatter([thresholds[best]], [gated_acc[best]], s=180, color="#55a467",
                marker="*", zorder=5, edgecolor="white", linewidth=1.3)
    ax.annotate(f"best: {gated_acc[best]:.2f}\n(retrieves {gated_rate[best]*100:.0f}% of queries)",
                 (thresholds[best], gated_acc[best] + 0.02), ha="center",
                 fontsize=9, color="#2e7d4f")

    ax.set_xlabel("confidence threshold (retrieve iff prior max-prob < threshold)")
    ax.set_ylabel("answer accuracy")
    ax.set_title(
        "Confidence-gated retrieval beats both always- and never-retrieve.\n"
        "Retrieve only when the prior is uncertain — if the model is calibrated.",
        fontsize=11)
    ax.legend(loc="lower center"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    print(f"  never={never:.3f}, always={always:.3f}, "
          f"best gated={gated_acc[best]:.3f} at thr={thresholds[best]:.2f}")

    fig.tight_layout()
    out = Path("figures/04a_sol3_confidence_gated.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
