"""RAG as Bayesian conditioning.

The model answers a query by combining two information sources:

  1. Parametric prior  p_0(a | q): what the LM "knows" from training. For
     queries it has memorized, this is peaked on the correct answer; for
     unfamiliar queries it is near-uniform (the model is guessing).

  2. Retrieved evidence: each retrieved document d is a noisy vote for the
     answer it supports. Reading documents multiplies the prior by a
     likelihood, giving the posterior

        p(a | q, docs) ∝ p_0(a | q) · ∏_d  L(d | a),

     where L(d | a) is large when document d supports answer a. This is
     exactly Bayes' rule with a naive-Bayes (conditional-independence)
     likelihood over the retrieved set.

The model's final answer is argmax_a p(a | q, docs). We can dial the prior
quality (how much the model already knows) and the evidence strength
(how much one document shifts belief) independently — which is what makes
the "when does retrieval help?" question quantitative.
"""

from __future__ import annotations

import numpy as np

from nano_agents.retrieval.corpus import RetrievalQA


def parametric_prior(env: RetrievalQA, query_idx: int, knowledge: float,
                      rng) -> np.ndarray:
    """The LM's prior over answers from parametric memory.

    knowledge in [0, 1]: probability the model has 'memorized' this query.
      - If memorized: prior is peaked on the correct answer (strength set by
        a concentration that grows with knowledge).
      - If not: prior is near-uniform with small random tilt (a guess).
    """
    n = env.n_answers
    correct = env.correct_answers[query_idx]
    if rng.random() < knowledge:
        # Memorized: concentrate on the correct answer.
        prior = np.full(n, (1.0 - 0.85) / (n - 1))
        prior[correct] = 0.85
    else:
        # Not memorized: near-uniform guess with a little noise.
        logits = 0.3 * rng.normal(0, 1, n)
        prior = np.exp(logits - logits.max())
        prior = prior / prior.sum()
    return prior


def posterior(env: RetrievalQA, prior: np.ndarray, retrieved_docs: list,
               evidence_strength: float = 2.0) -> np.ndarray:
    """Bayesian update of the prior given retrieved documents.

    Each document contributes a likelihood factor exp(evidence_strength) to
    the answer it supports. evidence_strength controls how decisive one
    document is. With many documents this becomes a vote count in log-space.
    """
    log_post = np.log(prior + 1e-12)
    for d in retrieved_docs:
        supported = env.doc_supports[d]
        log_post[supported] += evidence_strength
    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / post.sum()


def answer_accuracy(env: RetrievalQA, knowledge: float, k: int,
                     evidence_strength: float = 2.0, n_trials: int = 1,
                     seed: int = 0, use_retrieval: bool = True) -> float:
    """Mean accuracy of the RAG model over all queries.

    use_retrieval=False evaluates the parametric prior alone (no documents),
    which is the 'closed-book' baseline.
    """
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    for _ in range(n_trials):
        for q in range(env.n_queries):
            prior = parametric_prior(env, q, knowledge, rng)
            if use_retrieval and k > 0:
                docs = env.retrieve(q, k)
                post = posterior(env, prior, docs, evidence_strength)
            else:
                post = prior
            pred = int(np.argmax(post))
            correct += int(pred == env.correct_answers[q])
            total += 1
    return correct / total
