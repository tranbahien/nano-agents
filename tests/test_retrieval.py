"""Tests for the retrieval subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.retrieval import (
    RetrievalQA,
    answer_accuracy,
    parametric_prior,
    posterior,
)


def test_corpus_shapes():
    env = RetrievalQA(n_queries=10, n_answers=4, n_relevant=3,
                      n_distractors=50, embed_dim=8, seed=0)
    assert env.query_embeddings.shape == (10, 8)
    assert len(env.documents) == 10 * 3 + 50
    assert env.doc_embeddings.shape == (len(env.documents), 8)


def test_embeddings_are_unit_norm():
    env = RetrievalQA(n_queries=5, embed_dim=8, seed=0)
    norms = np.linalg.norm(env.doc_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_retrieval_finds_relevant_docs():
    """With a clear relevance gap, top-k should over-represent relevant docs."""
    env = RetrievalQA(n_queries=20, n_relevant=3, n_distractors=200,
                      relevance=0.8, embed_dim=16, seed=0)
    # Average precision at k=3 should be well above the random baseline.
    precisions = [env.retrieval_precision(q, k=3) for q in range(env.n_queries)]
    random_baseline = 3 / (3 + 200)  # if retrieval were random
    assert np.mean(precisions) > 5 * random_baseline


def test_retrieval_precision_recall_in_range():
    env = RetrievalQA(n_queries=10, n_relevant=3, seed=1)
    for q in range(env.n_queries):
        p = env.retrieval_precision(q, k=5)
        r = env.retrieval_recall(q, k=5)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0


def test_prior_peaks_on_correct_when_known():
    env = RetrievalQA(n_queries=5, n_answers=5, seed=0)
    # knowledge=1.0 -> always memorized -> prior argmax == correct.
    rng = np.random.default_rng(0)
    for q in range(env.n_queries):
        prior = parametric_prior(env, q, knowledge=1.0, rng=rng)
        assert int(np.argmax(prior)) == env.correct_answers[q]


def test_prior_normalizes():
    env = RetrievalQA(n_queries=5, n_answers=5, seed=0)
    rng = np.random.default_rng(0)
    for q in range(env.n_queries):
        for kn in (0.0, 1.0):
            prior = parametric_prior(env, q, knowledge=kn, rng=rng)
            assert abs(prior.sum() - 1.0) < 1e-9


def test_posterior_shifts_toward_evidence():
    """Retrieving documents that support the correct answer should raise its
    posterior probability above the prior."""
    env = RetrievalQA(n_queries=20, n_relevant=3, n_distractors=100,
                      relevance=0.8, embed_dim=16, seed=0)
    rng = np.random.default_rng(0)
    improved = 0
    for q in range(env.n_queries):
        correct = env.correct_answers[q]
        prior = parametric_prior(env, q, knowledge=0.0, rng=rng)  # ignorant prior
        docs = env.retrieve(q, k=3)
        post = posterior(env, prior, docs, evidence_strength=2.0)
        if post[correct] >= prior[correct]:
            improved += 1
    # Retrieval should help on the large majority of queries.
    assert improved > 0.7 * env.n_queries


def test_retrieval_beats_closed_book_for_ignorant_model():
    """A model with no parametric knowledge should do much better with
    retrieval than without."""
    env = RetrievalQA(n_queries=50, n_answers=5, n_relevant=3,
                      n_distractors=200, relevance=0.8, embed_dim=16,
                      seed=0)
    closed = answer_accuracy(env, knowledge=0.0, k=0, n_trials=3,
                              use_retrieval=False, seed=0)
    openbook = answer_accuracy(env, knowledge=0.0, k=3, n_trials=3,
                                use_retrieval=True, seed=0)
    assert openbook > closed + 0.2


def test_perfect_knowledge_high_without_retrieval():
    env = RetrievalQA(n_queries=50, n_answers=5, seed=0)
    closed = answer_accuracy(env, knowledge=1.0, k=0, n_trials=3,
                              use_retrieval=False, seed=0)
    assert closed > 0.9
