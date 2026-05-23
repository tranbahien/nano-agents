"""A synthetic retrieval-augmented QA environment.

The point of Post 4a is to see retrieval-augmented generation (RAG) as
Bayesian conditioning: the model's parametric memory is a *prior* over
answers, retrieved documents are *evidence*, and the answer distribution
after reading them is a *posterior*. To study that precisely we need a
setup where we know the ground truth — which documents are actually
relevant to which query, and what the correct answer is.

Design:
  - Each query has a true answer (one of `n_answers` discrete options) and
    an embedding vector.
  - The corpus has documents, each with an embedding and an "answer it
    supports". A document is *relevant* to a query if it supports the
    query's correct answer; relevant docs are embedded near the query,
    distractors are spread out.
  - A retriever returns the top-k documents by cosine similarity.

This is deliberately a toy. Real RAG embeds text with a learned encoder;
here we hand-build embeddings so retrieval quality is a knob we control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


@dataclass
class Document:
    idx: int
    embedding: np.ndarray
    supports_answer: int     # which answer this document is evidence for
    is_relevant_to: int      # query index it was generated to support (-1 = distractor)


class RetrievalQA:
    """Synthetic corpus + queries with known relevance and answers.

    Parameters
    ----------
    n_queries : number of queries.
    n_answers : size of the discrete answer set per query.
    n_relevant : how many genuinely-relevant documents exist per query.
    n_distractors : how many irrelevant documents in the corpus.
    embed_dim : embedding dimensionality.
    relevance_gap : how much closer relevant docs sit to their query than
                    distractors (higher = easier retrieval).
    """

    def __init__(self, n_queries: int = 50, n_answers: int = 5,
                 n_relevant: int = 3, n_distractors: int = 200,
                 embed_dim: int = 16, relevance: float = 0.7,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_queries = n_queries
        self.n_answers = n_answers
        self.embed_dim = embed_dim
        self.rng_seed = seed

        # Each query: a random unit embedding and a random correct answer.
        self.query_embeddings = _unit(rng.normal(0, 1, (n_queries, embed_dim)))
        self.correct_answers = rng.integers(0, n_answers, size=n_queries)

        docs: list[Document] = []
        idx = 0
        # Relevant documents: a convex blend of the query direction and a
        # random direction. `relevance` in [0, 1] is the blend weight:
        # relevance -> 1 makes relevant docs sit right on top of the query
        # (easy retrieval); relevance -> 0 makes them indistinguishable from
        # distractors (impossible retrieval).
        for q in range(n_queries):
            for _ in range(n_relevant):
                noise_dir = _unit(rng.normal(0, 1, embed_dim))
                emb = relevance * self.query_embeddings[q] + \
                    (1.0 - relevance) * noise_dir
                docs.append(Document(idx=idx, embedding=_unit(emb),
                                      supports_answer=int(self.correct_answers[q]),
                                      is_relevant_to=q))
                idx += 1
        # Distractor documents: random directions, supporting a random answer.
        for _ in range(n_distractors):
            emb = _unit(rng.normal(0, 1, embed_dim))
            docs.append(Document(idx=idx, embedding=emb,
                                  supports_answer=int(rng.integers(0, n_answers)),
                                  is_relevant_to=-1))
            idx += 1

        self.documents = docs
        self.doc_embeddings = np.stack([d.embedding for d in docs])
        self.doc_supports = np.array([d.supports_answer for d in docs])
        self.doc_relevant_to = np.array([d.is_relevant_to for d in docs])

    def retrieve(self, query_idx: int, k: int) -> list[int]:
        """Return indices of the top-k documents by cosine similarity."""
        q = self.query_embeddings[query_idx]
        sims = self.doc_embeddings @ q  # cosine sim (all unit vectors)
        topk = np.argsort(sims)[::-1][:k]
        return [int(i) for i in topk]

    def retrieval_precision(self, query_idx: int, k: int) -> float:
        """Fraction of the top-k retrieved docs that are genuinely relevant."""
        retrieved = self.retrieve(query_idx, k)
        rel = sum(1 for i in retrieved
                  if self.doc_relevant_to[i] == query_idx)
        return rel / k

    def retrieval_recall(self, query_idx: int, k: int) -> float:
        """Fraction of all relevant docs that appear in the top-k."""
        retrieved = set(self.retrieve(query_idx, k))
        total_rel = int(np.sum(self.doc_relevant_to == query_idx))
        if total_rel == 0:
            return 0.0
        found = sum(1 for i in retrieved
                    if self.doc_relevant_to[i] == query_idx)
        return found / total_rel
