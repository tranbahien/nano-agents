"""Retrieval-augmented generation as Bayesian conditioning.

Companion to posts/04a-rag-as-bayesian-conditioning.qmd.
"""

from .bayesian import answer_accuracy, parametric_prior, posterior
from .corpus import Document, RetrievalQA

__all__ = [
    "RetrievalQA",
    "Document",
    "parametric_prior",
    "posterior",
    "answer_accuracy",
]
