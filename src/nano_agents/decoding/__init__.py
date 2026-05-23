"""Decoding strategies and the synthetic LM used to study them.

Companion to posts/03a-decoding-as-inference.qmd.
"""

from .strategies import all_beams, beam_search, self_consistency
from .tinylm import TinyMarkovLM, softmax

__all__ = [
    "TinyMarkovLM",
    "softmax",
    "beam_search",
    "all_beams",
    "self_consistency",
]
