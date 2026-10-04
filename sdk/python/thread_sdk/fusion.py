"""Reciprocal Rank Fusion, the same rule as the core (`thread_core::fusion::rrf`) and the TypeScript SDK:
`score(d) = sum(weight / (k + rank))` with rank from 1, accumulated list by list in float64, ranked by score
descending then id ascending, truncated to `top_n`. Lists are id sequences best first; their own scores do not
matter, only the ranks, which is what makes a cosine list and a BM25 list comparable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["DEFAULT_RRF_K", "rrf"]

DEFAULT_RRF_K = 60


def rrf(lists: Iterable[tuple[float, Sequence[int]]], k: float, top_n: int) -> list[tuple[int, float]]:
    """Fuses ranked lists (`(weight, ids)`, ids best first) into the top `top_n` `(id, score)` pairs."""
    # Insertion order keeps the float64 sums in the same order as the Rust core (list order, then rank order).
    scores: dict[int, float] = {}
    for weight, ids in lists:
        for i, id in enumerate(ids):
            scores[id] = scores.get(id, 0.0) + weight / (k + i + 1)
    ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
    return ranked[:top_n]
