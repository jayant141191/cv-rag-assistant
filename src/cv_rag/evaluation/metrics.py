"""Deterministic metrics — no API calls, fully unit-tested.

Retrieval metrics are computed on what the retriever returned *before* the
relevance grader filtered it, so they measure the vector search itself.
"""

from __future__ import annotations

from statistics import mean


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def keyword_recall(text: str, keywords: list[str]) -> float | None:
    """Share of expected keywords present in `text` (case-insensitive). None if no keywords."""
    if not keywords:
        return None
    haystack = _norm(text)
    return sum(_norm(k) in haystack for k in keywords) / len(keywords)


def candidate_recall(retrieved_candidates: list[str], expected: list[str]) -> float | None:
    """Share of expected candidates that appear at least once in the retrieved chunks."""
    if not expected:
        return None
    found = set(retrieved_candidates)
    return sum(c in found for c in expected) / len(expected)


def candidate_precision(retrieved_candidates: list[str], expected: list[str]) -> float | None:
    """Share of retrieved chunks that belong to an expected candidate."""
    if not expected or not retrieved_candidates:
        return None
    wanted = set(expected)
    return sum(c in wanted for c in retrieved_candidates) / len(retrieved_candidates)


def _is_relevant(context: str, candidate: str, expected_candidates: list[str], keywords: list[str]) -> bool:
    if candidate not in expected_candidates:
        return False
    if not keywords:
        return True
    ctx = _norm(context)
    return any(_norm(k) in ctx for k in keywords)


def reciprocal_rank(contexts: list[str], candidates: list[str],
                    expected_candidates: list[str], keywords: list[str]) -> float | None:
    """1/rank of the first chunk from an expected candidate that contains an expected keyword."""
    if not expected_candidates:
        return None
    for rank, (ctx, cand) in enumerate(zip(contexts, candidates), start=1):
        if _is_relevant(ctx, cand, expected_candidates, keywords):
            return 1.0 / rank
    return 0.0


def hit(contexts: list[str], candidates: list[str],
        expected_candidates: list[str], keywords: list[str]) -> float | None:
    rr = reciprocal_rank(contexts, candidates, expected_candidates, keywords)
    return None if rr is None else float(rr > 0)


def safe_mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None
