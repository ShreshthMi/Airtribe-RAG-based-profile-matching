"""Ranking metrics for retrieval quality.

Pure functions over rank-ordered identifiers, so they can be unit-tested against
hand-computed values and reused for any ranking (candidates for a JD here).

Conventions, stated explicitly because these metrics are defined inconsistently in
the wild:

* An item is **relevant** when its graded label is >= 1. Grade 2 ("would shortlist")
  and grade 1 ("plausible") are both relevant for precision, recall, MRR and MAP;
  only NDCG distinguishes them, through its gain function.
* ``precision_at_k`` divides by ``k`` itself, not by ``min(k, |relevant|)``. With
  k=10 over 7 relevant candidates the ceiling is therefore 0.7, which is the honest
  reading of "how much of what I showed the recruiter was worth reading".
* ``average_precision`` is the TREC definition: the mean of the precision at each
  relevant item's rank, over the whole ranking, divided by the number of relevant
  items.
* ``ndcg_at_k`` uses exponential gain ``2**grade - 1`` (so grade 2 is worth 3 and
  grade 1 is worth 1) with the standard ``log2(rank + 1)`` discount.

A job description with no relevant candidate is a labelling error, not a score of
zero; `harness.load_labels` rejects it, so these functions simply return 0.0.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2

Ranked = Sequence[str]
Grades = Mapping[str, int]


def _validate(ranked: Ranked, k: int | None = None) -> None:
    if k is not None and k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(set(ranked)) != len(ranked):
        duplicates = sorted({item for item in ranked if list(ranked).count(item) > 1})
        raise ValueError(f"ranked list contains duplicates: {duplicates}")


def relevant_ids(grades: Grades, min_grade: int = 1) -> set[str]:
    """Identifiers whose graded label reaches `min_grade`."""
    return {key for key, grade in grades.items() if grade >= min_grade}


def precision_at_k(ranked: Ranked, relevant: Iterable[str], k: int) -> float:
    """Share of the top k that is relevant."""
    _validate(ranked, k)
    relevant = set(relevant)
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked: Ranked, relevant: Iterable[str], k: int) -> float:
    """Share of all relevant items that appear in the top k."""
    _validate(ranked, k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant)


def hit_rate_at_k(ranked: Ranked, relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top k, else 0.0."""
    _validate(ranked, k)
    relevant = set(relevant)
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Ranked, relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item (0.0 if none is ranked)."""
    _validate(ranked)
    relevant = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked: Ranked, relevant: Iterable[str]) -> float:
    """Mean precision at the rank of each relevant item (TREC AP)."""
    _validate(ranked)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def dcg_at_k(ranked: Ranked, grades: Grades, k: int) -> float:
    """Discounted cumulative gain with 2**grade - 1 gain and log2(rank + 1) discount."""
    _validate(ranked, k)
    return sum((2 ** grades.get(item, 0) - 1) / log2(position + 1)
               for position, item in enumerate(ranked[:k], start=1))


def ndcg_at_k(ranked: Ranked, grades: Grades, k: int) -> float:
    """DCG@k divided by the DCG of the best possible ordering of the same labels."""
    _validate(ranked, k)
    ideal = sorted(grades.values(), reverse=True)[:k]
    best = sum((2 ** grade - 1) / log2(position + 1) for position, grade in enumerate(ideal, start=1))
    if best == 0:
        return 0.0
    return dcg_at_k(ranked, grades, k) / best


def evaluate_ranking(ranked: Ranked, grades: Grades, k_values: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    """Every metric above for one ranking, keyed as 'precision@3', 'ndcg@10', 'mrr', ...."""
    _validate(ranked)
    relevant = relevant_ids(grades)
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(ranked, relevant, k)
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    scores["map"] = average_precision(ranked, relevant)
    # Same again counting only grade 2, which answers "did we surface the people a
    # recruiter would actually shortlist" rather than "anyone defensible".
    strong = relevant_ids(grades, min_grade=2)
    scores["mrr_strong"] = reciprocal_rank(ranked, strong)
    for k in k_values:
        scores[f"precision_strong@{k}"] = precision_at_k(ranked, strong, k)
    return scores


def macro_average(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries (every query weighted equally)."""
    if not per_query:
        return {}
    keys = sorted({key for scores in per_query.values() for key in scores})
    return {key: sum(scores.get(key, 0.0) for scores in per_query.values()) / len(per_query)
            for key in keys}
