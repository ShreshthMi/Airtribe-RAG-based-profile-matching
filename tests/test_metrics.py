"""Ranking metrics, checked against values computed by hand."""
from __future__ import annotations

from math import log2

import pytest

from evaluation import metrics

# ranking: a, b, c, d - relevant = {a, c} with a graded 2 and c graded 1
RANKED = ["a", "b", "c", "d"]
GRADES = {"a": 2, "c": 1}
RELEVANT = {"a", "c"}


def test_precision_at_k_divides_by_k_not_by_the_number_relevant():
    # 2 relevant items in the top 4 is 0.5, not 1.0
    assert metrics.precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)
    assert metrics.precision_at_k(RANKED, RELEVANT, 1) == pytest.approx(1.0)
    assert metrics.precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)


def test_recall_at_k_counts_relevant_items_found():
    assert metrics.recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.5)
    assert metrics.recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(1.0)


def test_recall_is_zero_when_nothing_is_relevant():
    assert metrics.recall_at_k(RANKED, set(), 3) == 0.0


def test_hit_rate_is_binary():
    assert metrics.hit_rate_at_k(RANKED, RELEVANT, 1) == 1.0
    assert metrics.hit_rate_at_k(RANKED, {"d"}, 2) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert metrics.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1.0)
    assert metrics.reciprocal_rank(RANKED, {"b", "d"}) == pytest.approx(0.5)
    assert metrics.reciprocal_rank(RANKED, {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_no_relevant_item_is_ranked():
    assert metrics.reciprocal_rank(RANKED, {"zz"}) == 0.0


def test_average_precision_averages_precision_at_each_hit():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert metrics.average_precision(RANKED, RELEVANT) == pytest.approx((1 + 2 / 3) / 2)


def test_ndcg_uses_exponential_gain_and_log_discount():
    expected_dcg = (2 ** 2 - 1) / log2(2) + (2 ** 1 - 1) / log2(4)
    expected_ideal = (2 ** 2 - 1) / log2(2) + (2 ** 1 - 1) / log2(3)
    assert metrics.dcg_at_k(RANKED, GRADES, 4) == pytest.approx(expected_dcg)
    assert metrics.ndcg_at_k(RANKED, GRADES, 4) == pytest.approx(expected_dcg / expected_ideal)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert metrics.ndcg_at_k(["a", "c", "b"], GRADES, 3) == pytest.approx(1.0)


def test_ndcg_is_zero_when_every_grade_is_zero():
    assert metrics.ndcg_at_k(RANKED, {}, 4) == 0.0


def test_relevant_ids_respects_the_minimum_grade():
    assert metrics.relevant_ids(GRADES) == {"a", "c"}
    assert metrics.relevant_ids(GRADES, min_grade=2) == {"a"}


def test_k_below_one_is_rejected():
    with pytest.raises(ValueError, match="k must be at least 1"):
        metrics.precision_at_k(RANKED, RELEVANT, 0)


def test_a_duplicated_candidate_in_the_ranking_is_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        metrics.precision_at_k(["a", "a", "b"], RELEVANT, 2)


def test_evaluate_ranking_reports_every_metric():
    scores = metrics.evaluate_ranking(RANKED, GRADES, k_values=(1, 3))
    assert set(scores) == {
        "precision@1", "recall@1", "ndcg@1", "hit_rate@1",
        "precision@3", "recall@3", "ndcg@3", "hit_rate@3",
        "mrr", "map", "mrr_strong", "precision_strong@1", "precision_strong@3",
    }
    assert scores["precision_strong@1"] == pytest.approx(1.0)  # 'a' is the grade-2 item


def test_macro_average_weights_queries_equally():
    averaged = metrics.macro_average({"x": {"mrr": 1.0}, "y": {"mrr": 0.0}})
    assert averaged == {"mrr": pytest.approx(0.5)}
    assert metrics.macro_average({}) == {}
