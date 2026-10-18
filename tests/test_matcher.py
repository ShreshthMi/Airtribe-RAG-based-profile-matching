"""End-to-end matching over a tiny index, plus the output contract and failure modes."""
from __future__ import annotations

import json

import pytest

import job_matcher
from job_matcher import JobMatcher, MatcherError, ScoreWeights
from tests.conftest import TINY_ML_JD, FakeEmbedder

# The assignment's required output fields.
REQUIRED_MATCH_FIELDS = {"candidate_name", "resume_path", "match_score", "matched_skills",
                         "relevant_excerpts", "reasoning"}


def test_the_index_contains_every_resume(tiny_index):
    _, report = tiny_index
    assert len(report.records) == 4
    assert report.chunk_count >= 4
    assert report.skipped == []


def test_the_build_reports_phase_timings(tiny_index):
    _, report = tiny_index
    assert {"load", "parse", "embed", "store", "total"} <= set(report.timings)
    assert report.timings["total"] > 0


def test_the_result_carries_every_field_the_assignment_asks_for(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=3)
    assert "job_description" in result and "top_matches" in result
    for match in result["top_matches"]:
        assert REQUIRED_MATCH_FIELDS <= set(match)
        assert isinstance(match["match_score"], int)
        assert 0 <= match["match_score"] <= 100
        assert isinstance(match["relevant_excerpts"], list)
        assert match["reasoning"]


def test_the_right_candidate_is_ranked_first(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=4)
    assert result["top_matches"][0]["resume_path"].endswith("ml_engineer.md")


def test_a_frontend_job_ranks_the_frontend_candidate_first(tiny_matcher):
    jd = "Frontend Engineer\n\nRequirements\n- React and TypeScript\n- 3+ years experience\n"
    result = tiny_matcher.match(jd, k=4)
    assert result["top_matches"][0]["resume_path"].endswith("frontend.txt")


def test_excerpts_are_tagged_with_their_section(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=1)
    assert all(excerpt.startswith("[") for excerpt in result["top_matches"][0]["relevant_excerpts"])


def test_a_must_have_moves_failing_candidates_out_with_a_reason(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=10, must_haves=["5+ years Python"])
    kept = {match["resume_path"] for match in result["top_matches"]}
    assert any(path.endswith("ml_engineer.md") for path in kept)
    assert all(not path.endswith("frontend.txt") for path in kept)
    rejected = {row["resume_path"]: row["failed_requirements"] for row in result["filtered_out"]}
    assert any("Python" in " ".join(reasons) for reasons in rejected.values())


def test_an_education_must_have_filters_on_degree_level(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=10, must_haves=["Master's degree"],
                                auto_requirements=False)
    kept = {match["resume_path"] for match in result["top_matches"]}
    assert any(path.endswith("ml_engineer.md") for path in kept)      # M.Tech
    assert all(not path.endswith("java_backend.txt") for path in kept)  # B.Eng


def test_apply_filters_false_keeps_everyone_and_annotates_them(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=10, must_haves=["5+ years Python"], apply_filters=False)
    assert len(result["top_matches"]) == 4
    assert result["filtered_out"] == []
    annotated = [match for match in result["top_matches"] if match["failed_requirements"]]
    assert annotated, "candidates failing a must-have should still say so"


def test_a_must_have_that_cannot_be_parsed_is_reported_not_dropped(tiny_matcher, capsys):
    # auto_requirements off, so the only requirement in play is the unparseable one
    result = tiny_matcher.match(TINY_ML_JD, k=4, must_haves=["fluent in Mandarin"],
                                auto_requirements=False)
    assert result["unparsed_must_haves"] == ["fluent in Mandarin"]
    assert result["must_have_requirements"] == []
    assert "could not interpret must-have" in capsys.readouterr().err
    assert len(result["top_matches"]) == 4   # nobody was filtered on it


def test_the_weights_used_are_recorded_in_the_result(tiny_matcher):
    weights = ScoreWeights(1.0, 0.0, 0.0, 0.0)
    result = tiny_matcher.match(TINY_ML_JD, k=1, weights=weights)
    assert result["score_weights"]["semantic"] == 1.0
    assert result["top_matches"][0]["score_breakdown"]["semantic"] == result["top_matches"][0]["match_score"]


def test_score_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ScoreWeights(0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="must not be negative"):
        ScoreWeights(1.5, -0.5, 0.0, 0.0)


def test_asking_for_more_matches_than_there_are_candidates_is_fine(tiny_matcher):
    assert len(tiny_matcher.match(TINY_ML_JD, k=99)["top_matches"]) <= 4


def test_min_score_drops_weak_matches(tiny_matcher):
    everyone = tiny_matcher.match(TINY_ML_JD, k=10, auto_requirements=False)
    cutoff = everyone["top_matches"][0]["match_score"]
    filtered = tiny_matcher.match(TINY_ML_JD, k=10, auto_requirements=False, min_score=cutoff)
    assert all(match["match_score"] >= cutoff for match in filtered["top_matches"])
    assert len(filtered["top_matches"]) < len(everyone["top_matches"])


def test_equal_scores_rank_in_a_stable_order(tiny_matcher):
    # a weight vector that gives everyone the same score still has to be deterministic
    weights = ScoreWeights(0.0, 0.0, 0.0, 1.0)
    first = tiny_matcher.match(TINY_ML_JD, k=4, weights=weights, auto_requirements=False)
    second = tiny_matcher.match(TINY_ML_JD, k=4, weights=weights, auto_requirements=False)
    paths = [match["resume_path"] for match in first["top_matches"]]
    assert paths == [match["resume_path"] for match in second["top_matches"]]
    assert paths == sorted(paths)


def test_an_empty_job_description_is_rejected(tiny_matcher):
    with pytest.raises(MatcherError, match="empty"):
        tiny_matcher.match("   \n ")


def test_a_missing_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(MatcherError, match="resume_rag.py"):
        JobMatcher(tmp_path)


def test_a_malformed_registry_is_reported(tmp_path):
    (tmp_path / "candidates.json").write_text("{not json")
    with pytest.raises(MatcherError, match="unreadable or malformed"):
        JobMatcher(tmp_path)


def test_an_empty_registry_is_reported(tmp_path):
    (tmp_path / "candidates.json").write_text(json.dumps({"candidates": []}))
    with pytest.raises(MatcherError, match="no candidates"):
        JobMatcher(tmp_path)


def test_querying_with_a_different_model_than_the_index_warns(tiny_index, monkeypatch, capsys):
    db_path, _ = tiny_index
    monkeypatch.setattr(job_matcher, "Embedder", FakeEmbedder)
    JobMatcher(db_path, model_name="some-other-model")
    assert "not comparable across models" in capsys.readouterr().err


def test_reasoning_names_the_matched_and_missing_critical_skills(tiny_matcher):
    result = tiny_matcher.match(TINY_ML_JD, k=4, auto_requirements=False)
    backend = next(match for match in result["top_matches"]
                   if match["resume_path"].endswith("java_backend.txt"))
    assert "missing" in backend["reasoning"]
    assert "critical skills" in backend["reasoning"]


def test_a_long_job_description_is_split_into_passages_rather_than_truncated(tiny_matcher):
    passages = tiny_matcher._jd_passages(TINY_ML_JD * 20)
    assert len(passages) > 1
    assert all(len(passage) <= 700 for passage in passages)
