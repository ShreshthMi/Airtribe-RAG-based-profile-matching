"""The evaluation harness: label validation, latency helpers and filter scoring."""
from __future__ import annotations

import json

import pytest

from evaluation import harness
from tests.conftest import TINY_ML_JD


# --- labels ---------------------------------------------------------------- #
def write_labels(tmp_path, labels, hard_negatives=None):
    """Create a label file plus the JD and resume files it refers to."""
    jd_dir = tmp_path / "jds"
    jd_dir.mkdir(exist_ok=True)
    resume_dir = tmp_path / "resumes"
    resume_dir.mkdir(exist_ok=True)
    for stem, rows in labels.items():
        (jd_dir / f"{stem}.txt").write_text("A job description")
        for row in rows:
            target = tmp_path / row["resume_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("A resume")
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"rubric": "2 = shortlist, 1 = plausible",
                                "labels": labels, "hard_negatives": hard_negatives or {}}))
    return path, jd_dir


def test_valid_labels_load(tmp_path):
    path, jd_dir = write_labels(tmp_path, {"role": [{"resume_path": "resumes/a.txt", "grade": 2},
                                                    {"resume_path": "resumes/b.txt", "grade": 1}]},
                                hard_negatives={"role": [{"resume_path": "resumes/b.txt", "why": "junior"}]})
    labelled = harness.load_labels(path, jd_dir, tmp_path)
    assert len(labelled) == 1
    job = labelled[0]
    assert job.stem == "role"
    assert job.relevant == {"resumes/a.txt", "resumes/b.txt"}
    assert job.strong == {"resumes/a.txt"}
    assert job.hard_negatives[0]["why"] == "junior"


def test_a_missing_label_file_is_reported(tmp_path):
    with pytest.raises(harness.LabelError, match="not found"):
        harness.load_labels(tmp_path / "absent.json")


def test_a_label_file_without_labels_is_reported(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"rubric": "x"}))
    with pytest.raises(harness.LabelError, match="no 'labels' key"):
        harness.load_labels(path, tmp_path, tmp_path)


def test_a_label_naming_a_missing_job_description_is_rejected(tmp_path):
    path, jd_dir = write_labels(tmp_path, {"role": [{"resume_path": "resumes/a.txt", "grade": 2}]})
    (jd_dir / "role.txt").unlink()
    with pytest.raises(harness.LabelError, match="does not exist"):
        harness.load_labels(path, jd_dir, tmp_path)


def test_a_label_naming_a_missing_resume_is_rejected(tmp_path):
    path, jd_dir = write_labels(tmp_path, {"role": [{"resume_path": "resumes/a.txt", "grade": 2}]})
    (tmp_path / "resumes" / "a.txt").unlink()
    with pytest.raises(harness.LabelError, match="does not exist"):
        harness.load_labels(path, jd_dir, tmp_path)


def test_an_out_of_range_grade_is_rejected(tmp_path):
    path, jd_dir = write_labels(tmp_path, {"role": [{"resume_path": "resumes/a.txt", "grade": 3}]})
    with pytest.raises(harness.LabelError, match="must be 1 or 2"):
        harness.load_labels(path, jd_dir, tmp_path)


def test_a_duplicated_label_is_rejected(tmp_path):
    path, jd_dir = write_labels(tmp_path, {"role": [{"resume_path": "resumes/a.txt", "grade": 2},
                                                    {"resume_path": "resumes/a.txt", "grade": 1}]})
    with pytest.raises(harness.LabelError, match="labelled twice"):
        harness.load_labels(path, jd_dir, tmp_path)


def test_a_job_description_with_no_relevant_candidate_is_rejected(tmp_path):
    path, jd_dir = write_labels(tmp_path, {"role": []})
    with pytest.raises(harness.LabelError, match="recall is undefined"):
        harness.load_labels(path, jd_dir, tmp_path)


def test_the_projects_own_labels_are_valid(project_root):
    labelled = harness.load_labels(project_root / "job_descriptions" / "relevance_labels.json",
                                   project_root / "job_descriptions", project_root)
    assert len(labelled) >= 5, "the assignment asks for at least 5 job descriptions"
    assert all(len(job.relevant) >= 4 for job in labelled)


# --- latency --------------------------------------------------------------- #
def test_percentile_is_nearest_rank_so_it_returns_a_real_sample():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert harness.percentile(samples, 0.5) == 3.0
    assert harness.percentile(samples, 0.95) == 5.0
    assert harness.percentile(samples, 1.0) == 5.0
    assert harness.percentile([7.0], 0.5) == 7.0


def test_percentile_validates_its_inputs():
    with pytest.raises(ValueError, match="empty sample"):
        harness.percentile([], 0.5)
    with pytest.raises(ValueError, match="fraction must be"):
        harness.percentile([1.0], 0.0)


def test_summarise_latency_converts_to_milliseconds():
    summary = harness.summarise_latency([0.010, 0.020, 0.030])
    assert summary.runs == 3
    assert summary.mean_ms == pytest.approx(20.0)
    assert summary.min_ms == pytest.approx(10.0)
    assert summary.max_ms == pytest.approx(30.0)
    assert set(summary.as_dict()) == {"runs", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"}


def test_summarise_latency_needs_at_least_one_sample():
    with pytest.raises(ValueError, match="no latency samples"):
        harness.summarise_latency([])


def test_time_repeatedly_discards_the_warmup_run():
    calls = []
    result, summary = harness.time_repeatedly(lambda: calls.append(1) or "done", repeat=3, warmup=2)
    assert result == "done"
    assert len(calls) == 5        # 2 warm-up + 3 measured
    assert summary.runs == 3      # only the measured runs are summarised


def test_time_repeatedly_needs_at_least_one_measured_run():
    with pytest.raises(ValueError, match="repeat must be at least 1"):
        harness.time_repeatedly(lambda: None, repeat=0)


def test_timed_records_the_block_duration():
    samples = []
    with harness.timed(samples):
        sum(range(1000))
    assert len(samples) == 1 and samples[0] >= 0


# --- evaluation over a real (tiny) index ----------------------------------- #
def test_evaluating_a_job_scores_the_full_unfiltered_ranking(tiny_matcher, tiny_corpus, tmp_path):
    # labels use the same sandbox-relative paths the loader reports, e.g. "resumes0/x.md"
    labels = {"ml": [{"resume_path": f"{tiny_corpus.name}/ml_engineer.md", "grade": 2}]}
    path = tmp_path / "labels.json"
    jd_dir = tmp_path / "jds"
    jd_dir.mkdir()
    (jd_dir / "ml.txt").write_text(TINY_ML_JD)
    path.write_text(json.dumps({"labels": labels}))
    job = harness.load_labels(path, jd_dir, tiny_corpus.parent)[0]

    evaluation = harness.evaluate_job(tiny_matcher, job)
    # every candidate is ranked, including those that fail the JD's 5+ years Python rule
    assert len(evaluation.ranking) == 4
    assert evaluation.scores["precision@1"] == 1.0
    assert evaluation.scores["mrr"] == 1.0


def test_evaluate_all_macro_averages_across_jobs(tiny_matcher, tiny_corpus, tmp_path):
    jd_dir = tmp_path / "jds"
    jd_dir.mkdir()
    (jd_dir / "ml.txt").write_text(TINY_ML_JD)
    (jd_dir / "fe.txt").write_text("Frontend Engineer\n\nRequirements\n- React and TypeScript\n")
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": {
        "ml": [{"resume_path": f"{tiny_corpus.name}/ml_engineer.md", "grade": 2}],
        "fe": [{"resume_path": f"{tiny_corpus.name}/frontend.txt", "grade": 2}],
    }}))
    labelled = harness.load_labels(path, jd_dir, tiny_corpus.parent)
    evaluations, macro = harness.evaluate_all(tiny_matcher, labelled)
    assert len(evaluations) == 2
    assert macro["precision@1"] == pytest.approx(1.0)


def test_the_filter_is_scored_separately_from_the_ranking(tiny_matcher, tiny_corpus, tmp_path):
    jd_dir = tmp_path / "jds"
    jd_dir.mkdir()
    (jd_dir / "ml.txt").write_text(TINY_ML_JD)
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": {
        "ml": [{"resume_path": f"{tiny_corpus.name}/ml_engineer.md", "grade": 2}]}}))
    job = harness.load_labels(path, jd_dir, tiny_corpus.parent)[0]

    report = harness.evaluate_filter(tiny_matcher, job)
    assert report["requirements"]                      # the JD's 5+ years Python
    assert report["false_rejects"] == []               # the relevant candidate survives
    assert report["removed"] >= 1                      # the frontend candidate does not
    assert report["recall_after_filter"] == pytest.approx(1.0)
