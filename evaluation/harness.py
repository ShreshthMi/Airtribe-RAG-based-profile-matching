"""Drive the matcher over the labelled job descriptions and report accuracy and latency.

Two things are measured, and kept apart on purpose:

* **Retrieval quality** - how well the hybrid score *ranks* the 43 candidates for each
  JD, measured against hand-labelled relevance with `apply_filters=False` so nobody is
  removed before scoring. Mixing the must-have filter into the ranking metrics would
  score two different mechanisms as one number.
* **Must-have filtering** - evaluated separately as a classification problem: of the
  candidates the filter removed, how many were actually relevant (a false reject), and
  how many irrelevant candidates did it correctly keep out?

Latency is reported for indexing and per query, with the embedding, vector-search,
BM25 and scoring phases timed individually.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation import metrics

DEFAULT_K_VALUES = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JobLabels:
    """Hand-labelled relevance for one job description."""

    stem: str
    jd_path: Path
    grades: dict[str, int]                      # resume_path -> 2 | 1
    hard_negatives: list[dict[str, str]] = field(default_factory=list)

    @property
    def relevant(self) -> set[str]:
        return metrics.relevant_ids(self.grades)

    @property
    def strong(self) -> set[str]:
        return metrics.relevant_ids(self.grades, min_grade=2)


class LabelError(Exception):
    """The relevance label file is missing, malformed or inconsistent with the dataset."""


def load_labels(path: Path | str = "job_descriptions/relevance_labels.json",
                jd_dir: Path | str = "job_descriptions",
                resume_dir: Path | str = ".") -> list[JobLabels]:
    """Load and validate the relevance labels.

    Everything that would silently corrupt a metric is rejected here rather than
    producing a plausible-looking number: a missing JD file, a resume_path that does not
    exist, a duplicate label, a grade outside {1, 2}, or a JD with no relevant candidate
    at all (which would make recall undefined).
    """
    path, jd_dir, resume_dir = Path(path), Path(jd_dir), Path(resume_dir)
    if not path.is_file():
        raise LabelError(f"Relevance labels not found: {path}")
    try:
        payload = json.loads(path.read_text())
        raw_labels = payload["labels"]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        raise LabelError(f"{path} is unreadable or has no 'labels' key: {exc}") from exc

    hard_negatives = payload.get("hard_negatives") or {}
    out: list[JobLabels] = []
    for stem, rows in sorted(raw_labels.items()):
        jd_path = jd_dir / f"{stem}.txt"
        if not jd_path.is_file():
            raise LabelError(f"Labels name '{stem}' but {jd_path} does not exist")
        grades: dict[str, int] = {}
        for row in rows:
            resume_path = row["resume_path"]
            grade = row["grade"]
            if grade not in (1, 2):
                raise LabelError(f"{stem}: grade for {resume_path} must be 1 or 2, got {grade!r}")
            if resume_path in grades:
                raise LabelError(f"{stem}: {resume_path} is labelled twice")
            if not (resume_dir / resume_path).is_file():
                raise LabelError(f"{stem}: labelled resume {resume_path} does not exist")
            grades[resume_path] = grade
        if not grades:
            raise LabelError(f"{stem}: no relevant candidates labelled, so recall is undefined")
        out.append(JobLabels(stem=stem, jd_path=jd_path, grades=grades,
                             hard_negatives=list(hard_negatives.get(stem, []))))
    return out


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatencySummary:
    """Wall-clock timings in milliseconds over `runs` repetitions."""

    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float]:
        return {"runs": self.runs, "mean_ms": self.mean_ms, "p50_ms": self.p50_ms,
                "p95_ms": self.p95_ms, "min_ms": self.min_ms, "max_ms": self.max_ms}


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile - no interpolation, so a reported value is a real sample."""
    if not samples:
        raise ValueError("percentile of an empty sample")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(-(-len(ordered) * fraction // 1)) - 1))
    return ordered[index]


def summarise_latency(seconds: Sequence[float]) -> LatencySummary:
    """Summarise a list of durations given in seconds."""
    if not seconds:
        raise ValueError("no latency samples to summarise")
    ms = [value * 1000 for value in seconds]
    return LatencySummary(runs=len(ms), mean_ms=sum(ms) / len(ms), p50_ms=percentile(ms, 0.5),
                          p95_ms=percentile(ms, 0.95), min_ms=min(ms), max_ms=max(ms))


@contextmanager
def timed(sink: list[float]):
    """Append the duration of the block, in seconds, to `sink`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        sink.append(time.perf_counter() - start)


def time_repeatedly(call: Callable[[], Any], repeat: int = 5, warmup: int = 1) -> tuple[Any, LatencySummary]:
    """Run `call` `warmup` + `repeat` times; time only the measured runs.

    The warm-up run is discarded because the first query pays for lazy model loading and
    an empty page cache, which would otherwise dominate a small sample.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")
    for _ in range(max(0, warmup)):
        call()
    samples: list[float] = []
    result = None
    for _ in range(repeat):
        with timed(samples):
            result = call()
    return result, summarise_latency(samples)


# --------------------------------------------------------------------------- #
# Retrieval quality
# --------------------------------------------------------------------------- #
def ranked_paths(result: dict) -> list[str]:
    """resume_paths from a matcher result, in rank order."""
    return [match["resume_path"] for match in result["top_matches"]]


def rank_candidates(matcher, jd_text: str, weights=None, k: int | None = None) -> dict:
    """Full, unfiltered ranking of every candidate for `jd_text`."""
    return matcher.match(jd_text, k=k or len(matcher.candidates), weights=weights, apply_filters=False)


@dataclass(frozen=True)
class QueryEvaluation:
    stem: str
    scores: dict[str, float]
    ranking: list[str]
    result: dict
    latency: LatencySummary | None = None


def evaluate_job(matcher, labels: JobLabels, jd_text: str | None = None, weights=None,
                 k_values: Sequence[int] = DEFAULT_K_VALUES,
                 repeat: int = 0) -> QueryEvaluation:
    """Rank candidates for one JD and score the ranking against its labels."""
    text = jd_text if jd_text is not None else labels.jd_path.read_text()
    latency = None
    if repeat:
        result, latency = time_repeatedly(lambda: rank_candidates(matcher, text, weights), repeat=repeat)
    else:
        result = rank_candidates(matcher, text, weights)
    ranking = ranked_paths(result)
    return QueryEvaluation(stem=labels.stem, scores=metrics.evaluate_ranking(ranking, labels.grades, k_values),
                           ranking=ranking, result=result, latency=latency)


def evaluate_all(matcher, labelled: Sequence[JobLabels], weights=None,
                 k_values: Sequence[int] = DEFAULT_K_VALUES, repeat: int = 0,
                 jd_texts: dict[str, str] | None = None) -> tuple[list[QueryEvaluation], dict[str, float]]:
    """Evaluate every labelled JD; returns (per-JD evaluations, macro-averaged metrics)."""
    evaluations = [evaluate_job(matcher, labels, (jd_texts or {}).get(labels.stem), weights, k_values, repeat)
                   for labels in labelled]
    macro = metrics.macro_average({evaluation.stem: evaluation.scores for evaluation in evaluations})
    return evaluations, macro


# --------------------------------------------------------------------------- #
# Must-have filter, scored separately from the ranking
# --------------------------------------------------------------------------- #
def evaluate_filter(matcher, labels: JobLabels, jd_text: str | None = None, weights=None) -> dict[str, Any]:
    """Score the must-have filter as a classification of relevant vs. irrelevant.

    `false_rejects` are the expensive mistake: a candidate a human labelled relevant who
    the filter removed anyway, usually because a date-derived years estimate disagrees
    with the human's reading of the resume.
    """
    text = jd_text if jd_text is not None else labels.jd_path.read_text()
    result = matcher.match(text, k=len(matcher.candidates), weights=weights, apply_filters=True)
    kept = {match["resume_path"] for match in result["top_matches"]}
    removed = {row["resume_path"] for row in result["filtered_out"]}
    relevant = labels.relevant
    false_rejects = sorted(removed & relevant)
    reasons = {row["resume_path"]: row["failed_requirements"] for row in result["filtered_out"]}
    return {
        "stem": labels.stem,
        "requirements": result["must_have_requirements"],
        "kept": len(kept),
        "removed": len(removed),
        "false_rejects": false_rejects,
        "false_reject_reasons": {path: reasons[path] for path in false_rejects},
        "true_rejects": len(removed - relevant),
        "relevant_kept": len(kept & relevant),
        "relevant_total": len(relevant),
        # Of everyone the filter let through, how many are relevant?
        "precision_after_filter": (len(kept & relevant) / len(kept)) if kept else 0.0,
        "recall_after_filter": (len(kept & relevant) / len(relevant)) if relevant else 0.0,
    }
