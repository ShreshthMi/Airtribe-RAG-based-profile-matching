"""
Part B: Job matching engine.

Given a job description (JD):
  1. Embed the JD (split into passages so long JDs aren't truncated by the model).
  2. Semantic search: retrieve similar resume chunks from ChromaDB and aggregate per candidate.
  3. Keyword search: BM25 over full resume text + exact matching of critical skills.
  4. Hybrid score (0-100) = semantic + critical-skill coverage + BM25 + experience fit.
  5. Filter by must-have requirements (e.g. "5+ years Python", "Kubernetes", "Master's degree").
  6. Return the top-K (default 10) with matched skills, excerpts and reasoning.

Usage
  python job_matcher.py --jd-file jds/senior_ml_engineer.txt
  python job_matcher.py --jd "Senior Python developer ..." --must-have "5+ years Python" --k 10 --output out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from resume_loader import LoaderError, load_text_file
from resume_rag import (
    COLLECTION_NAME, DEFAULT_MODEL, EDUCATION_LEVELS, REGISTRY_FILE, Embedder,
    extract_skills, split_sections,
)


class MatcherError(Exception):
    """The index is missing or unusable, or the job description is empty."""


@dataclass(frozen=True)
class ScoreWeights:
    """Relative weight of each scoring component; must sum to 1.

    Injectable so the notebook can run semantic-only / BM25-only / skills-only
    ablations against the same index instead of hard-coding one blend.
    """

    semantic: float = 0.50
    critical_skills: float = 0.30
    keyword_bm25: float = 0.10
    experience_fit: float = 0.10

    def __post_init__(self) -> None:
        total = self.semantic + self.critical_skills + self.keyword_bm25 + self.experience_fit
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"score weights must sum to 1.0, got {total:g}")
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"score weight {name} must not be negative, got {value:g}")


DEFAULT_WEIGHTS = ScoreWeights()
# MiniLM cosine similarities for related resume/JD text typically fall in this band
SIM_FLOOR, SIM_CEIL = 0.15, 0.65

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or our the to we will with you your
this that these those who what which their they them us can should must able strong experience years year plus
work working role team including such etc using use""".split())


def tokenize(text: str) -> list[str]:
    return [t.strip(".") for t in _TOKEN_RE.findall(text.lower()) if t.strip(".") and t.strip(".") not in _STOPWORDS]


# --------------------------------------------------------------------------- #
# Must-have requirements
# --------------------------------------------------------------------------- #
@dataclass
class Requirement:
    raw: str
    skill: str | None = None
    min_years: float | None = None
    education_rank: int | None = None

    def check(self, cand: dict) -> tuple[bool, str]:
        """Return (passes, explanation) for one candidate."""
        if self.education_rank is not None:
            ok = cand["education_rank"] >= self.education_rank
            return ok, f"education {cand['education_level']}"
        if self.skill and self.min_years is not None:
            yrs = cand["skill_years"].get(self.skill, 0.0)
            return yrs >= self.min_years, f"{yrs:g} yrs {self.skill}"
        if self.skill:
            return self.skill in cand["skills"], f"{self.skill} {'present' if self.skill in cand['skills'] else 'missing'}"
        if self.min_years is not None:
            return cand["experience_years"] >= self.min_years, f"{cand['experience_years']:g} yrs total"
        return True, ""


# "5-7 years" states a minimum of five, not seven. Collapse a range to its lower bound
# before any years pattern sees it.
_YEAR_RANGE_RE = re.compile(r"(\d{1,2})\s*(?:-|\u2013|\u2014|to)\s*\d{1,2}(\s*\+?\s*(?:years?|yrs?))", re.IGNORECASE)


def lower_bound_years(text: str) -> str:
    """Rewrite '5-7 years' as '5 years' so a range is read as the minimum it states."""
    return _YEAR_RANGE_RE.sub(r"\1\2", text)


def parse_requirement(text: str) -> Requirement | None:
    """Parse strings like '5+ years Python', 'Python 5 years', '3 years experience', 'Kubernetes', 'Master's degree'."""
    raw = text.strip()
    normalised = lower_bound_years(raw)
    # A degree requirement naming several levels states the LOWEST as its bar:
    # "Bachelor's or Master's degree" requires a Bachelor's, not a Master's.
    ranks = [rank for _, rank, pattern in EDUCATION_LEVELS
             if re.search(pattern, normalised, re.IGNORECASE)]
    if ranks and re.search(r"degree|level|education|ph\.?d|master|bachelor|associate", raw, re.I):
        return Requirement(raw=raw, education_rank=min(ranks))
    years = None
    m = re.search(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)", normalised, re.IGNORECASE)
    if m:
        years = float(m.group(1))
    skills = extract_skills(re.sub(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)", " ", normalised, flags=re.I))
    if skills:
        return Requirement(raw=raw, skill=skills[0], min_years=years)
    if years is not None:
        return Requirement(raw=raw, min_years=years)
    return None


def extract_requirements_from_jd(jd: str) -> list[Requirement]:
    """Pull 'N+ years <skill>' / 'N+ years of experience' statements out of the JD text.

    Every gap in the pattern is horizontal whitespace only. With a plain `\\s` the match ran
    past the end of the line, so a bullet reading "5+ years of experience" followed by a
    separate "Python and SQL" bullet silently became a hard "5+ years Python" filter - a
    requirement the job description never stated, applied without anyone asking for it.
    """
    reqs: list[Requirement] = []
    seen = set()
    gap = r"[^\S\n]"
    pattern = re.compile(
        rf"(\d{{1,2}}){gap}*\+?{gap}*(?:years?|yrs?)(?:{gap}+of)?"
        rf"(?:{gap}+(?:professional|hands-on|industry|production))?"
        rf"(?:{gap}+experience)?(?:{gap}+(?:in|with|using|building))?{gap}+([A-Za-z+#./ -]{{0,40}})",
        re.IGNORECASE,
    )
    for m in pattern.finditer(lower_bound_years(jd)):
        years = float(m.group(1))
        skills = extract_skills(re.split(r"[,;(]| and | or ", m.group(2))[0])
        key = (skills[0] if skills else None, years)
        if key in seen:
            continue
        seen.add(key)
        label = f"{int(years)}+ years {skills[0]}" if skills else f"{int(years)}+ years experience"
        reqs.append(Requirement(raw=label, skill=skills[0] if skills else None, min_years=years))
    return reqs


def critical_skills_from_jd(jd: str) -> list[str]:
    """Skills listed under Requirements / Must-have style sections (falls back to every skill in the JD)."""
    critical: list[str] = []
    in_required = False
    for line in jd.splitlines():
        low = line.lower().strip()
        if re.match(r"^[#*\s]*(requirements|required|must[- ]haves?|qualifications|what you need|minimum qualifications)", low):
            in_required = True
            continue
        if re.match(r"^[#*\s]*(nice[- ]to[- ]haves?|preferred|bonus|benefits|about us|perks)", low):
            in_required = False
            continue
        if in_required or re.search(r"\b(must|required|requires|mandatory|essential)\b", low):
            critical += [s for s in extract_skills(line) if s not in critical]
    return critical or extract_skills(jd)


# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #
class JobMatcher:
    def __init__(self, db_path: str | Path = "chroma_db", model_name: str | None = None):
        import chromadb

        db_path = Path(db_path)
        registry_path = db_path / REGISTRY_FILE
        if not registry_path.is_file():
            raise MatcherError(
                f"No resume index found at {db_path} (missing {REGISTRY_FILE}). Build one first:\n"
                f"  python resume_rag.py --resumes resumes --db {db_path} --reset"
            )
        try:
            registry = json.loads(registry_path.read_text())
            candidates = registry["candidates"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise MatcherError(f"Index registry {registry_path} is unreadable or malformed: {exc}") from exc
        if not candidates:
            raise MatcherError(f"Index at {db_path} contains no candidates; re-run resume_rag.py.")

        self.candidates: dict[str, dict] = {c["candidate_id"]: c for c in candidates}
        self.model_name = model_name or registry.get("model", DEFAULT_MODEL)
        indexed_with = registry.get("model", DEFAULT_MODEL)
        if self.model_name != indexed_with:
            print(f"[job_matcher] warning: querying with {self.model_name} but the index was built with "
                  f"{indexed_with}; similarities are not comparable across models.", file=sys.stderr)
        self.embedder = Embedder(self.model_name)
        try:
            self.collection = chromadb.PersistentClient(path=str(db_path)).get_collection(COLLECTION_NAME)
        except Exception as exc:  # chromadb raises assorted types for a missing collection
            raise MatcherError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found in {db_path}: {exc}\n"
                f"  python resume_rag.py --resumes resumes --db {db_path} --reset"
            ) from exc
        self._ids = list(self.candidates)
        self._bm25 = BM25Okapi([tokenize(self.candidates[i]["full_text"]) for i in self._ids])

    # ---- retrieval -------------------------------------------------------- #
    def _jd_passages(self, jd: str, max_chars: int = 600) -> list[str]:
        """Split the JD into passages (by section, then by size) so each fits the embedding model."""
        passages: list[str] = []
        for _, body in split_sections(jd) or [("all", jd)]:
            buf = ""
            for line in [l for l in body.splitlines() if l.strip()]:
                if buf and len(buf) + len(line) > max_chars:
                    passages.append(buf)
                    buf = ""
                buf = f"{buf}\n{line}" if buf else line
            if buf:
                passages.append(buf)
        return [jd[:max_chars]] + passages  # whole-JD head + detailed passages

    def semantic_search(self, jd: str, n_chunks: int = 50,
                        timings: dict[str, float] | None = None) -> dict[str, list[dict]]:
        """Return candidate_id -> list of {text, section, similarity}, best similarity per chunk.

        When `timings` is given, the embedding and vector-search phases are measured
        directly into it. Earlier versions derived those phases by subtracting separately
        timed runs from the end-to-end time, which is unsound - the subtraction produced
        negative durations - so nothing here is inferred by residual any more.
        """
        passages = self._jd_passages(jd)
        n = min(n_chunks, self.collection.count())
        if n <= 0:  # an empty collection would make chromadb reject n_results=0
            return {}
        phase = time.perf_counter()
        query_embeddings = self.embedder.embed(passages)
        if timings is not None:
            timings["embed_jd"] = time.perf_counter() - phase
        phase = time.perf_counter()
        res = self.collection.query(query_embeddings=query_embeddings, n_results=n,
                                    include=["documents", "metadatas", "distances"])
        if timings is not None:
            timings["vector_search"] = time.perf_counter() - phase
        best: dict[str, dict] = {}
        for docs, metas, dists in zip(res["documents"], res["metadatas"], res["distances"]):
            for doc, meta, dist in zip(docs, metas, dists):
                sim = 1.0 - dist  # cosine distance -> similarity
                key = f"{meta['candidate_id']}:{meta['chunk_index']}"
                if key not in best or sim > best[key]["similarity"]:
                    best[key] = {"candidate_id": meta["candidate_id"], "text": doc,
                                 "section": meta["section"], "similarity": sim}
        hits: dict[str, list[dict]] = {}
        for hit in best.values():
            hits.setdefault(hit["candidate_id"], []).append(hit)
        for lst in hits.values():
            lst.sort(key=lambda h: h["similarity"], reverse=True)
        return hits

    def keyword_scores(self, jd: str) -> dict[str, float]:
        """BM25 score per candidate, normalised to 0-1."""
        scores = self._bm25.get_scores(tokenize(jd))
        top = max(scores) if len(scores) and max(scores) > 0 else 1.0
        return {cid: float(s) / top for cid, s in zip(self._ids, scores)}

    # ---- scoring ---------------------------------------------------------- #
    @staticmethod
    def _semantic_component(hits: list[dict]) -> float:
        if not hits:
            return 0.0
        sims = [h["similarity"] for h in hits]
        raw = 0.6 * sims[0] + 0.4 * (sum(sims[:3]) / len(sims[:3]))
        return max(0.0, min(1.0, (raw - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)))

    @staticmethod
    def _excerpts(hits: list[dict], matched: list[str], n: int = 3, width: int = 280) -> list[str]:
        """Top chunks, trimmed around the first matched skill where possible."""
        out = []
        for h in hits[:n]:
            text = " ".join(h["text"].split())
            if len(text) > width:
                pos = min((text.lower().find(s.lower()) for s in matched if s.lower() in text.lower()), default=0)
                start = max(0, pos - width // 3)
                text = ("..." if start else "") + text[start:start + width].strip() + "..."
            out.append(f"[{h['section'].title()}] {text}")
        return out

    @staticmethod
    def _reasoning(score: int, cand: dict, matched: list[str], missing: list[str], critical: list[str],
                   section_sims: dict[str, float], req_notes: list[str], min_years: float | None) -> str:
        # No verdict adjective. The thresholds behind "Strong match" / "Good match" were
        # never calibrated against the relevance labels: on the shipped evaluation set 37
        # of 268 candidate/JD pairs a human would call irrelevant were described as a
        # "Good" or "Strong" match, including two DevOps engineers with no ML background
        # for the ML role. The score ranks candidates against one job description; it does
        # not grade them, and the reasoning should not imply that it does.
        parts = [f"Score {score}/100."]
        if critical:
            parts.append(f"Has {len(matched)}/{len(critical)} critical skills"
                         + (f" ({', '.join(matched)})" if matched else "")
                         + (f"; missing {', '.join(missing)}." if missing else "."))
        top_sections = sorted(section_sims.items(), key=lambda kv: kv[1], reverse=True)[:2]
        if top_sections:
            parts.append("Closest sections to the JD: "
                         + ", ".join(f"{s.title()} ({v:.2f})" for s, v in top_sections) + ".")
        exp = f"{cand['experience_years']:g} years of experience"
        if min_years:
            exp += f" (JD asks {min_years:g}+)"
        parts.append(exp + f"; education: {cand['education_level']}.")
        if req_notes:
            parts.append("Must-haves met: " + "; ".join(req_notes) + ".")
        return " ".join(parts)

    def match(self, jd: str, k: int = 10, must_haves: list[str] | None = None, auto_requirements: bool = True,
              critical_skills: list[str] | None = None, min_score: int = 0,
              weights: ScoreWeights | None = None, apply_filters: bool = True,
              timings: dict[str, float] | None = None) -> dict:
        """Rank candidates against `jd`.

        With `apply_filters=False` nobody is excluded: must-haves are still parsed, so
        experience fit and the reasoning still reflect them, but a candidate who fails
        one is ranked and annotated with `failed_requirements` instead of being moved to
        `filtered_out`. That is what the evaluation harness measures, so retrieval
        quality is scored on the ranking rather than on the post-filter shortlist.
        """
        if not jd or not jd.strip():
            raise MatcherError("The job description is empty.")
        started = time.perf_counter()
        weights = weights or DEFAULT_WEIGHTS
        critical = critical_skills or critical_skills_from_jd(jd)
        parsed = [(m, parse_requirement(m)) for m in (must_haves or [])]
        requirements = [r for _, r in parsed if r]
        # A must-have we cannot parse must never be dropped quietly: the caller would
        # believe a filter was applied that never ran.
        unparsed = [raw for raw, r in parsed if r is None]
        for raw in unparsed:
            print(f"[job_matcher] warning: could not interpret must-have {raw!r}; it was NOT applied. "
                  f'Use forms like "5+ years Python", "Kubernetes" or "Master\'s degree".', file=sys.stderr)
        if auto_requirements:
            known = {(r.skill, r.min_years) for r in requirements}
            requirements += [r for r in extract_requirements_from_jd(jd) if (r.skill, r.min_years) not in known]
        total_years_req = max((r.min_years for r in requirements if r.min_years and not r.skill), default=None)
        any_years_req = max((r.min_years for r in requirements if r.min_years), default=None)

        if timings is not None:
            timings["parse_jd"] = time.perf_counter() - started
        sem_hits = self.semantic_search(jd, timings=timings)
        phase = time.perf_counter()
        kw = self.keyword_scores(jd)
        if timings is not None:
            timings["bm25"] = time.perf_counter() - phase
        phase = time.perf_counter()
        # Hoisted out of the per-candidate loop: this runs ~80 compiled patterns over the
        # JD, and recomputing it per candidate made scoring two thirds of query latency.
        jd_skills = extract_skills(jd)

        results, filtered_out = [], []
        for cid, cand in self.candidates.items():
            hits = sem_hits.get(cid, [])
            sem = self._semantic_component(hits)
            matched = [s for s in critical if s in cand["skills"]]
            missing = [s for s in critical if s not in cand["skills"]]
            skill_cov = len(matched) / len(critical) if critical else 0.0
            target_years = total_years_req or any_years_req
            exp_fit = min(1.0, cand["experience_years"] / target_years) if target_years else 1.0
            exact = 100 * (weights.semantic * sem + weights.critical_skills * skill_cov
                           + weights.keyword_bm25 * kw.get(cid, 0.0)
                           + weights.experience_fit * exp_fit)
            score = round(exact)

            checks = [(r, *r.check(cand)) for r in requirements]
            failed = [f"{r.raw} (has {note})" for r, ok, note in checks if not ok]
            if failed and apply_filters:
                filtered_out.append({"candidate_name": cand["candidate_name"], "resume_path": cand["resume_path"],
                                     "match_score": score, "failed_requirements": failed})
                continue
            if score < min_score:
                continue

            section_sims: dict[str, float] = {}
            for h in hits:
                section_sims[h["section"]] = max(section_sims.get(h["section"], 0.0), h["similarity"])
            all_matched = [s for s in jd_skills if s in cand["skills"]]
            results.append({
                "candidate_name": cand["candidate_name"],
                "resume_path": cand["resume_path"],
                "match_score": score,
                "matched_skills": all_matched,
                "relevant_excerpts": self._excerpts(hits, all_matched),
                "reasoning": self._reasoning(score, cand, matched, missing, critical, section_sims,
                                             [f"{r.raw} ({note})" for r, ok, note in checks], target_years),
                "matched_sections": [s for s, _ in sorted(section_sims.items(), key=lambda kv: kv[1], reverse=True)[:3]],
                "score_breakdown": {"semantic": round(100 * sem), "critical_skills": round(100 * skill_cov),
                                    "keyword_bm25": round(100 * kw.get(cid, 0.0)), "experience_fit": round(100 * exp_fit)},
                "experience_years": cand["experience_years"],
                "education": cand["education"],
                "failed_requirements": failed,
                # Ranking uses this, not the rounded match_score: rounding to an integer
                # before sorting threw away the information that separates near-ties and
                # left the alphabetical tie-break to decide them.
                "match_score_exact": round(exact, 4),
            })

        # Rank on the unrounded score, with resume_path as a deterministic tie-break so
        # genuinely equal scores order identically on every run; reproducible ranking is a
        # precondition for the retrieval metrics.
        results.sort(key=lambda r: (-r["match_score_exact"], r["resume_path"]))
        if timings is not None:
            timings["score_and_explain"] = time.perf_counter() - phase
            timings["total"] = time.perf_counter() - started
        return {
            "job_description": jd,
            "critical_skills": critical,
            "must_have_requirements": [r.raw for r in requirements],
            "unparsed_must_haves": unparsed,
            "score_weights": asdict(weights),
            "top_matches": results[:k],
            "filtered_out": sorted(filtered_out, key=lambda r: (-r["match_score"], r["resume_path"])),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Match resumes to a job description (hybrid RAG search).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jd", help="Job description text")
    src.add_argument("--jd-file", help="Path to a job description file")
    parser.add_argument("--db", default="chroma_db", help="ChromaDB directory created by resume_rag.py")
    parser.add_argument("--k", type=int, default=10, help="Number of matches to return")
    parser.add_argument("--must-have", action="append", default=[],
                        help='Must-have requirement, repeatable: "5+ years Python", "Kubernetes", "Master\'s degree"')
    parser.add_argument("--critical-skill", action="append", default=[], help="Override critical skills (repeatable)")
    parser.add_argument("--no-auto-requirements", action="store_true", help="Don't infer 'N+ years X' must-haves from the JD")
    parser.add_argument("--min-score", type=int, default=0, help="Drop matches below this score")
    parser.add_argument("--output", help="Write JSON result to this file (also printed to stdout)")
    parser.add_argument("--brief", action="store_true", help="Print only the fields from the assignment's output format")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if not 0 <= args.min_score <= 100:
        parser.error("--min-score must be between 0 and 100")
    try:
        # Job descriptions go through the same Milestone 1 reader as resumes, so a JD
        # can be a .pdf or .docx just as easily as a .txt.
        jd = load_text_file(args.jd_file) if args.jd_file else args.jd
        result = JobMatcher(args.db).match(jd.strip(), k=args.k, must_haves=args.must_have,
                                           auto_requirements=not args.no_auto_requirements,
                                           critical_skills=args.critical_skill or None, min_score=args.min_score)
    except (MatcherError, LoaderError) as exc:
        parser.exit(2, f"error: {exc}\n")
    if args.brief:
        keep = ["candidate_name", "resume_path", "match_score", "matched_skills", "relevant_excerpts", "reasoning"]
        result = {"job_description": result["job_description"],
                  "top_matches": [{k: m[k] for k in keep} for m in result["top_matches"]]}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
