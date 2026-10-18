"""
Part A: RAG system setup for resumes.

Pipeline
  1. Load resumes with the file-system tools (file_tools.py).
  2. Split each resume into sections (Summary, Skills, Experience, Education, ...)
     and chunk inside a section only, so a chunk never mixes Education with Experience.
  3. Extract metadata: name, skills, total experience years, per-skill years, education.
  4. Embed chunks with a HuggingFace sentence-transformers model.
  5. Store chunks + metadata in a persistent ChromaDB collection; also write a
     candidates.json registry (full text + metadata) used for keyword/BM25 search.

Usage
  python resume_rag.py --resumes resumes --db chroma_db
  python resume_rag.py --resumes resumes --db chroma_db --reset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

from file_tools import iter_documents, relative_path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "resume_chunks"
REGISTRY_FILE = "candidates.json"

# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective", "about me", "about"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tech stack", "key skills"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "career history"],
    "education": ["education", "academic background", "academics", "qualifications"],
    "projects": ["projects", "key projects", "personal projects", "selected projects"],
    "certifications": ["certifications", "certificates", "licenses", "awards", "achievements"],
    "publications": ["publications", "research"],
}
_HEADING_LOOKUP = {alias: canon for canon, aliases in SECTION_ALIASES.items() for alias in aliases}


def _match_heading(line: str) -> str | None:
    """Return canonical section name if `line` looks like a section heading."""
    cleaned = re.sub(r"^[#*\s]+|[*:\s]+$", "", line.strip()).lower()
    if not cleaned or len(cleaned) > 40:
        return None
    return _HEADING_LOOKUP.get(cleaned)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split resume text into (section, body) pairs. Text before the first heading is 'header'."""
    sections: list[tuple[str, list[str]]] = [("header", [])]
    for line in text.splitlines():
        heading = _match_heading(line)
        if heading:
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "\n".join(lines).strip()]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_section(section: str, body: str, max_chars: int = 900, overlap_lines: int = 1) -> list[str]:
    """Chunk a section body without crossing section boundaries.

    Short sections stay whole. Long sections (typically Experience) are split on
    blank lines (one block per role/project) and blocks are packed up to
    `max_chars`; an oversized block is split by lines with a small overlap.
    """
    if len(body) <= max_chars:
        return [body]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            lines = block.splitlines()
            buf: list[str] = []
            for line in lines:
                if buf and len("\n".join(buf + [line])) > max_chars:
                    chunks.append("\n".join(buf))
                    buf = buf[-overlap_lines:] if overlap_lines else []
                buf.append(line)
            if buf:
                chunks.append("\n".join(buf))
        elif current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# Skill vocabulary (shared with job_matcher.py)
# --------------------------------------------------------------------------- #
SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python"], "Java": ["java"], "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript"], "Go": ["golang", "go lang"], "Rust": ["rust"], "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", ".net"], "Scala": ["scala"], "Kotlin": ["kotlin"], "Swift": ["swift"],
    "Ruby": ["ruby", "rails", "ruby on rails"], "PHP": ["php"], "R": ["r programming", "rstudio"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql"],
    "Machine Learning": ["machine learning", "ml models", "ml engineer", "ml pipelines"],
    "Deep Learning": ["deep learning", "neural networks", "neural network"],
    "NLP": ["nlp", "natural language processing"], "Computer Vision": ["computer vision", "opencv", "image recognition"],
    "LLMs": ["llm", "llms", "large language models", "gpt", "generative ai", "genai"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "TensorFlow": ["tensorflow", "keras"], "PyTorch": ["pytorch", "torch"], "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"], "NumPy": ["numpy"], "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop", "hdfs"], "Airflow": ["airflow"], "Kafka": ["kafka"], "dbt": ["dbt"],
    "Snowflake": ["snowflake"], "MLOps": ["mlops", "mlflow", "kubeflow"], "Hugging Face": ["hugging face", "huggingface", "transformers library"],
    "LangChain": ["langchain"], "Vector Databases": ["vector database", "vector databases", "chromadb", "pinecone", "weaviate", "faiss"],
    "Statistics": ["statistics", "statistical modeling", "a/b testing", "hypothesis testing"],
    "Data Visualization": ["data visualization", "tableau", "power bi", "matplotlib", "seaborn"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"], "GCP": ["gcp", "google cloud", "bigquery", "vertex ai"],
    "Azure": ["azure"], "Docker": ["docker", "containers"], "Kubernetes": ["kubernetes", "k8s", "eks", "gke"],
    "Terraform": ["terraform"], "CI/CD": ["ci/cd", "jenkins", "github actions", "gitlab ci", "circleci"],
    "Linux": ["linux", "bash", "shell scripting"], "Git": ["git"],
    "React": ["react", "react.js", "reactjs"], "Angular": ["angular"], "Vue": ["vue", "vue.js"],
    "Node.js": ["node.js", "nodejs", "node", "express"], "Django": ["django"], "Flask": ["flask"], "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "spring"], "GraphQL": ["graphql"], "REST APIs": ["rest api", "rest apis", "restful"],
    "Microservices": ["microservices", "microservice"], "MongoDB": ["mongodb", "mongo"], "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "opensearch"], "HTML/CSS": ["html", "css", "tailwind"],
    "iOS": ["ios"], "Android": ["android"], "Figma": ["figma"],
    "Agile": ["agile", "scrum", "kanban"], "Project Management": ["project management", "jira", "pmp"],
    "Leadership": ["team lead", "led a team", "managed a team", "mentored", "leadership"],
    "Excel": ["excel", "spreadsheets"], "Selenium": ["selenium"], "System Design": ["system design", "distributed systems"],
}

_SKILL_PATTERNS = {
    canon: re.compile(
        r"(?<![\w+#.])(" + "|".join(re.escape(a) for a in sorted(aliases + [canon.lower()], key=len, reverse=True)) + r")(?![\w+#])",
        re.IGNORECASE,
    )
    for canon, aliases in SKILL_ALIASES.items()
}
# A bare "Go" or "R" is too ambiguous in lowercase prose; accept it only as a capitalised token in a list.
_SPECIAL_CASE = {"Go": re.compile(r"(?:^|[,|/•(\s])Go(?=[,|/)\s]|$)"), "R": re.compile(r"(?:^|[,|/•(\s])R(?=[,|/)]|$)")}


# Using a framework implies using its language (counts toward per-skill years of experience).
SKILL_IMPLIES = {
    "PyTorch": "Python", "TensorFlow": "Python", "scikit-learn": "Python", "Pandas": "Python", "NumPy": "Python",
    "Django": "Python", "Flask": "Python", "FastAPI": "Python", "Hugging Face": "Python", "LangChain": "Python",
    "Spring Boot": "Java", "React": "JavaScript", "Angular": "TypeScript", "Node.js": "JavaScript",
}


def extract_skills(text: str) -> list[str]:
    """Return canonical skill names mentioned in `text`."""
    found = []
    for canon, pattern in _SKILL_PATTERNS.items():
        special = _SPECIAL_CASE.get(canon)
        if pattern.search(text) or (special and special.search(text)):
            found.append(canon)
    return found


# --------------------------------------------------------------------------- #
# Experience / education / name extraction
# --------------------------------------------------------------------------- #
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_DATE = r"(?:(?P<{p}m>[A-Za-z]{{3,9}}\.?)\s+|(?P<{p}n>\d{{1,2}})/)?(?P<{p}y>(?:19|20)\d{{2}})"
_RANGE_RE = re.compile(
    _DATE.format(p="s") + r"\s*(?:-|–|—|to)\s*(?:" + _DATE.format(p="e") + r"|(?P<present>present|current|now|today))",
    re.IGNORECASE,
)


def _to_month_index(month_name: str | None, month_num: str | None, year: str, default_month: int) -> int:
    month = default_month
    if month_name:
        month = _MONTHS.get(month_name[:3].lower(), default_month)
    elif month_num:
        month = max(1, min(12, int(month_num)))
    return int(year) * 12 + (month - 1)


def find_date_ranges(text: str, today: date | None = None) -> list[tuple[int, int, int, int]]:
    """Return (start_month, end_month, match_start, match_end) for every date range found."""
    today = today or date.today()
    ranges = []
    for m in _RANGE_RE.finditer(text):
        start = _to_month_index(m.group("sm"), m.group("sn"), m.group("sy"), 1)
        if m.group("present"):
            end = today.year * 12 + today.month - 1
        else:
            end = _to_month_index(m.group("em"), m.group("en"), m.group("ey"), 12)
        if end >= start:
            ranges.append((start, end, m.start(), m.end()))
    return ranges


def _merged_years(intervals: list[tuple[int, int]]) -> float:
    """Total years covered by month intervals, merging overlaps."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total, (cur_s, cur_e) = 0, intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
    total += cur_e - cur_s + 1
    return round(total / 12, 1)


def extract_experience(experience_text: str, full_text: str) -> tuple[float, dict[str, float]]:
    """Total years of experience and years per skill.

    Each date range in the Experience section starts a role; the role's text runs
    until the next date range. Skills mentioned in a role inherit its duration.
    Explicit statements like "6 years of Python" or "8+ years of experience" are
    used when they give a higher number.
    """
    ranges = find_date_ranges(experience_text)
    role_intervals: list[tuple[int, int]] = []
    per_skill: dict[str, list[tuple[int, int]]] = {}
    for i, (s, e, _, m_end) in enumerate(ranges):
        role_intervals.append((s, e))
        # role text = from the line containing this range up to the next range's line
        role_start = experience_text.rfind("\n", 0, ranges[i][2]) + 1
        prev_line_start = experience_text.rfind("\n", 0, max(role_start - 1, 0)) + 1  # title line above dates
        role_end = experience_text.rfind("\n", 0, ranges[i + 1][2]) if i + 1 < len(ranges) else len(experience_text)
        role_text = experience_text[prev_line_start:max(role_end, m_end)]
        role_skills = set(extract_skills(role_text))
        role_skills |= {SKILL_IMPLIES[k] for k in role_skills if k in SKILL_IMPLIES}
        for skill in role_skills:
            per_skill.setdefault(skill, []).append((s, e))

    total_years = _merged_years(role_intervals)
    skill_years = {k: _merged_years(v) for k, v in per_skill.items()}

    for m in re.finditer(r"(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+|hands-on\s+)?experience",
                         full_text, re.IGNORECASE):
        total_years = max(total_years, float(m.group(1)))
    for m in re.finditer(r"(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience\s+(?:in|with)\s+)?([A-Za-z+#./ -]{1,40})",
                         full_text, re.IGNORECASE):
        for skill in extract_skills(m.group(2)):
            skill_years[skill] = max(skill_years.get(skill, 0.0), float(m.group(1)))
    for m in re.finditer(r"([A-Za-z+#./ -]{1,30})\((\d{1,2})\+?\s*(?:years?|yrs?)\)", full_text, re.IGNORECASE):
        for skill in extract_skills(m.group(1).split(",")[-1]):
            skill_years[skill] = max(skill_years.get(skill, 0.0), float(m.group(2)))
    return total_years, dict(sorted(skill_years.items()))


EDUCATION_LEVELS = [  # (level, rank, pattern) - highest first
    ("PhD", 4, r"\b(ph\.?\s?d|doctorate|doctor of philosophy)(?![a-z])"),
    ("Master", 3, r"\b(master'?s?|m\.?\s?sc?\.?|m\.?tech|m\.e\.|mba|m\.a\.)(?![a-z])"),
    ("Bachelor", 2, r"\b(bachelor'?s?|b\.?\s?sc?\.?|b\.?tech|b\.e\.|b\.a\.|undergraduate)(?![a-z])"),
    ("Associate", 1, r"\b(associate'?s? degree|diploma)(?![a-z])"),
]


def extract_education(education_text: str) -> tuple[str, str, int]:
    """Return (highest_level, summary_line, rank)."""
    lines = [l.strip(" -•*\t") for l in education_text.splitlines() if l.strip()]
    for level, rank, pattern in EDUCATION_LEVELS:
        for line in lines:
            if re.search(pattern, line, re.IGNORECASE):
                return level, line, rank
    return "Unknown", (lines[0] if lines else ""), 0


def extract_name(header_text: str, path: Path) -> str:
    """Name from a 'Name:' field, else the first short line of the header, else the file name."""
    m = re.search(r"^\s*name\s*[:\-]\s*(.+)$", header_text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    for line in header_text.splitlines():
        line = re.sub(r"^[#*\s]+|[*\s]+$", "", line)
        if line and len(line.split()) <= 5 and not re.search(r"[@\d|]", line):
            return line
    return path.stem.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    id: str
    text: str
    section: str
    chunk_index: int


@dataclass
class ResumeRecord:
    candidate_id: str
    candidate_name: str
    resume_path: str
    skills: list[str]
    experience_years: float
    skill_years: dict[str, float]
    education_level: str
    education: str
    education_rank: int
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)


def process_resume(path: Path, text: str, base_dir: Path | None = None, max_chars: int = 900) -> ResumeRecord:
    """Parse one resume into metadata + section-aware chunks."""
    sections = split_sections(text)
    by_name: dict[str, str] = {}
    for name, body in sections:
        by_name[name] = f"{by_name[name]}\n\n{body}" if name in by_name else body

    name = extract_name(by_name.get("header", text[:300]), path)
    total_years, skill_years = extract_experience(by_name.get("experience", ""), text)
    level, edu_line, rank = extract_education(by_name.get("education", ""))
    rel_path = relative_path(path, base_dir)
    candidate_id = hashlib.sha1(rel_path.encode()).hexdigest()[:12]

    chunks: list[Chunk] = []
    for section, body in sections:
        if section == "header":
            continue  # contact details only; name is kept as metadata
        for piece in chunk_section(section, body, max_chars=max_chars):
            idx = len(chunks)
            chunks.append(Chunk(id=f"{candidate_id}-{idx}", text=piece, section=section, chunk_index=idx))

    return ResumeRecord(
        candidate_id=candidate_id, candidate_name=name, resume_path=rel_path,
        skills=extract_skills(text), experience_years=total_years, skill_years=skill_years,
        education_level=level, education=edu_line, education_rank=rank,
        sections={k: v for k, v in by_name.items() if k != "header"}, full_text=text, chunks=chunks,
    )


# --------------------------------------------------------------------------- #
# Embeddings + vector store
# --------------------------------------------------------------------------- #
class Embedder:
    """HuggingFace embedding model, returning L2-normalised vectors (cosine similarity).

    Uses sentence-transformers when installed. If it isn't (or the HuggingFace hub
    is unreachable) and the model is all-MiniLM-L6-v2, falls back to ChromaDB's
    bundled ONNX export of the same model, which needs no PyTorch.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            self._encode = lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.backend = "sentence-transformers"
        except Exception as exc:
            if not model_name.endswith("all-MiniLM-L6-v2"):
                raise RuntimeError(f"Could not load {model_name} with sentence-transformers: {exc}") from exc
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            onnx_model = ONNXMiniLM_L6_V2()
            self._encode = lambda texts: onnx_model(texts)
            self.backend = "onnx (chromadb)"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vectors = np.asarray(self._encode(texts), dtype=np.float32)
        vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return vectors.tolist()


def get_collection(db_path: str | Path, reset: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def chunk_metadata(record: ResumeRecord, chunk: Chunk) -> dict:
    """Flat metadata stored with each chunk (Chroma metadata values must be scalars)."""
    return {
        "candidate_id": record.candidate_id,
        "candidate_name": record.candidate_name,
        "resume_path": record.resume_path,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "skills": ", ".join(record.skills),
        "experience_years": float(record.experience_years),
        "education_level": record.education_level,
        "education_rank": record.education_rank,
    }


def build_index(resume_dir: str | Path, db_path: str | Path, model_name: str = DEFAULT_MODEL,
                reset: bool = False, max_chars: int = 900) -> list[ResumeRecord]:
    """Load, chunk, embed and store every resume in `resume_dir`."""
    resume_dir, db_path = Path(resume_dir), Path(db_path)
    records = [process_resume(p, t, base_dir=resume_dir.parent, max_chars=max_chars) for p, t in iter_documents(resume_dir)]
    if not records:
        raise SystemExit(f"No resumes found in {resume_dir}")

    embedder = Embedder(model_name)
    collection = get_collection(db_path, reset=reset)
    for record in records:
        # drop stale chunks for this candidate before re-adding (idempotent re-index)
        collection.delete(where={"candidate_id": record.candidate_id})
        # the embedded text carries the section label so "Education: ..." and "Experience: ..." stay distinct
        docs = [f"{c.section.title()}: {c.text}" for c in record.chunks]
        collection.add(
            ids=[c.id for c in record.chunks],
            documents=[c.text for c in record.chunks],
            embeddings=embedder.embed(docs),
            metadatas=[chunk_metadata(record, c) for c in record.chunks],
        )

    registry = {
        "model": model_name,
        "candidates": [{k: v for k, v in asdict(r).items() if k != "chunks"} for r in records],
    }
    db_path.mkdir(parents=True, exist_ok=True)
    (db_path / REGISTRY_FILE).write_text(json.dumps(registry, indent=2))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Index resumes into ChromaDB for RAG-based matching.")
    parser.add_argument("--resumes", default="resumes", help="Directory of resumes (.txt/.md/.pdf/.docx)")
    parser.add_argument("--db", default="chroma_db", help="ChromaDB persistence directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--max-chars", type=int, default=900, help="Max characters per chunk")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before indexing")
    args = parser.parse_args()

    records = build_index(args.resumes, args.db, args.model, args.reset, args.max_chars)
    total_chunks = sum(len(r.chunks) for r in records)
    print(f"Indexed {len(records)} resumes / {total_chunks} chunks into {args.db} ({args.model})\n")
    for r in records:
        print(f"- {r.candidate_name:<20} {r.experience_years:>4} yrs  {r.education_level:<9} "
              f"chunks={len(r.chunks):<2} skills={', '.join(r.skills[:8])}{' ...' if len(r.skills) > 8 else ''}")


if __name__ == "__main__":
    main()
