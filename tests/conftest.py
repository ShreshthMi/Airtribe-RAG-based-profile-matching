"""Shared fixtures.

Integration tests use a deterministic hashed bag-of-words embedder instead of MiniLM.
Lexical overlap still produces real cosine similarity, so a test can assert which
candidate wins, while a whole test run stays under a second instead of paying 12
seconds of model loading. The tests that genuinely need the real model are marked
`slow` and reuse one session-scoped index.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import job_matcher  # noqa: E402
import resume_rag  # noqa: E402

# Every experience figure in the tests is measured against this date, so "Present"
# cannot make an assertion fail next month.
REFERENCE_DATE = date(2026, 1, 1)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder with the Embedder interface."""

    DIMENSIONS = 96

    def __init__(self, model_name: str = "fake"):
        self.model_name = model_name
        self.backend = "fake"

    @staticmethod
    def _bucket(token: str) -> int:
        digest = hashlib.md5(token.encode()).hexdigest()[:8]
        return int(digest, 16) % FakeEmbedder.DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.DIMENSIONS
            for token in _TOKEN_RE.findall(text.lower()):
                vector[self._bucket(token)] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


TINY_RESUMES = {
    "ml_engineer.md": """# Asha Varma
asha@example.com | Pune

## Summary
Machine learning engineer with 7 years of experience shipping NLP and LLM systems.

## Skills
Python, PyTorch, NLP, LLMs, RAG, Hugging Face, Vector Databases, Docker, Kubernetes, AWS

## Experience
Senior Machine Learning Engineer, Northwind
Jan 2019 - Dec 2025
- Built retrieval augmented generation pipelines in Python and PyTorch.
- Deployed models with Docker on Kubernetes on AWS.

## Education
M.Tech in Computer Science, 2018
""",
    "java_backend.txt": """Bruno Sala
bruno@example.com | Milan

SUMMARY
Backend engineer with 8 years of experience building Java microservices.

SKILLS
Java, Spring Boot, Kafka, Redis, Microservices, REST APIs, Kubernetes, AWS

WORK HISTORY
Staff Engineer, Tessera Pay
Jan 2018 - Dec 2025
- Payment microservices in Java and Spring Boot with Kafka events.

EDUCATION
B.Eng. Software Engineering, 2017
""",
    "frontend.txt": """CLARA FONTAINE
clara@example.com | Lyon

PROFILE
Frontend engineer with 3 years of experience in React and TypeScript.

KEY SKILLS
JavaScript, TypeScript, React, HTML, CSS, Figma

EXPERIENCE
Frontend Engineer, Papeterie
Jan 2023 - Dec 2025
- Built the checkout in React and TypeScript.

EDUCATION
B.A. Design, 2022

INTERESTS
Monty Python, playing Go, rock climbing
""",
    "prose_only.txt": """Dev Patel
dev@example.com

I am a junior data analyst with 2 years of experience writing SQL and building
Tableau dashboards. Previously a teacher. No formal sections in this resume.
""",
}

TINY_ML_JD = """Senior Machine Learning Engineer

Requirements
- 5+ years of Python experience
- Strong NLP background, hands-on with PyTorch
- Experience with LLMs, RAG and vector databases
- Docker and Kubernetes on AWS
"""


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory) -> Path:
    """A four-resume corpus covering ML, Java backend, frontend and heading-less prose."""
    directory = tmp_path_factory.mktemp("resumes")
    for name, text in TINY_RESUMES.items():
        (directory / name).write_text(text)
    return directory


@pytest.fixture(scope="session")
def tiny_index(tiny_corpus, tmp_path_factory):
    """(db_path, BuildReport) for the tiny corpus, built with the fake embedder."""
    db_path = tmp_path_factory.mktemp("chroma")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(resume_rag, "Embedder", FakeEmbedder)
        report = resume_rag.build_index(tiny_corpus, db_path, model_name="fake", reset=True,
                                        today=REFERENCE_DATE)
    return db_path, report


@pytest.fixture
def tiny_matcher(tiny_index, monkeypatch):
    """A JobMatcher over the tiny index, using the fake embedder."""
    db_path, _ = tiny_index
    monkeypatch.setattr(job_matcher, "Embedder", FakeEmbedder)
    return job_matcher.JobMatcher(db_path)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
