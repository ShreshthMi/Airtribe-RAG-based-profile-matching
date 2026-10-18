"""Embedding back ends: the sentence-transformers path and the ONNX fallback."""
from __future__ import annotations

import sys

import pytest

from resume_rag import DEFAULT_MODEL, Embedder


def test_an_unavailable_model_is_reported_rather_than_silently_substituted(monkeypatch):
    """Falling back to MiniLM for a model the caller explicitly asked for would make
    similarities incomparable without saying so."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="Could not load"):
        Embedder("BAAI/bge-small-en-v1.5")


@pytest.mark.slow
def test_vectors_are_normalised_so_cosine_distance_is_meaningful():
    embedder = Embedder(DEFAULT_MODEL)
    vectors = embedder.embed(["Python and PyTorch", "Java and Spring Boot"])
    assert len(vectors) == 2
    for vector in vectors:
        assert sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_related_text_scores_higher_than_unrelated_text():
    embedder = Embedder(DEFAULT_MODEL)
    jd, close, far = embedder.embed([
        "machine learning engineer building NLP models in PyTorch",
        "ML engineer training deep learning models with PyTorch for NLP",
        "pastry chef specialising in laminated doughs",
    ])
    similar = sum(a * b for a, b in zip(jd, close))
    different = sum(a * b for a, b in zip(jd, far))
    assert similar > different


@pytest.mark.slow
def test_the_onnx_fallback_serves_the_same_model_without_pytorch(monkeypatch):
    """The README promises `pip install chromadb rank-bm25` is enough to run everything."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = Embedder(DEFAULT_MODEL)
    assert embedder.backend == "onnx (chromadb)"
    vectors = embedder.embed(["Python and PyTorch"])
    assert len(vectors[0]) == 384      # MiniLM-L6 dimensionality
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0, abs=1e-5)
