"""The two command-line entry points, including their failure exits."""
from __future__ import annotations

import json

import pytest

import job_matcher
import resume_rag
from tests.conftest import REFERENCE_DATE, TINY_ML_JD, FakeEmbedder


def run(module, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", [module.__name__] + argv)
    monkeypatch.setattr(module, "Embedder", FakeEmbedder)
    module.main()


# --- resume_rag.py --------------------------------------------------------- #
def test_indexing_prints_a_summary_with_timings(tiny_corpus, tmp_path, monkeypatch, capsys):
    run(resume_rag, ["--resumes", str(tiny_corpus), "--db", str(tmp_path / "db"),
                     "--reset", "--as-of", REFERENCE_DATE.isoformat()], monkeypatch)
    out = capsys.readouterr().out
    assert "Indexed 4 resumes" in out
    assert "ms/resume" in out
    assert "Asha Varma" in out


def test_the_registry_records_how_the_index_was_built(tiny_corpus, tmp_path, monkeypatch):
    db = tmp_path / "db"
    run(resume_rag, ["--resumes", str(tiny_corpus), "--db", str(db), "--reset",
                     "--as-of", "2026-01-01"], monkeypatch)
    registry = json.loads((db / "candidates.json").read_text())
    assert registry["loader"] == "milestone1 fs_tools"
    assert registry["as_of"] == "2026-01-01"
    assert registry["chunk_count"] >= 4
    assert "timings_seconds" in registry and registry["timings_seconds"]["total"] > 0
    assert registry["skipped"] == []


def test_a_bad_as_of_date_is_rejected_before_any_work(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run(resume_rag, ["--resumes", str(tmp_path), "--db", str(tmp_path / "db"),
                         "--as-of", "last tuesday"], monkeypatch)
    assert exit_info.value.code == 2
    assert "--as-of must be an ISO date" in capsys.readouterr().err


def test_an_empty_resume_directory_exits_with_a_message(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No readable resumes"):
        run(resume_rag, ["--resumes", str(empty), "--db", str(tmp_path / "db")], monkeypatch)


# --- job_matcher.py -------------------------------------------------------- #
def test_brief_output_is_exactly_the_assignments_schema(tiny_index, tmp_path, monkeypatch, capsys):
    db_path, _ = tiny_index
    jd = tmp_path / "jd.txt"
    jd.write_text(TINY_ML_JD)
    run(job_matcher, ["--jd-file", str(jd), "--db", str(db_path), "--brief"], monkeypatch)
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"job_description", "top_matches"}
    for match in payload["top_matches"]:
        assert set(match) == {"candidate_name", "resume_path", "match_score",
                              "matched_skills", "relevant_excerpts", "reasoning"}


def test_output_is_written_to_a_file_as_well_as_stdout(tiny_index, tmp_path, monkeypatch, capsys):
    db_path, _ = tiny_index
    destination = tmp_path / "out.json"
    run(job_matcher, ["--jd", TINY_ML_JD, "--db", str(db_path), "--output", str(destination)],
        monkeypatch)
    written = json.loads(destination.read_text())
    assert written == json.loads(capsys.readouterr().out)
    assert written["top_matches"]


def test_a_job_description_can_be_passed_inline(tiny_index, monkeypatch, capsys):
    db_path, _ = tiny_index
    run(job_matcher, ["--jd", "Frontend engineer with React and TypeScript", "--db", str(db_path)],
        monkeypatch)
    assert json.loads(capsys.readouterr().out)["top_matches"]


@pytest.mark.parametrize("bad_args, message", [
    (["--k", "0"], "--k must be at least 1"),
    (["--min-score", "500"], "--min-score must be between 0 and 100"),
])
def test_out_of_range_arguments_are_rejected(tiny_index, monkeypatch, capsys, bad_args, message):
    db_path, _ = tiny_index
    with pytest.raises(SystemExit) as exit_info:
        run(job_matcher, ["--jd", "anything", "--db", str(db_path)] + bad_args, monkeypatch)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_a_missing_index_exits_cleanly_rather_than_with_a_traceback(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run(job_matcher, ["--jd", "anything", "--db", str(tmp_path / "nope")], monkeypatch)
    assert exit_info.value.code == 2
    assert "error: No resume index found" in capsys.readouterr().err


def test_a_missing_job_description_file_exits_cleanly(tiny_index, tmp_path, monkeypatch, capsys):
    db_path, _ = tiny_index
    with pytest.raises(SystemExit) as exit_info:
        run(job_matcher, ["--jd-file", str(tmp_path / "absent.txt"), "--db", str(db_path)], monkeypatch)
    assert exit_info.value.code == 2
    assert "error: File not found" in capsys.readouterr().err


def test_a_job_description_may_itself_be_a_docx(tiny_index, project_root, monkeypatch, capsys):
    """JDs go through the same Milestone 1 reader as resumes, so binary formats work."""
    db_path, _ = tiny_index
    docx = project_root / "resumes" / "resume_elena_rodriguez.docx"
    if not docx.is_file():
        pytest.skip("dataset fixture not present")
    run(job_matcher, ["--jd-file", str(docx), "--db", str(db_path), "--brief",
                      "--no-auto-requirements"], monkeypatch)
    payload = json.loads(capsys.readouterr().out)
    # the DOCX was read through fs_tools, tables and all, and used as the query text
    assert "Terraform" in payload["job_description"]
    assert payload["top_matches"]


def test_re_indexing_without_reset_drops_chunks_for_a_deleted_resume(tmp_path, monkeypatch):
    """Without this, a removed resume stayed searchable in the collection forever."""
    corpus = tmp_path / "resumes"
    corpus.mkdir()
    (corpus / "keep.txt").write_text("Keep Me\nk@example.com\n\nSKILLS\nPython\n")
    (corpus / "drop.txt").write_text("Drop Me\nd@example.com\n\nSKILLS\nKotlin\n")
    db = tmp_path / "db"

    run(resume_rag, ["--resumes", str(corpus), "--db", str(db), "--reset"], monkeypatch)
    first = json.loads((db / "candidates.json").read_text())
    assert {c["candidate_name"] for c in first["candidates"]} == {"Keep Me", "Drop Me"}

    (corpus / "drop.txt").unlink()
    run(resume_rag, ["--resumes", str(corpus), "--db", str(db)], monkeypatch)   # no --reset

    second = json.loads((db / "candidates.json").read_text())
    assert {c["candidate_name"] for c in second["candidates"]} == {"Keep Me"}

    import chromadb
    collection = chromadb.PersistentClient(path=str(db)).get_collection(resume_rag.COLLECTION_NAME)
    names = {meta["candidate_name"] for meta in collection.get(include=["metadatas"])["metadatas"]}
    assert names == {"Keep Me"}, "stale chunks for the deleted resume are still in the index"
