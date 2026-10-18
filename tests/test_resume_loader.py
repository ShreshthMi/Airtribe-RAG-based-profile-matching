"""The adapter over Milestone 1's fs_tools, including the properties it inherits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import resume_loader


def test_the_vendored_tools_are_verbatim_copies():
    assert resume_loader.verify_vendored_tools() == []


def test_a_modified_vendored_tool_is_reported(tmp_path, monkeypatch):
    (tmp_path / "fs_tools.py").write_text("# tampered\n")
    (tmp_path / "extractors.py").write_text("# tampered\n")
    monkeypatch.setattr(resume_loader, "VENDORED_TOOLS_DIR", tmp_path)
    monkeypatch.setattr(resume_loader, "tools_dir", lambda: tmp_path)
    problems = resume_loader.verify_vendored_tools()
    assert len(problems) == 2
    assert "does not match Milestone 1 commit" in problems[0]


def test_only_resume_formats_are_listed(tmp_path):
    (tmp_path / "a.txt").write_text("Alice\nEngineer")
    (tmp_path / "b.md").write_text("# Bob")
    (tmp_path / "notes.csv").write_text("x,y")       # fs_tools can read it; not a resume
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    names = [entry["name"] for entry in resume_loader.list_resumes(tmp_path)]
    assert names == ["a.txt", "b.md"]


def test_listing_is_not_recursive_so_a_nested_resume_is_skipped(tmp_path):
    (tmp_path / "top.txt").write_text("Top Candidate")
    nested = tmp_path / "pdfs"
    nested.mkdir()
    (nested / "buried.txt").write_text("Buried Candidate")
    names = [entry["name"] for entry in resume_loader.list_resumes(tmp_path)]
    assert names == ["top.txt"]


def test_hidden_files_are_refused(tmp_path):
    (tmp_path / "visible.txt").write_text("Visible")
    (tmp_path / ".hidden.txt").write_text("Hidden")
    names = [entry["name"] for entry in resume_loader.list_resumes(tmp_path)]
    assert names == ["visible.txt"]


def test_a_missing_directory_raises_rather_than_returning_nothing(tmp_path):
    with pytest.raises(resume_loader.LoaderError, match="not found"):
        resume_loader.list_resumes(tmp_path / "nope")


def test_load_resumes_reports_display_paths_and_metadata(tmp_path):
    (tmp_path / "ada.txt").write_text("Ada Lovelace\nEngineer")
    loaded, skipped = resume_loader.load_resumes(tmp_path)
    assert skipped == []
    assert len(loaded) == 1
    document = loaded[0]
    assert document.display_path == f"{tmp_path.name}/ada.txt"
    assert document.text.startswith("Ada Lovelace")
    assert document.metadata["extension"] == ".txt"
    assert document.metadata["word_count"] > 0


def test_an_empty_resume_is_skipped_with_a_reason_not_silently_dropped(tmp_path):
    (tmp_path / "real.txt").write_text("Real Candidate")
    (tmp_path / "blank.txt").write_text("   \n\n")
    loaded, skipped = resume_loader.load_resumes(tmp_path)
    assert [document.metadata["name"] for document in loaded] == ["real.txt"]
    assert len(skipped) == 1
    assert skipped[0].display_path.endswith("blank.txt")
    assert skipped[0].reason


def test_a_corrupt_file_is_skipped_without_aborting_the_batch(tmp_path):
    (tmp_path / "good.txt").write_text("Good Candidate")
    (tmp_path / "broken.docx").write_bytes(b"not a zip archive at all")
    loaded, skipped = resume_loader.load_resumes(tmp_path)
    assert [document.metadata["name"] for document in loaded] == ["good.txt"]
    assert len(skipped) == 1 and skipped[0].display_path.endswith("broken.docx")


def test_the_sandbox_root_is_restored_afterwards(tmp_path, monkeypatch):
    monkeypatch.setenv("FS_ASSISTANT_ROOT", "/some/original/root")
    resume_loader.list_resumes(tmp_path)
    assert os.environ["FS_ASSISTANT_ROOT"] == "/some/original/root"


def test_the_sandbox_variable_is_removed_again_when_it_was_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("FS_ASSISTANT_ROOT", raising=False)
    resume_loader.list_resumes(tmp_path)
    assert "FS_ASSISTANT_ROOT" not in os.environ


def test_the_sandbox_refuses_paths_outside_the_resume_directory(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not read me")
    inner = tmp_path / "resumes"
    inner.mkdir()
    (inner / "candidate.txt").write_text("Candidate")
    sys.path.insert(0, str(resume_loader.tools_dir()))
    import fs_tools

    with resume_loader.sandbox_root(inner):
        escape = fs_tools.read_file("../secret.txt")
        absolute = fs_tools.read_file(str(secret))
    assert escape["success"] is False and escape["content"] is None
    assert absolute["success"] is False


def test_load_text_file_reads_a_job_description(tmp_path):
    jd = tmp_path / "jd.txt"
    jd.write_text("Senior Engineer\n\nRequirements\n- 5+ years Python\n")
    assert "5+ years Python" in resume_loader.load_text_file(jd)


def test_load_text_file_raises_on_a_missing_or_empty_file(tmp_path):
    with pytest.raises(resume_loader.LoaderError, match="File not found"):
        resume_loader.load_text_file(tmp_path / "absent.txt")
    blank = tmp_path / "blank.txt"
    blank.write_text("\n")
    with pytest.raises(resume_loader.LoaderError, match="no extractable text"):
        resume_loader.load_text_file(blank)


def test_a_missing_tools_directory_is_reported_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv(resume_loader.TOOLS_DIR_ENV_VAR, str(tmp_path))
    with pytest.raises(resume_loader.LoaderError, match="Milestone 1 tools not found"):
        resume_loader.list_resumes(tmp_path)


# --- the formats the swap to fs_tools was made for ------------------------- #
def test_docx_tables_and_headers_are_read(project_root):
    """The stand-in loader read DOCX paragraphs only, losing table-based skills."""
    docx = project_root / "resumes" / "resume_elena_rodriguez.docx"
    if not docx.is_file():
        pytest.skip("dataset fixture not present")
    text = resume_loader.load_text_file(docx)
    assert "Terraform" in text          # lives in a table cell
    assert "elena.rodriguez@example.com" in text   # lives in the document header


def test_a_multi_page_pdf_is_read(project_root):
    pdf = project_root / "resumes" / "resume_priya_sharma.pdf"
    if not pdf.is_file():
        pytest.skip("dataset fixture not present")
    text = resume_loader.load_text_file(pdf)
    assert "Priya Sharma" in text
    assert len(text) > 500
