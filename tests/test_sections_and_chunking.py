"""Section detection and section-aware chunking."""
from __future__ import annotations

from pathlib import Path

import resume_rag


def sections_of(text: str) -> dict[str, str]:
    return dict(resume_rag.split_sections(text))


def test_markdown_all_caps_and_synonym_headings_are_all_recognised():
    text = """# Dana Woods
dana@example.com

## Profile
Engineer.

TECHNICAL SKILLS
Python, SQL

Work History
Engineer, Acme
Jan 2020 - Dec 2021

ACADEMIC BACKGROUND
B.Sc. Computing, 2019
"""
    found = sections_of(text)
    assert set(found) >= {"header", "summary", "skills", "experience", "education"}
    assert "Python, SQL" in found["skills"]


def test_a_line_that_merely_mentions_a_section_word_is_not_a_heading():
    found = sections_of("Led the skills matrix project for the education sector team\n")
    assert set(found) == {"header"}


def test_text_before_the_first_heading_becomes_the_header():
    found = sections_of("Ravi Patel\nravi@example.com\n\nSKILLS\nGo\n")
    assert "ravi@example.com" in found["header"]


def test_a_short_section_is_kept_whole():
    assert resume_rag.chunk_section("skills", "Python, SQL", max_chars=900) == ["Python, SQL"]


def test_a_long_section_splits_on_blank_lines_and_respects_the_size_limit():
    role = "Engineer, Acme\n" + "\n".join(f"- Did thing number {i}" for i in range(12))
    body = "\n\n".join([role] * 4)
    chunks = resume_rag.chunk_section("experience", body, max_chars=400)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_an_oversized_single_block_is_split_by_lines_with_overlap():
    block = "\n".join(f"- A reasonably long bullet point number {i}" for i in range(40))
    chunks = resume_rag.chunk_section("experience", block, max_chars=300)
    assert len(chunks) > 1
    # consecutive chunks share their boundary line, so context is not lost at the seam
    assert chunks[0].splitlines()[-1] == chunks[1].splitlines()[0]


def test_chunks_never_mix_two_sections():
    text = """# Mina Roy
mina@example.com

## Skills
Python, Kubernetes

## Education
PhD in Statistics, 2015
"""
    record = resume_rag.process_resume(Path("mina.md"), text)
    for chunk in record.chunks:
        if chunk.section == "education":
            assert "Kubernetes" not in chunk.text
        if chunk.section == "skills":
            assert "PhD" not in chunk.text


def test_the_header_is_not_embedded_when_real_sections_exist():
    text = "# Nia Dlamini\nnia@example.com | +27 11 555 0000\n\n## Skills\nGo, Rust\n"
    record = resume_rag.process_resume(Path("nia.md"), text)
    assert {chunk.section for chunk in record.chunks} == {"skills"}
    assert all("nia@example.com" not in chunk.text for chunk in record.chunks)


def test_a_resume_with_no_headings_is_still_chunked_and_retrievable():
    """Otherwise it would be indexed with zero chunks and be invisible to search."""
    prose = ("Dev Patel\ndev@example.com\n\n"
             "I am a junior data analyst with 2 years of experience writing SQL "
             "and building Tableau dashboards.\n")
    record = resume_rag.process_resume(Path("dev.txt"), prose)
    assert len(record.chunks) >= 1
    assert record.chunks[0].section == "summary"
    assert "SQL" in record.chunks[0].text


def test_chunk_ids_are_stable_and_unique_per_candidate():
    text = "# Sam Ortiz\nsam@example.com\n\n## Skills\nJava\n\n## Education\nB.E. 2016\n"
    first = resume_rag.process_resume(Path("sam.md"), text, display_path="resumes/sam.md")
    second = resume_rag.process_resume(Path("sam.md"), text, display_path="resumes/sam.md")
    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in second.chunks]
    assert len({chunk.id for chunk in first.chunks}) == len(first.chunks)


def test_the_candidate_id_follows_the_path_so_two_people_can_share_a_name():
    text = "# Priya Sharma\np@example.com\n\n## Skills\nPython\n"
    one = resume_rag.process_resume(Path("a.md"), text, display_path="resumes/priya_sharma.md")
    two = resume_rag.process_resume(Path("b.pdf"), text, display_path="resumes/resume_priya_sharma.pdf")
    assert one.candidate_name == two.candidate_name == "Priya Sharma"
    assert one.candidate_id != two.candidate_id
