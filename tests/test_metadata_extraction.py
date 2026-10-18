"""Skills, experience years, education and name extraction."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import resume_rag
from resume_rag import extract_education, extract_experience, extract_name, extract_skills

TODAY = date(2026, 1, 1)


# --- skills ---------------------------------------------------------------- #
@pytest.mark.parametrize("text, expected", [
    ("k8s and Docker", "Kubernetes"),
    ("sklearn pipelines", "scikit-learn"),
    ("PYTHON automation scripts", "Python"),
    ("Retrieval-Augmented Generation", "RAG"),
    ("huggingface transformers", "Hugging Face"),
    ("Experience with PySpark", "Spark"),
])
def test_aliases_and_casing_resolve_to_the_canonical_skill(text, expected):
    assert expected in extract_skills(text)


def test_a_skill_inside_a_longer_word_is_not_matched():
    assert "Swift" not in extract_skills("Built the Swiftcart checkout flow")
    assert "Java" not in extract_skills("Wrote JavaScript and TypeScript")
    assert "Go" not in extract_skills("A good goal for the organisation")


def test_java_is_still_found_when_it_is_really_mentioned():
    assert "Java" in extract_skills("Java 17 and Spring Boot services")


def test_bare_go_and_r_are_matched_only_as_list_items():
    assert "Go" in extract_skills("Languages: Go, Python, Bash")
    assert "R" in extract_skills("Python, R, SQL")


def test_a_hobby_that_contains_a_skill_name_is_not_a_skill():
    assert extract_skills("Interests: rock climbing and Monty Python") == []


def test_interests_sections_do_not_contribute_skills():
    text = """CLARA FONTAINE
clara@example.com

KEY SKILLS
React, TypeScript

INTERESTS
Monty Python, playing Go
"""
    record = resume_rag.process_resume(Path("clara.txt"), text)
    assert "React" in record.skills
    assert "Go" not in record.skills          # only mentioned under INTERESTS
    # the section is still chunked, so it remains retrievable
    assert "interests" in {chunk.section for chunk in record.chunks}


def test_warehouses_are_their_own_skills_rather_than_rolling_up_into_a_cloud():
    """A JD asking for BigQuery should not silently demand all of GCP."""
    assert extract_skills("Modelling in BigQuery") == ["BigQuery"]
    assert set(extract_skills("GCP (Vertex AI, BigQuery)")) == {"GCP", "BigQuery"}


# --- experience years ------------------------------------------------------ #
def test_overlapping_roles_are_not_double_counted():
    text = """Engineer, Acme
Jan 2018 - Dec 2021
- Work.

Consultant, Beta
Jan 2020 - Dec 2022
- More work.
"""
    total, _ = extract_experience(text, text, today=TODAY)
    assert total == pytest.approx(5.0)        # Jan 2018 - Dec 2022, not 4 + 3


def test_a_career_gap_is_not_bridged():
    text = """Analyst, Acme
Jan 2018 - Dec 2019

Analyst, Beta
Jan 2022 - Dec 2022
"""
    total, _ = extract_experience(text, text, today=TODAY)
    assert total == pytest.approx(3.0)        # 2 years + 1 year, gap excluded


@pytest.mark.parametrize("dates, expected", [
    ("Jan 2020 - Dec 2021", 2.0),
    ("2020 - 2021", 2.0),
    ("03/2020 - 02/2021", 1.0),
    ("Jan 2020 – Dec 2020", 1.0),            # en dash
    ("January 2020 to December 2020", 1.0),
])
def test_date_range_formats(dates, expected):
    text = f"Engineer, Acme\n{dates}\n- Work.\n"
    total, _ = extract_experience(text, text, today=TODAY)
    assert total == pytest.approx(expected)


def test_present_is_measured_against_the_reference_date():
    text = "Engineer, Acme\nJan 2024 - Present\n- Python work.\n"
    assert extract_experience(text, text, today=date(2026, 1, 1))[0] == pytest.approx(2.1)
    assert extract_experience(text, text, today=date(2025, 1, 1))[0] == pytest.approx(1.1)


def test_per_skill_years_stay_inside_the_role_that_mentions_them():
    text = """Senior MLOps Engineer, Grab
Jan 2018 - Dec 2023
- Built a Kubeflow platform on Kubernetes.

Computer Vision Engineer, Shopee
Jan 2024 - Dec 2024
- OpenCV pipelines.
"""
    total, per_skill = extract_experience(text, text, today=TODAY)
    assert total == pytest.approx(7.0)
    assert per_skill["MLOps"] == pytest.approx(6.0)             # not the full 7
    assert per_skill["Computer Vision"] == pytest.approx(1.0)   # not the full 7


def test_per_skill_years_work_when_dates_sit_on_the_title_line():
    text = """Senior Product Manager - Granite Cloud (2022 - 2024)
Partnered with the Java backend team.

Product Manager - Bluebird Payroll (2018 - 2021)
Owned payroll approvals.
"""
    _, per_skill = extract_experience(text, text, today=TODAY)
    assert per_skill["Java"] == pytest.approx(3.0)   # the Granite role only


def test_a_framework_counts_towards_its_language():
    text = "ML Engineer, Acme\nJan 2020 - Dec 2021\n- Trained models in PyTorch.\n"
    _, per_skill = extract_experience(text, text, today=TODAY)
    assert per_skill["Python"] == pytest.approx(2.0)


def test_an_explicit_years_claim_raises_but_never_lowers_the_total():
    text = "Engineer, Acme\nJan 2024 - Dec 2024\n- Work.\n"
    claim = "Engineer with 9 years of experience.\n" + text
    assert extract_experience(text, claim, today=TODAY)[0] == pytest.approx(9.0)
    modest = "Engineer with 1 year of experience.\n" + text
    assert extract_experience(text, modest, today=TODAY)[0] == pytest.approx(1.0)


def test_a_resume_with_no_dates_yields_zero_rather_than_failing():
    assert extract_experience("", "No dates anywhere here.", today=TODAY) == (0.0, {})


# --- education ------------------------------------------------------------- #
@pytest.mark.parametrize("line, level", [
    ("Ph.D. in Statistics, Stanford", "PhD"),
    ("M.Tech in Computer Science, IIT Delhi", "Master"),
    ("M.Eng. Computer Science, MIT", "Master"),
    ("MBA, INSEAD", "Master"),
    ("B.Eng. Software Engineering, TU Delft", "Bachelor"),
    ("B.Sc. Computer Science, 2018", "Bachelor"),
    ("Associate's degree in IT", "Associate"),
    ("Certificate of attendance", "Unknown"),
])
def test_degree_levels(line, level):
    assert extract_education(line)[0] == level


def test_the_highest_degree_wins_regardless_of_order():
    text = "B.Tech in IT, 2013\nM.Tech in Computer Science, 2015"
    assert extract_education(text)[0] == "Master"
    assert extract_education(text)[2] == 3


# --- names ----------------------------------------------------------------- #
@pytest.mark.parametrize("header, expected", [
    ("Name: Sara Kim\nsara@example.com", "Sara Kim"),
    ("# Priya Sharma\npriya@example.com", "Priya Sharma"),
    ("DAVID KIM\nMachine Learning Engineer", "David Kim"),
    ("Tomás Herrera\ntomas@example.com", "Tomás Herrera"),
])
def test_name_extraction(header, expected):
    assert extract_name(header, Path("x.txt")) == expected


def test_a_mixed_case_name_is_not_re_cased():
    assert extract_name("Fiona McDonald\nf@example.com", Path("x.txt")) == "Fiona McDonald"


def test_the_filename_is_the_last_resort():
    assert extract_name("contact@example.com | +1 555 0100", Path("jane_roe.txt")) == "Jane Roe"


# --- defects found by an examiner's own inputs ----------------------------- #
@pytest.mark.parametrize("text", ["go-live planning", "the go-ahead was given", "a good goal"])
def test_a_bare_lowercase_go_is_not_the_language(text):
    """'go-live' matched Go because the canonical name was in the general pattern."""
    assert "Go" not in extract_skills(text)


@pytest.mark.parametrize("text", ["R&D budget", "see section r below"])
def test_a_bare_lowercase_r_is_not_the_language(text):
    assert "R" not in extract_skills(text)


def test_go_and_r_are_still_found_in_a_skills_list():
    assert "Go" in extract_skills("Languages: Go, Python, Bash")
    assert "R" in extract_skills("Python, R, SQL")


@pytest.mark.parametrize("line, level", [
    ("Certified Scrum Master", "Unknown"),      # not a degree
    ("Scrum Master certification", "Unknown"),
    ("Master's degree in Physics", "Master"),
    ("Master of Science, Delft", "Master"),
    ("BA in Economics", "Bachelor"),
    ("MA in Linguistics", "Master"),
])
def test_a_job_title_containing_master_is_not_a_degree(line, level):
    assert extract_education(line)[0] == level


def test_a_skill_mentioned_once_and_never_declared_earns_no_years():
    """A product manager who "partnered with the Java backend team" was credited with
    5.8 years of Java and passed a "5+ years Java" filter."""
    text = """Dana Pike
dana@example.com

SUMMARY
Product manager with 6 years of experience shipping B2B products.

SKILLS
Product strategy, Roadmapping, SQL, Figma

EXPERIENCE
Senior Product Manager, Granite
Jan 2019 - Dec 2024
- Partnered with the Java backend team to redesign permissions.
"""
    record = resume_rag.process_resume(Path("dana.txt"), text, today=TODAY)
    assert "Java" in record.skills, "the mention is real and should still be visible"
    assert record.skill_years.get("Java", 0.0) == 0.0, "but it must not become years of experience"
    assert record.skill_years.get("SQL", 0.0) > 0 or "SQL" in record.skills


def test_a_declared_skill_still_earns_its_role_duration():
    text = """Ola Berg
ola@example.com

SKILLS
Java, Spring Boot, Kafka

EXPERIENCE
Backend Engineer, Tessera
Jan 2019 - Dec 2024
- Java and Spring Boot services.
"""
    record = resume_rag.process_resume(Path("ola.txt"), text, today=TODAY)
    assert record.skill_years["Java"] == pytest.approx(6.0)


def test_a_skill_used_across_two_roles_is_corroborated_without_being_declared():
    text = """Sam Reyes
sam@example.com

SKILLS
Product strategy

EXPERIENCE
Engineer, One
Jan 2019 - Dec 2020
- Built services in Rust.

Engineer, Two
Jan 2021 - Dec 2022
- More Rust work.
"""
    record = resume_rag.process_resume(Path("sam.txt"), text, today=TODAY)
    assert record.skill_years.get("Rust", 0.0) == pytest.approx(4.0)
