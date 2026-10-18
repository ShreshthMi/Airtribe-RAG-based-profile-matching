"""Must-have parsing and critical-skill detection from job descriptions."""
from __future__ import annotations

import pytest

from job_matcher import critical_skills_from_jd, extract_requirements_from_jd, parse_requirement


@pytest.mark.parametrize("text, skill, years", [
    ("5+ years Python", "Python", 5.0),
    ("Python 5 years", "Python", 5.0),
    ("8 yrs Kubernetes", "Kubernetes", 8.0),
    ("Kubernetes", "Kubernetes", None),
    ("3 years experience", None, 3.0),
])
def test_requirement_forms(text, skill, years):
    requirement = parse_requirement(text)
    assert requirement is not None
    assert requirement.skill == skill
    assert requirement.min_years == years


@pytest.mark.parametrize("text, rank", [("Master's degree", 3), ("PhD required", 4), ("Bachelor degree", 2)])
def test_education_requirements(text, rank):
    assert parse_requirement(text).education_rank == rank


def test_an_uninterpretable_requirement_returns_none_so_the_caller_can_warn():
    assert parse_requirement("fluent in Mandarin") is None
    assert parse_requirement("") is None


def test_requirement_checks_against_a_candidate():
    candidate = {"skills": ["Python", "Docker"], "skill_years": {"Python": 6.0},
                 "experience_years": 7.0, "education_level": "Bachelor", "education_rank": 2}
    assert parse_requirement("5+ years Python").check(candidate)[0] is True
    assert parse_requirement("8+ years Python").check(candidate)[0] is False
    assert parse_requirement("Docker").check(candidate)[0] is True
    assert parse_requirement("Kubernetes").check(candidate)[0] is False
    assert parse_requirement("6 years experience").check(candidate)[0] is True
    assert parse_requirement("Master's degree").check(candidate)[0] is False


def test_a_skill_requirement_uses_years_for_that_skill_not_the_career_total():
    candidate = {"skills": ["Python"], "skill_years": {"Python": 2.0},
                 "experience_years": 12.0, "education_level": "Master", "education_rank": 3}
    passes, note = parse_requirement("5+ years Python").check(candidate)
    assert passes is False
    assert "2 yrs Python" in note


def test_years_requirements_are_read_out_of_the_job_description():
    jd = """Senior ML Engineer

Requirements
- 5+ years of Python experience building production systems
- 2 years with Kubernetes
"""
    found = {(r.skill, r.min_years) for r in extract_requirements_from_jd(jd)}
    assert ("Python", 5.0) in found
    assert ("Kubernetes", 2.0) in found


def test_a_bare_years_requirement_becomes_a_total_experience_requirement():
    requirements = extract_requirements_from_jd("We need 6+ years of experience in backend development.")
    assert [(r.skill, r.min_years) for r in requirements] == [(None, 6.0)]


def test_critical_skills_come_from_the_requirements_section_only():
    jd = """Senior ML Engineer

Requirements
- Python and PyTorch

Nice to have
- Spark and Terraform
"""
    critical = critical_skills_from_jd(jd)
    assert "Python" in critical and "PyTorch" in critical
    assert "Spark" not in critical and "Terraform" not in critical


def test_sentences_phrased_as_mandatory_also_count():
    assert "Kafka" in critical_skills_from_jd("Kafka experience is required for this role.")


def test_a_job_description_with_no_headings_falls_back_to_every_skill_mentioned():
    jd = "We want someone who has built RAG systems in Python with LangChain."
    critical = critical_skills_from_jd(jd)
    assert {"Python", "RAG", "LangChain"} <= set(critical)


# --- defects found by an examiner's own job descriptions ------------------- #
def test_a_years_figure_does_not_bind_to_a_skill_on_the_next_line():
    """This silently became a hard "5+ years Python" filter the JD never stated."""
    jd = """Senior Engineer

Requirements
- 5+ years of experience
- Python and SQL
"""
    found = [(r.skill, r.min_years) for r in extract_requirements_from_jd(jd)]
    assert found == [(None, 5.0)]


def test_a_years_figure_still_binds_to_a_skill_on_its_own_line():
    jd = "Requirements\n- 5+ years of Python experience\n"
    assert [(r.skill, r.min_years) for r in extract_requirements_from_jd(jd)] == [("Python", 5.0)]


@pytest.mark.parametrize("phrase", ["5-7 years", "5 – 7 years", "5 to 7 years"])
def test_a_year_range_states_its_lower_bound(phrase):
    jd = f"Requirements\n- {phrase} of backend experience\n"
    assert [r.min_years for r in extract_requirements_from_jd(jd)] == [5.0]
    assert parse_requirement(f"{phrase} Java").min_years == 5.0


def test_a_degree_requirement_naming_two_levels_takes_the_lower_as_its_bar():
    assert parse_requirement("Bachelor's or Master's degree").education_rank == 2
    assert parse_requirement("Master's degree").education_rank == 3


def test_a_scrum_master_credential_is_not_a_degree_requirement():
    requirement = parse_requirement("Certified Scrum Master")
    assert requirement is None or requirement.education_rank is None
