# Reproduce every deliverable from a clean checkout.
#
#   make setup     create the virtualenv and install everything
#   make all       index, match, test and execute the notebook
#   make index     build the ChromaDB index from resumes/
#   make match     run the documented queries into outputs/
#   make test      run the test suite with coverage
#   make notebook  execute notebooks/experiments.ipynb in place
#
# Python 3.12 is used rather than 3.14: sentence-transformers pulls in PyTorch, and
# 3.14 wheels are not verified here.

PYTHON  := .venv/bin/python
UV      := uv
AS_OF   := 2026-10-18
RESUMES := resumes
DB      := chroma_db

.PHONY: all setup install index match test test-all notebook video screencast clean clean-index

all: index match test notebook

setup:
	$(UV) venv --python 3.12 .venv
	VIRTUAL_ENV=.venv $(UV) pip install -r requirements-dev.txt

install:
	VIRTUAL_ENV=.venv $(UV) pip install -r requirements-dev.txt

# Part A: section-aware chunking, metadata extraction, embeddings, ChromaDB.
# --as-of pins the reference date for open-ended roles so results are reproducible.
index:
	$(PYTHON) resume_rag.py --resumes $(RESUMES) --db $(DB) --reset --as-of $(AS_OF)

# Part B: the documented queries.
match:
	mkdir -p outputs
	$(PYTHON) job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt \
		--k 10 --output outputs/senior_ml_engineer.json > /dev/null
	$(PYTHON) job_matcher.py --jd-file job_descriptions/backend_engineer.txt \
		--must-have "5+ years Java" --k 10 --output outputs/backend_engineer.json > /dev/null
	$(PYTHON) job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt \
		--k 10 --brief --output outputs/senior_ml_engineer_brief.json > /dev/null
	@echo "wrote outputs/*.json"

# Fast suite: skips the three tests that load the real embedding model.
test:
	$(PYTHON) -m pytest -m "not slow" --cov=resume_rag --cov=resume_loader \
		--cov=job_matcher --cov=evaluation --cov-report=term-missing

test-all:
	$(PYTHON) -m pytest

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/experiments.ipynb

# Rebuild the demo video from live output and the executed notebook's charts.
# Run `make notebook` first if the charts are stale. Needs ffmpeg.
video:
	$(PYTHON) scripts/make_demo_video.py

# Literal screen recording of the demo running, with the notebook's charts appended.
# Needs macOS Screen Recording permission for your terminal app; the script checks and
# tells you how to grant it. Records the WHOLE DISPLAY - see the warning in the script.
screencast:
	./scripts/record_demo.sh

# Only ever removes generated artefacts, every path named explicitly.
clean-index:
	rm -rf ./$(DB)

clean: clean-index
	rm -rf ./.pytest_cache ./htmlcov
	rm -f ./.coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
