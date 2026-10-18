# Demo video script (3-4 minutes)

A shot list with what to say and what to have on screen. Timings are cumulative.
Everything here is real output from this repository - nothing needs staging.

**Before recording**

```bash
make index                          # so the model is warm and the DB exists
open notebooks/experiments.ipynb    # have the executed notebook ready in a tab
```

Have three things open: a terminal, the notebook, and `outputs/senior_ml_engineer.json`.

---

## 0:00-0:25 - What it does and what it is built from

> "This is a retrieval-augmented matching system for resumes. You give it a job
> description and it ranks candidates out of a vector index of 43 resumes, scoring each
> one and explaining why. Part A builds the index, Part B does the matching, and every
> file is read through the file-system tools I built for Milestone 1."

On screen: the repository tree. Point at `resume_rag.py`, `job_matcher.py`,
`milestone1_tools/`.

## 0:25-1:05 - Part A: indexing

Run it live:

```bash
python resume_rag.py --resumes resumes --db chroma_db --reset --as-of 2026-10-18
```

> "Forty-three resumes in four formats - text, markdown, PDF and Word - loaded through
> Milestone 1's fs_tools, which is sandboxed to the resume folder and detects each file's
> real format from its contents rather than trusting the extension. Each resume is split
> by section - summary, skills, experience, education - and chunked inside a section only,
> so a chunk never mixes education with work history. Then the metadata: skills against a
> canonical vocabulary, total years merged from date ranges so overlapping roles aren't
> double counted, years per individual skill, and highest degree."

Point at the per-candidate table and then the timing line.

> "179 chunks, and the build is dominated by loading the embedding model once - the actual
> work is just over a second - and most of that is opening the database, not the data."

## 1:05-2:05 - Part B: matching

```bash
python job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt --k 5
```

> "The job description is embedded, split into passages so nothing gets truncated, and
> queried against the index. Scoring is hybrid: half semantic similarity, 30% exact
> coverage of the skills the JD lists as required, then BM25 and experience fit."

Show the top result, read part of its `reasoning`, then scroll to `filtered_out`:

> "The JD asks for five-plus years of Python, which the system extracted on its own. Six
> candidates fail it and are moved out with the reason given - including this one, who
> scores 84 on text similarity but whose Python only spans his most recent role. He reads
> like the job and isn't qualified for it, and the system says so rather than hiding him."

Then the sharper query:

```bash
python job_matcher.py --jd-file job_descriptions/backend_engineer.txt \
  --must-have "5+ years Java" --brief
```

> "That JD names Java but never attaches a number to it, so I state the requirement
> myself. Five candidates pass, and the fifth is instructive - she is a product manager
> who the parser credits with 5.8 years of Java because she mentions partnering with a
> Java backend team. She passes the filter and the reasoning asserts the qualification.
> The score buries her at 15, but the filter should not have let her through, and that is
> a real defect, not a feature."

## 2:05-3:05 - Evaluation

Switch to the notebook, section 5.

> "Accuracy is measured against graded relevance labels - written by a separate Claude
> session against a stated rubric, not by a human annotator, which is the main caveat on
> every number here. Across
> seven job descriptions: MRR 1.00 - the top hit is relevant every time - recall@10 0.98,
> and nDCG@5 0.97."

Scroll to section 7a, the ablation chart.

> "The interesting result is this ablation. Semantic similarity alone gets nDCG@5 0.86,
> and puts an irrelevant candidate first for one role in seven. The hybrid gets 0.97. The
> embedding model thinks 'ML platform engineer' and 'ML engineer' are nearly the same
> thing; only an exact check against the required skills separates them. That's what the
> 30% skill weight buys."

Scroll to section 6, latency.

> "A query is about 7 milliseconds, and nearly two thirds of that is embedding the job
> description - the vector search itself is a fifth of it at this corpus size."

## 3:05-3:40 - Honesty and limits

Scroll to section 9, the error analysis, then section 10.

> "One relevant candidate is missed across all seven roles: a computer-vision PhD whose
> similarity to the ML job is high but who lacks the LLM and RAG keywords. That's the same
> skill weight that made the hybrid work, now costing me a good adjacent candidate - the
> precision-recall trade-off made visible."

> "And the caveats, which matter more than the numbers: 43 synthetic resumes written for
> this assignment and labelled by the same project, seven queries, no held-out split. I
> found a weighting that scores marginally better and deliberately did not adopt it,
> because tuning weights on the same labels you report accuracy with is how an evaluation
> stops meaning anything."

## 3:40-4:00 - Close

> "Everything reproduces with `make all`: the index, both queries, 177 fast tests at 94%
> coverage, and the notebook. The labels, the dataset card and the limitations are all in
> the repo."

---

## Make sure these are visible on screen at some point

- `milestone1_tools/PROVENANCE.md` - evidence the Milestone 1 tools are used verbatim.
- A `filtered_out` entry with its reason - rejections are explainable, not silent.
- The ablation chart - the strongest single piece of analysis.
- Section 10, Limitations - the part that separates a demo from an evaluation.
