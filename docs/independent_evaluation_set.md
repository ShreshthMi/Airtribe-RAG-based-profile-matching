# Protocol: building an independent evaluation set

This is the fix for the most serious limitation of this project's evaluation, recorded as
a procedure so it can actually be carried out rather than just regretted. It is written to
be followed by anyone - a person or an agent - who has **not** read `resumes/`.

## The problem it solves

The seven job descriptions in `job_descriptions/` were written by the same process that
wrote the resumes. They therefore name required skills in the words the resumes use -
"PyTorch", "Kafka", "React". The consequences are measured, not suspected:

- A scoring blend with **zero semantic weight** (`0/.6/.2/.2`) ties the full hybrid on
  every metric: nDCG@5 0.972, P@1 1.00, P@3 1.00, recall@10 0.98, MAP 0.933 (notebook
  §7a). On this set the embeddings and the vector store contribute nothing measurable.
- Whole-document chunking scores within 0.003 nDCG@5 of section-aware chunking (§7b), so
  the evaluation cannot discriminate between those either.
- Restating the same roles with no literal skill token recovers the difference: the dense
  half is worth +0.091 nDCG@5 and +0.143 MAP (§7d). But those paraphrases were written by
  the same author as the system, which bounds how much they prove.

None of that is a finding about retrieval. It is an artefact of one author writing both
sides of the evaluation. Until the queries come from somewhere else, this project cannot
produce evidence for or against its own architecture.

## The one rule

**The job descriptions must be written without reading the resumes, or anything derived
from them.** Everything else here is detail; this is the protocol.

Off limits while writing:

| Path | Why |
|---|---|
| `resumes/` | the corpus itself |
| `job_descriptions/*.txt` | the existing JDs, written from the resumes |
| `job_descriptions/relevance_labels.json` | label reasons quote resume content |
| `evaluation/paraphrased_jds/` | paraphrases of the existing JDs |
| `outputs/*.json` | contain matched skills and resume excerpts |
| `chroma_db/` | contains every resume's full text |
| `notebooks/experiments.ipynb` | candidate tables, excerpts, extracted skills |
| `DATASET.md` below "Coverage" | per-file notes and edge-case descriptions |

`README.md` and the "Coverage" table of `DATASET.md` are safe: they describe the *shape* of
the pool, not its vocabulary, and the writer needs that much or the job descriptions will
have no plausible candidates.

## Phase 1 - write, blind

Write **15** job descriptions, allocated roughly in proportion to the pool's role families
(ML and AI 10, backend 6, data science 5, frontend 5, DevOps/SRE 4, mobile 4, product and
design 4, data engineering 3, application security 1, embedded 1), across a spread of
seniority.

**Source them from real job postings**: read actual postings for each role and write your
own version from what you have read. No verbatim copying, and no real company name,
product, salary, location or contact detail. The point is vocabulary that did not come
from this corpus - real postings describe the same work in the words a recruiter chose,
which is exactly the condition dense retrieval exists for.

Each one:

- plain text, 120-250 words, in whatever structure real postings for that role use.
  **Vary it deliberately** - some with a "Requirements" heading, some with "What you'll
  need", some with none, some as prose. Format variety is itself valuable, because the
  parser has opinions about headings and this is how you find where it breaks.
- a years requirement in at least ten of the fifteen, phrased **differently each time**:
  "5+ years", "at least four years", "5-7 years", "a minimum of three years". Do not
  standardise it; the requirement parser has been wrong about several of these forms.
- the technologies the role genuinely requires. The goal is not to avoid naming
  technologies, only to avoid this corpus's phrasing - which the writer has not seen.
- a degree requirement in two or three ("Bachelor's or Master's in a quantitative field").

Write them to `job_descriptions/independent/<role_slug>.txt`, not reusing the existing
stems. **Then stop and commit them before going further.** Phase 2 changes what you know.

## Phase 2 - label, with the queries frozen

Only now read the resumes. Label all 15 against all 43 candidates on the existing rubric,
so the two sets stay comparable: grade 2 would shortlist, grade 1 plausible, unlisted not
a match. Write `job_descriptions/independent/relevance_labels.json` in the same schema as
the existing file (`description`, `rubric`, `labels`, `hard_negatives`).

- **At least 4 relevant per job description**, or recall is degenerate at k=10 over 43
  candidates. If a role has fewer, record that rather than stretching grades to reach the
  number: too few candidates is a finding about the pool, not a labelling problem.
- Label from the resume and the job description only. **Do not run the matcher** before
  labelling is finished. The existing set's provenance is already compromised on exactly
  this point and the repository now says so.
- **Change nothing in any job description during this phase**, even if labelling reveals
  an awkward one. Editing queries after seeing the corpus is the failure this protocol
  exists to escape.

## Verification to run and record

1. **Vocabulary independence.** Per job description: the share of content words that
   appear anywhere in `resumes/`, and the longest shared word sequence with any single
   resume. Do the same for the existing seven as a baseline. The new set should overlap
   less. **If it does not, the exercise failed and the write-up must say so** - this is
   the measurement that decides whether the protocol worked.
2. **Label sanity.** Every `resume_path` exists; no duplicates within a job description;
   nothing both labelled and listed as a hard negative for the same one; grades only 1 or
   2; at least 4 relevant each, exceptions named.
3. **Provenance.** `job_descriptions/independent/PROVENANCE.md`: who wrote them and how,
   that the resumes were unread until phase 2, what sources were used, and anything at all
   that weakens the independence claim. A disclosed slip is recoverable; an undisclosed
   one poisons the exercise.

## What to do with it

Keep the existing seven. The independent set is a **second** evaluation set, not a
replacement, because the comparison between them is the experiment: if the dense-retrieval
half contributes nothing on same-author queries and something real on independently written
ones, that result needs both sets to exist. `evaluation/harness.load_labels` already takes
a label path, a JD directory and a resume root, so a second set needs no new code - only a
third column in notebook §7d.

Fifteen new queries also makes 22 in total, enough to hold out a split so that weight
tuning stops being measured on the data it is tuned against. That is the other limitation
§10 records, and this protocol is a precondition for fixing it.

## A caveat worth keeping

An AI agent writing these is still *an* author, merely not the same one that wrote the
resumes. That breaks the shared-vocabulary coupling, which is the specific defect measured
above, but it does not make the set externally sourced. The strongest version of this
protocol uses real postings directly, or a person who has never seen the corpus. Any
write-up should claim the weaker thing that is actually true.
