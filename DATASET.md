# Dataset card

43 resumes and 7 job descriptions, with hand-made relevance labels for every resume and job description pair. Every person in the dataset is fictional.

| Path | What it holds |
|---|---|
| `resumes/` | 43 resumes in one flat folder: 19 `.txt`, 18 `.md`, 3 `.pdf`, 3 `.docx` |
| `job_descriptions/*.txt` | 7 job descriptions |
| `job_descriptions/relevance_labels.json` | Graded relevance labels and hard negatives for each job description |

## Where the resumes come from

| Source | Count | Files | Notes |
|---|---|---|---|
| Original assignment set | 12 | the `.md`/`.txt` files present in the first commit | Synthetic people; employers are real company names |
| Milestone 1 fixtures | 8 | `resume_*.pdf`, `resume_*.docx`, `resume_*.txt` | Copied unchanged from `LLM-Powered-File-System-Assistant/resumes`, generated there by `scripts/generate_resumes.py`; fictional people and companies, 555-01xx phone numbers |
| Written for this milestone | 23 | all other files | Fictional people and companies |

All email addresses use `example.com`.

## Coverage

| Role family | Resumes |
|---|---|
| ML and AI engineering or research | 10 |
| Backend | 6 |
| Data science and analytics | 5 |
| Frontend and full-stack JavaScript | 5 |
| DevOps, SRE and cloud | 4 |
| Mobile | 4 |
| Product, design and engineering management | 4 |
| Data engineering | 3 |
| Application security | 1 |
| Embedded systems | 1 |

Experience runs from 2 to 14 years as of October 2026. 5 people have under 5 years, 19 have 5 to 8, 12 have 8 to 10, and 7 have 10 or more. Candidates are based across Asia, Europe, Africa, the Middle East, the Americas and Australia. The plain-text resumes are 553 to 1,386 characters long (791 on average).

## Job descriptions

| File | Role | Stated minimum | Relevant (strong) | Hard negatives |
|---|---|---|---|---|
| `senior_ml_engineer.txt` | Senior ML Engineer, GenAI platform | 5+ years Python | 7 (3) | 3 |
| `backend_engineer.txt` | Senior Backend Engineer, payments | 6+ years backend | 4 (2) | 3 |
| `data_engineer.txt` | Data Engineer, streaming analytics | 4+ years data engineering | 4 (3) | 2 |
| `frontend_engineer.txt` | Frontend Engineer, design systems | 3+ years React and TypeScript | 4 (3) | 3 |
| `devops_sre.txt` | Site Reliability Engineer | 5+ years DevOps or SRE | 5 (3) | 2 |
| `data_scientist.txt` | Data Scientist, product analytics | 3+ years as a data scientist or analyst | 5 (3) | 3 |
| `mobile_engineer.txt` | Senior Mobile Engineer, iOS and Android | 4+ years native mobile | 4 (2) | 1 |

12 resumes are relevant to no job description. They act as distractors.

## Relevance labels

`relevance_labels.json` grades each resume for each job description:

- **2, strong match:** meets every stated minimum (years and required stack) and the core of the role.
- **1, partial match:** meets the stated minimum years and much of the required stack, or works in an adjacent role a recruiter would still shortlist.
- **0, not a match:** this includes anyone below a stated minimum. Resumes not listed under a job description are grade 0.

Each label carries a one-line reason. Use grade 1 or higher as "relevant" for precision@k, recall@k and MRR, and the grades themselves for nDCG. `resume_path` matches the `resume_path` field the matcher returns, and the keys match the job description file names.

`hard_negatives` lists the grade-0 resumes most likely to fool a matcher, each with a reason. Examples are strong keyword overlap with too few years, or the right words in the wrong role.

The labels were written from the resume and job description text before any matcher output was seen, by a single annotator (Claude, an AI assistant). They have not been independently reviewed.

## Intentional edge cases

No file is corrupt or malformed. These variations are deliberate:

- **Formats:**
  - `resume_priya_sharma.pdf` runs to two pages.
  - `resume_elena_rodriguez.docx` keeps its contact details in the page header and its skills in a table.
- **Headings:** Markdown headings, ALL-CAPS headings and synonyms such as Profile, Core Competencies, Work History, Academic Background, Professional Summary and Technical Skills.
- **Dates:**
  - Month and year: `Jan 2019 - Present`
  - Year only: `2016 - 2019`, used in the Milestone 1 files
  - Numeric: `03/2022 - Present`, in `diego_fernandez.txt`
- **Names:**
  - Accents: `tomas_herrera.md`, `diego_fernandez.txt`.
  - ALL-CAPS names: two Milestone 1 `.txt` files.
  - Two pairs of different people share a name: `john_doe.txt` and `resume_john_doe.pdf`, and `priya_sharma.md` and `resume_priya_sharma.pdf`. Match on `resume_path`, never on name.
- **Word traps:**
  - "JavaScript" without Java: `resume_john_smith.docx`
  - "Swiftcart", a company name: `priyanka_nair.txt`
  - "Monty Python", a hobby: `resume_sofia_novak.txt`
  - A product manager who mentions a Java team: `resume_aisha_bello.docx`
  - A designer who mentions React engineers, iOS and Android: `yuki_tanaka.txt`
  - Python spelled several ways in the Milestone 1 files ("Django/Python", "python 3", "PYTHON")
- **Career shape:** `jessica_park.txt` switched careers after a short gap. She has 8.7 years of total experience but only 1 as a data scientist.

## Limitations

- The resumes are synthetic, short and in English. Real resumes are longer and noisier.
- Each job description has only 4 to 7 relevant resumes, so a single rank swap moves precision noticeably.
- The labels come from a single annotator, so there is no measure of inter-annotator agreement.
- "Present" is resolved against today's date, so years of experience grow over time. The labels assume October 2026. From December 2027 the years-based hard negatives start passing their minimums, beginning with `aisha_khan.md` reaching 5 years of Python.

## Adding a resume

The Milestone 1 loader lists one folder without recursion and skips hidden files. So:

1. Put the file directly in `resumes/`. Files in subfolders are skipped.
2. Use `.txt`, `.md`, `.pdf` or `.docx`.
3. Don't start the file name with a dot, and remove any `~$` Office lock files.
4. Rebuild the index.
5. If the resume is relevant to a job description, add it to `relevance_labels.json`.
