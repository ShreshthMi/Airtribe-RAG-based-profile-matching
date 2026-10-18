# RAG-Based Resume Matching

Two scripts implement the assignment:

| File | Part | What it does |
|---|---|---|
| `file_tools.py` | Milestone 1 tools | Lists and reads resumes (`.txt`, `.md`, `.pdf`, `.docx`). |
| `resume_rag.py` | Part A | Section-aware chunking, metadata extraction, HuggingFace embeddings, ChromaDB storage. |
| `job_matcher.py` | Part B | JD embedding, top-K semantic retrieval, hybrid (BM25 + critical skills) scoring, must-have filters, JSON output. |

Sample data: `resumes/` (12 synthetic resumes) and `job_descriptions/` (2 JDs). Example outputs are in `outputs/`.

## Setup

```bash
pip install -r requirements.txt
```

No API keys needed: embeddings use the HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model, downloaded on first run. If `sentence-transformers` (and therefore PyTorch) isn't installed, `Embedder` falls back to ChromaDB's bundled ONNX export of the same model, so `pip install chromadb rank-bm25` alone is enough to run everything. The sample outputs in `outputs/` were produced with that ONNX fallback.

## Run

```bash
# Part A: build the vector store
python resume_rag.py --resumes resumes --db chroma_db --reset

# Part B: match a job description
python job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt --k 10 --output outputs/senior_ml_engineer.json

# Extra must-haves, and only the fields from the assignment's output format
python job_matcher.py --jd-file job_descriptions/backend_engineer.txt --must-have "5+ years Java" --brief
```

## How it works

### Part A: `resume_rag.py`
1. **Loading.** `file_tools.iter_documents()` reads each resume to plain text.
2. **Section-aware chunking.** Headings such as *Summary, Skills, Experience, Education, Projects* (and common synonyms like "Work History") are detected. Each section becomes its own chunk; long sections such as Experience are split on blank lines (one block per role) and packed up to ~900 chars. A chunk never crosses a section boundary, and each embedded chunk is prefixed with its section label (`"Experience: ..."`).
3. **Metadata extraction.**
   - **Name**: a `Name:` field, else the first short line of the header.
   - **Skills**: matched against a ~75-skill vocabulary with aliases (e.g. `k8s` maps to Kubernetes, `sklearn` to scikit-learn).
   - **Experience years**: date ranges in Experience (`Jan 2019 - Present`, `2016 – 2019`, `03/2020 - 06/2022`) are merged so overlapping roles aren't double-counted. Statements like "8+ years of experience" are used if higher.
   - **Per-skill years**: every skill mentioned in a role inherits that role's duration (frameworks imply their language, e.g. PyTorch counts toward Python). This powers filters like "5+ years Python".
   - **Education**: highest degree level (PhD > Master > Bachelor > Associate) plus the degree line.
4. **Storage.** Chunks, embeddings and flat metadata (`candidate_name`, `resume_path`, `section`, `skills`, `experience_years`, `education_level`, ...) go into a persistent ChromaDB collection (cosine space), so you can filter with `where={"experience_years": {"$gte": 5}}`. A `candidates.json` registry stores full text and structured metadata for keyword search and requirement checks. Re-indexing is idempotent.

### Part B: `job_matcher.py`
1. **Semantic search.** The JD is split into passages (MiniLM truncates long inputs), each is embedded and queried against ChromaDB, and chunk hits are aggregated per candidate (best similarity plus top-3 mean).
2. **Keyword search.** BM25 over full resume text, plus exact matching of **critical skills**: skills in the JD's Requirements/Must-have section or in sentences with "must/required".
3. **Score (0–100)** = 50% semantic + 30% critical-skill coverage + 10% BM25 + 10% experience fit. The breakdown is included in `score_breakdown`.
4. **Must-have filtering.** "N+ years X" statements are read from the JD automatically, and you can add more with `--must-have` (`"5+ years Python"`, `"Kubernetes"`, `"Master's degree"`, `"3+ years experience"`). Candidates who fail are excluded from `top_matches` and listed in `filtered_out` with the reason.
5. **Reasoning.** Each match lists matched and missing critical skills, the sections closest to the JD (with similarity), experience against the requirement, education, and which must-haves were met. `relevant_excerpts` are the top-matching chunks, tagged by section.

### Output
The assignment's fields (`job_description`, `top_matches[].candidate_name/resume_path/match_score/matched_skills/relevant_excerpts/reasoning`) are always present. The default output adds `critical_skills`, `must_have_requirements`, `filtered_out`, `matched_sections`, `score_breakdown`, `experience_years` and `education`. Use `--brief` to print the exact assignment schema only.

## Swapping components
- **Embedding model**: `--model` on `resume_rag.py` (e.g. `BAAI/bge-small-en-v1.5`). The matcher reads the model name from the registry, so both sides stay consistent.
- **OpenAI/Cohere**: replace `Embedder.embed()` in `resume_rag.py`; nothing else changes.
- **Pinecone/Weaviate**: replace `get_collection()` and the `collection.query` call in `JobMatcher.semantic_search()`.
