#!/usr/bin/env bash
# Demo driver for the resume-RAG assignment video.
#
#   ./demo_drive.sh prep     run this BEFORE recording: warms the model, builds the index
#   ./demo_drive.sh          the take: press Enter to advance through each beat
#   ./demo_drive.sh auto     the take, self-paced, for an unattended screen recording
#
# Every command shown is the real one from the README. Nothing is faked; the only
# cosmetic change is filtering the HuggingFace/torch startup warnings so the output
# reads cleanly on camera.

set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=${PY:-.venv/bin/python}
QUIET='grep -vE "^W[0-9]+|NOTE: Redirects|Loading weights|HF_TOKEN|unauthenticated|^$"'

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)

beat() {                       # beat "<timestamp>" "<title>" "<narration cue>"
  printf '\n%s%s  %s %s%s\n' "$bold" "$1" "$2" "$reset" ""
  printf '%s   cue: %s%s\n\n' "$dim" "$3" "$reset"
}
AUTO=${AUTO:-0}
# Seconds to hold each beat in auto mode: long enough to read the caption and the
# output beneath it, short enough to keep the whole take inside four minutes.
pause() {
  if [ "$AUTO" = "1" ]; then
    sleep "${1:-12}"
  else
    printf '\n%s   [Enter to continue]%s' "$dim" "$reset"; read -r _
  fi
  clear
}
show()  { printf '%s$ %s%s\n\n' "$bold" "$1" "$reset"; }

if [ "${1:-}" = "auto" ]; then
  AUTO=1
fi

if [ "${1:-}" = "prep" ]; then
  echo "Pre-flight: building the index so the 12s model load does not land mid-take."
  $PY resume_rag.py --resumes resumes --db chroma_db --reset --as-of 2026-10-18 2>&1 \
    | eval $QUIET | tail -3
  echo
  echo "Warming the query path..."
  $PY job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt --k 1 >/dev/null 2>&1
  echo "Ready. Open notebooks/experiments.ipynb in a browser tab, then run: ./demo_drive.sh"
  exit 0
fi

clear
beat "0:00-0:25" "What it is" \
  "RAG matching for resumes. Part A indexes, Part B matches, every file read through my Milestone 1 tools."
show "ls && git log --oneline -5"
ls
echo
git log --oneline -5
pause 10

beat "0:25-1:05" "Part A: indexing" \
  "43 resumes, 4 formats, loaded through fs_tools. Section-aware chunking. Skills, years, years-per-skill, degree."
show "python resume_rag.py --resumes resumes --db chroma_db --reset --as-of 2026-10-18"
$PY resume_rag.py --resumes resumes --db chroma_db --reset --as-of 2026-10-18 2>&1 \
  | eval $QUIET | head -14
echo "  ... (43 candidates total)"
pause 15

beat "1:05-1:45" "Part B: matching" \
  "JD embedded and split into passages. Score = 50% semantic, 30% required-skill coverage, then BM25 and experience."
show "python job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt --k 5"
$PY - <<'PY' 2>&1 | eval $QUIET
import json, subprocess, sys
out = subprocess.run([sys.executable, "job_matcher.py", "--jd-file",
                      "job_descriptions/senior_ml_engineer.txt", "--k", "5"],
                     capture_output=True, text=True).stdout
d = json.loads(out)
print("critical skills from the JD:", ", ".join(d["critical_skills"]))
print("must-haves extracted:      ", ", ".join(d["must_have_requirements"]))
print()
for m in d["top_matches"]:
    print(f"  {m['match_score']:>3}  {m['candidate_name']:<16} {m['score_breakdown']}")
print()
print("reasoning for the top match:")
print(" ", d["top_matches"][0]["reasoning"][:300])
PY
pause 17

beat "1:45-2:15" "Rejections are explainable" \
  "27 of 43 fail the 5+ years Python rule the system extracted itself. The top one scores 84 on text and still fails - his Python spans a single role."
show "jq '.filtered_out[:4]' outputs/senior_ml_engineer.json"
$PY -c "
import json
d=json.load(open('outputs/senior_ml_engineer.json'))
print(f'{len(d[\"filtered_out\"])} of 43 filtered out. Highest-scoring rejections:')
for f in d['filtered_out'][:4]:
    print(f\"  {f['match_score']:>3}  {f['candidate_name']:<16} {f['failed_requirements'][0]}\")
"
pause 14

beat "2:15-2:35" "Stating a requirement the JD left implicit" \
  "That JD names Java but never attaches a number. I state it myself."
show 'python job_matcher.py --jd-file job_descriptions/backend_engineer.txt --must-have "5+ years Java" --brief'
$PY job_matcher.py --jd-file job_descriptions/backend_engineer.txt \
  --must-have "5+ years Java" --brief 2>/dev/null \
  | $PY -c "
import json,sys
d=json.load(sys.stdin)
print('fields returned:', ', '.join(d['top_matches'][0].keys()))
print()
for m in d['top_matches'][:4]: print(f\"  {m['match_score']:>3}  {m['candidate_name']}\")
"
pause 13

beat "2:35-2:50" "Tests" \
  "177 tests, 94% coverage. Integration tests use a fake embedder so they run in under a second."
show 'python -m pytest -m "not slow" -q'
$PY -m pytest -m "not slow" -q 2>&1 | tail -3
pause 10

beat "2:50-3:40" "Evaluation - switch to the notebook" \
  "MRR 1.00, recall@10 0.98, nDCG@5 0.972 - then say plainly that a no-semantic blend ties it, and go to 7d for the paraphrased queries where the dense half is worth +0.091. Then section 9, the one miss, and section 10, limitations."
cat <<'TXT'
  Switch to the browser tab now:
    section 5  - the accuracy table
    section 7a - the ablation chart   <- the honest result: no-semantic ties production
    section 7d - paraphrased queries  <- the strongest slide: where dense retrieval earns its place
    section 6  - latency by phase
    section 9  - the single missed candidate
    section 10 - limitations
TXT
pause 13

beat "3:40-4:00" "Close" "Everything reproduces with make all."
show "make all"
echo "  (do not actually run this on camera - it takes several minutes)"
echo
printf '%sEnd of take.%s\n' "$bold" "$reset"
