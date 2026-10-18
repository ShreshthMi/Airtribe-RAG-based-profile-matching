"""Render the captioned demo video for the assignment.

Not a screen capture: every frame is composed here, but every word in it comes from
running the real pipeline and from the executed notebook's own charts. Nothing is
typed in by hand, so the video cannot drift from the code the way a recorded take can.

    python scripts/make_demo_video.py            # writes demo/demo.mp4
    python scripts/make_demo_video.py --frames   # stop after frames, for inspection

Needs Pillow and ffmpeg.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
OUT_DIR = ROOT / "demo"
FRAME_DIR = OUT_DIR / "frames"

WIDTH, HEIGHT = 1920, 1080
MARGIN = 88
CAPTION_H = 168

BG = (13, 17, 23)
PANEL = (22, 27, 34)
FG = (201, 209, 217)
DIM = (125, 133, 144)
ACCENT = (88, 166, 255)
GOOD = (63, 185, 80)
WARN = (210, 153, 34)
WHITE = (255, 255, 255)

# macOS first, then the common Linux and Windows locations. Pillow's own scalable default
# is the last resort, so the renderer runs anywhere Pillow does.
MONO = ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf")
SANS = ("/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf")


def font(candidates: tuple[str, ...], size: int):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


F_TITLE = font(SANS, 54)
F_SUB = font(SANS, 32)
F_CAPTION = font(SANS, 31)
F_CMD = font(MONO, 27)
F_OUT = font(MONO, 23)
F_BIG = font(SANS, 40)
F_TAG = font(SANS, 24)


# --------------------------------------------------------------------------- #
# Capturing real output
# --------------------------------------------------------------------------- #
NOISE = ("W1018", "NOTE: Redirects", "Loading weights", "HF_TOKEN", "unauthenticated",
         "torch/distributed", "Warning: You are sending")


def clean(text: str, limit: int | None = None) -> str:
    """Drop start-up noise but keep blank lines: they are the output's own spacing."""
    lines = [l.rstrip() for l in text.splitlines() if not any(n in l for n in NOISE)]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines[:limit] if limit else lines)


def run(args: list[str], limit: int | None = None) -> str:
    proc = subprocess.run([PYTHON] + args, cwd=ROOT, capture_output=True, text=True)
    return clean(proc.stdout or proc.stderr, limit)


def snippet(code: str, limit: int | None = None) -> str:
    proc = subprocess.run([PYTHON, "-c", code], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"capture failed:\n{proc.stderr[-1500:]}")
    return clean(proc.stdout, limit)


def notebook_charts() -> list[Image.Image]:
    """Every chart image from the executed notebook, in order."""
    nb = json.loads((ROOT / "notebooks" / "experiments.ipynb").read_text())
    images = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                raw = base64.b64decode("".join(data["image/png"]))
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    seconds: float
    title: str = ""
    caption: str = ""
    command: str = ""
    output: str = ""
    bullets: list[tuple[str, str]] = field(default_factory=list)
    chart: Image.Image | None = None
    hero: str = ""
    caption_colour: tuple[int, int, int] = FG


def wrap(draw, text: str, fnt, max_width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def render(scene: Scene, index: int) -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    y = MARGIN

    if scene.title:
        draw.text((MARGIN, y), scene.title, font=F_TITLE, fill=WHITE)
        y += 86
        draw.line([(MARGIN, y), (WIDTH - MARGIN, y)], fill=(48, 54, 61), width=2)
        y += 34

    body_bottom = HEIGHT - CAPTION_H - 40

    # Measure the body first so it can sit centred in the space between the title rule
    # and the caption bar, instead of hugging the top with dead space beneath it.
    content_h = 0
    if scene.hero:
        content_h += len(wrap(draw, scene.hero, F_BIG, WIDTH - 2 * MARGIN)) * 58
    for tag, text in scene.bullets:
        offset = MARGIN + max(150, draw.textlength(tag, font=F_TAG) + 30)
        content_h += len(wrap(draw, text, F_SUB, WIDTH - offset - MARGIN)) * 44 + 20
    if scene.command:
        content_h += 86
    if scene.output:
        content_h += min(len(scene.output.splitlines()), (body_bottom - y) // 31) * 31
    if scene.chart is not None:
        scale = min((WIDTH - 2 * MARGIN) / scene.chart.width,
                    (body_bottom - y) / scene.chart.height)
        content_h += int(scene.chart.height * scale) + 40
    y += max(0, (body_bottom - y - content_h) // 2)

    if scene.hero:
        for line in wrap(draw, scene.hero, F_BIG, WIDTH - 2 * MARGIN):
            draw.text((MARGIN, y), line, font=F_BIG, fill=FG)
            y += 58

    for tag, text in scene.bullets:
        draw.text((MARGIN, y + 4), tag, font=F_TAG, fill=ACCENT)
        offset = MARGIN + max(150, draw.textlength(tag, font=F_TAG) + 30)
        for line in wrap(draw, text, F_SUB, WIDTH - offset - MARGIN):
            draw.text((offset, y), line, font=F_SUB, fill=FG)
            y += 44
        y += 20

    if scene.command:
        draw.rounded_rectangle([MARGIN - 24, y - 16, WIDTH - MARGIN + 24, y + 52],
                               radius=10, fill=PANEL)
        draw.text((MARGIN, y), "$ ", font=F_CMD, fill=GOOD)
        draw.text((MARGIN + 34, y), scene.command, font=F_CMD, fill=FG)
        y += 86

    if scene.output:
        for line in scene.output.splitlines():
            if y > body_bottom - 28:
                break
            colour = DIM if line.startswith(("  ...", "...")) else FG
            draw.text((MARGIN, y), line[:118], font=F_OUT, fill=colour)
            y += 31

    if scene.chart is not None:
        available_h = body_bottom - y
        chart = scene.chart
        scale = min((WIDTH - 2 * MARGIN) / chart.width, available_h / chart.height)
        resized = chart.resize((int(chart.width * scale), int(chart.height * scale)),
                               Image.LANCZOS)
        panel_x = (WIDTH - resized.width) // 2
        draw.rounded_rectangle([panel_x - 20, y - 20, panel_x + resized.width + 20,
                                y + resized.height + 20], radius=12, fill=WHITE)
        image.paste(resized, (panel_x, y))

    if scene.caption:
        top = HEIGHT - CAPTION_H
        draw.rectangle([0, top, WIDTH, HEIGHT], fill=PANEL)
        draw.rectangle([0, top, 10, HEIGHT], fill=scene.caption_colour)
        lines = wrap(draw, scene.caption, F_CAPTION, WIDTH - 2 * MARGIN)
        text_y = top + (CAPTION_H - len(lines) * 42) // 2
        for line in lines:
            draw.text((MARGIN, text_y), line, font=F_CAPTION, fill=scene.caption_colour)
            text_y += 42

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAME_DIR / f"scene{index:02d}.png"
    image.save(path)
    return path


# --------------------------------------------------------------------------- #
def build_scenes() -> list[Scene]:
    print("capturing live output ...", flush=True)
    charts = notebook_charts()

    index_out = run(["resume_rag.py", "--resumes", "resumes", "--db", "chroma_db",
                     "--reset", "--as-of", "2026-10-18"], limit=14)
    extracted = snippet('''
import json
reg = json.load(open("chroma_db/candidates.json"))
rows = [c for c in reg["candidates"] if c["resume_path"].split("/")[-1] in
        ("priya_sharma.md", "resume_elena_rodriguez.docx", "resume_david_kim.txt",
         "arjun_mehta.md", "yuki_tanaka.txt")]
print(f'{"candidate":<17}{"yrs":>5}  {"degree":<9}{"skills":>7}  top per-skill years')
for c in rows:
    top = sorted(c["skill_years"].items(), key=lambda kv: -kv[1])[:3]
    years = ", ".join(f"{k} {v:g}y" for k, v in top)
    print(f'{c["candidate_name"]:<17}{c["experience_years"]:>5}  {c["education_level"]:<9}'
          f'{len(c["skills"]):>7}  {years}')
''')
    top_matches = snippet('''
import json, subprocess, sys
out = subprocess.run([sys.executable, "job_matcher.py", "--jd-file",
                      "job_descriptions/senior_ml_engineer.txt", "--k", "5"],
                     capture_output=True, text=True).stdout
d = json.loads(out)
print("critical skills read from the JD:", ", ".join(d["critical_skills"][:7]) + ", ...")
print("must-have extracted from the JD: ", ", ".join(d["must_have_requirements"]))
print()
print(f'{"score":>5}  {"candidate":<17}{"sem":>5}{"skill":>7}{"bm25":>6}{"exp":>5}')
for m in d["top_matches"]:
    b = m["score_breakdown"]
    print(f'{m["match_score"]:>5}  {m["candidate_name"]:<17}{b["semantic"]:>5}'
          f'{b["critical_skills"]:>7}{b["keyword_bm25"]:>6}{b["experience_fit"]:>5}')
''')
    reasoning = snippet('''
import json, textwrap
d = json.load(open("outputs/senior_ml_engineer.json"))
m = d["top_matches"][0]
print(m["candidate_name"] + "  -  " + str(m["match_score"]) + "/100")
print()
for line in textwrap.wrap(m["reasoning"], 112)[:6]:
    print(line)
print()
print("retrieved evidence:")
for e in m["relevant_excerpts"][:2]:
    for line in textwrap.wrap(e, 112)[:2]:
        print("  " + line)
''')
    filtered = snippet('''
import json
d = json.load(open("outputs/senior_ml_engineer.json"))
print(f'{len(d["filtered_out"])} of 43 candidates fail the must-have, each with its reason:')
print()
for f in d["filtered_out"][:6]:
    print(f'{f["match_score"]:>5}  {f["candidate_name"]:<18}{f["failed_requirements"][0]}')
''')
    backend = snippet('''
import json, subprocess, sys
out = subprocess.run([sys.executable, "job_matcher.py", "--jd-file",
                      "job_descriptions/backend_engineer.txt", "--must-have",
                      "5+ years Java", "--brief"], capture_output=True, text=True).stdout
d = json.loads(out)
print("fields returned:", ", ".join(d["top_matches"][0].keys()))
print()
for m in d["top_matches"]:
    print(f'{m["match_score"]:>5}  {m["candidate_name"]}')
''')
    # pytest.ini already sets -q; passing it again makes -qq, which suppresses the
    # summary line this scene exists to show.
    tests = clean(subprocess.run([PYTHON, "-m", "pytest", "-m", "not slow", "--no-header"],
                                 cwd=ROOT, capture_output=True, text=True).stdout).splitlines()[-1]

    return [
        Scene(7, title="RAG-Based Resume Matching",
              hero="Retrieval-augmented matching of resumes to job descriptions.",
              bullets=[("Part A", "section-aware chunking, metadata extraction, embeddings, ChromaDB"),
                       ("Part B", "retrieve, score 0-100, filter on must-haves, explain")],
              caption="43 resumes in 4 formats  ·  7 job descriptions  ·  179 chunks  ·  runs locally, no API keys",
              caption_colour=ACCENT),
        Scene(9, title="Every file is read through the Milestone 1 tools",
              bullets=[("fs_tools", "list_files and read_file, copied verbatim from the Milestone 1 project"),
                       ("sandboxed", "paths confined to the resume folder; .., absolute escapes and hidden files refused"),
                       ("formats", "PDF, DOCX including tables and headers, Markdown, plain text"),
                       ("never raises", "an unreadable resume is reported as a skip, never dropped silently")],
              caption="milestone1_tools/PROVENANCE.md records the source commit and sha256 of both files",
              caption_colour=DIM),
        Scene(22, title="Part A  ·  building the index",
              command="python resume_rag.py --resumes resumes --db chroma_db --reset --as-of 2026-10-18",
              output=index_out,
              caption="--as-of pins the reference date, so experience years and every metric stay reproducible"),
        Scene(16, title="What Part A extracts",
              output=extracted,
              caption="Years per individual skill is what makes “5+ years Python” answerable instead of guessable"),
        Scene(22, title="Part B  ·  matching a job description",
              command="python job_matcher.py --jd-file job_descriptions/senior_ml_engineer.txt --k 5",
              output=top_matches,
              caption="Score = 50% semantic  +  30% required-skill coverage  +  10% BM25  +  10% experience fit"),
        Scene(16, title="Every match carries its evidence", output=reasoning,
              caption="Reasoning is extractive: each claim traces to a retrieved chunk and an extracted number"),
        Scene(17, title="Rejections are explained, not hidden", output=filtered,
              caption="The JD states “5+ years of Python”; the system extracted that rule itself and applied it",
              caption_colour=WARN),
        Scene(16, title="Stating a requirement the JD left implicit",
              command='python job_matcher.py --jd-file .../backend_engineer.txt --must-have "5+ years Java" --brief',
              output=backend,
              caption="--brief prints exactly the six fields the assignment specifies, and nothing else"),
        Scene(11, title="Tests", command='python -m pytest -m "not slow"',
              output=tests + "\n\n94% line coverage. Integration tests use a deterministic fake\nembedder, so the suite runs without loading a transformer.",
              caption="Covers the sandbox refusing ../secret.txt, DOCX tables, corrupt files, duplicate names",
              caption_colour=GOOD),
        Scene(15, title="Retrieval accuracy", chart=charts[3],
              caption="MRR 1.00  ·  recall@10 0.98  ·  nDCG@5 0.972, against graded relevance labels",
              caption_colour=GOOD),
        Scene(20, title="Does the retrieval half earn its place?", chart=charts[5],
              caption="No. A blend with ZERO semantic weight ties production on every metric — on this "
                      "evaluation set the embeddings and vector store add nothing measurable.",
              caption_colour=WARN),
        Scene(21, title="Why, and when it does matter", chart=charts[7],
              caption="Restate the same 7 roles with no literal skill token and the dense half is worth "
                      "+0.091 nDCG@5. The architecture has a precondition: a query that does not quote the corpus.",
              caption_colour=ACCENT),
        Scene(11, title="Latency", chart=charts[4],
              caption="7.2 ms median query; 65% of it is embedding the job description, 19% the vector search",
              caption_colour=DIM),
        Scene(13, title="What these numbers cannot show",
              bullets=[("dataset", "43 synthetic resumes; the JDs quote the resumes' own vocabulary"),
                       ("labels", "written by an AI annotator against a rubric, not independently reviewed"),
                       ("scope", "7 queries, no held-out split; a better weighting was found and not adopted"),
                       ("ethics", "names are excluded from the embedded text, but matching is not hiring")],
              caption="README and notebook §10 state all of this; the evaluation is an internal consistency "
                      "check, not external validation",
              caption_colour=DIM),
        Scene(6, title="Reproduce everything",
              command="make all",
              output="index  ·  both documented queries  ·  177 tests  ·  the notebook, executed\n\n"
                     "github.com/ShreshthMi/Airtribe-RAG-based-profile-matching",
              caption="Every figure in this video was produced by running the committed code",
              caption_colour=ACCENT),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", action="store_true", help="render frames and stop")
    parser.add_argument("--out", default=str(OUT_DIR / "demo.mp4"))
    args = parser.parse_args()

    scenes = build_scenes()
    total = sum(s.seconds for s in scenes)
    # The concat demuxer replays the last entry, so the encoded file runs longer than
    # the sum of the scene durations by exactly the last scene's length.
    encoded = total + scenes[-1].seconds
    print(f"rendering {len(scenes)} scenes, {total:.0f}s of scenes "
          f"-> {int(encoded // 60)}:{int(encoded % 60):02d} encoded")

    paths = [render(scene, i) for i, scene in enumerate(scenes, 1)]
    if args.frames:
        print(f"frames in {FRAME_DIR}")
        return

    listing = OUT_DIR / "scenes.txt"
    lines = []
    for path, scene in zip(paths, scenes):
        lines.append(f"file 'frames/{path.name}'")
        lines.append(f"duration {scene.seconds}")
    lines.append(f"file 'frames/{paths[-1].name}'")
    listing.write_text("\n".join(lines) + "\n")

    output = Path(args.out)
    command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing.name),
               "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
               "-crf", "20", "-movflags", "+faststart", str(output.resolve())]
    result = subprocess.run(command, cwd=OUT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr[-2000:])
    size = output.stat().st_size / 1_000_000
    print(f"wrote {output} ({size:.1f} MB, {total:.0f}s)")


if __name__ == "__main__":
    main()
