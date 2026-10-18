"""Record the demo as a terminal session and render it as video.

Not a screen capture: the demo runs under a pseudo-terminal, every chunk of output is
timestamped as it arrives, and the session is then replayed through a terminal emulator
and drawn frame by frame. The result shows output appearing progressively at the speed
it actually ran - a command's results filling in line by line, the index build visibly
taking the twelve seconds it takes - without needing Screen Recording permission or
putting anything else that is on the display into the file.

    python scripts/record_terminal_session.py            # record, render, append charts
    python scripts/record_terminal_session.py --replay   # re-render the last recording

Needs pyte and ffmpeg.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import termios
import time
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "demo"
SESSION = OUT_DIR / "_session.json"

COLS, ROWS = 126, 30
FPS = 10
WIDTH, HEIGHT = 1920, 1080
PAD_X, PAD_Y = 64, 56
LINE_H = 32

BG = (13, 17, 23)
FG = (201, 209, 217)
COLOURS = {
    "default": FG, "white": (255, 255, 255), "green": (63, 185, 80),
    "blue": (88, 166, 255), "cyan": (57, 197, 187), "yellow": (210, 153, 34),
    "red": (248, 81, 73), "magenta": (188, 140, 255), "black": (110, 118, 129),
    "brightblack": (110, 118, 129),
}
MONO = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 23)
MONO_BOLD = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 23, index=1)
# Measured rather than guessed, so columns line up exactly with the emulator's grid.
CELL_W = MONO.getlength("M")


# --------------------------------------------------------------------------- #
def record(command: list[str]) -> list[tuple[float, str]]:
    """Run `command` under a PTY, returning (seconds_since_start, output) chunks."""
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    environment = {**os.environ, "TERM": "xterm-256color",
                   "COLUMNS": str(COLS), "LINES": str(ROWS), "AUTO": "1"}
    process = subprocess.Popen(command, cwd=ROOT, stdin=slave, stdout=slave, stderr=slave,
                               env=environment, close_fds=True)
    os.close(slave)

    chunks: list[tuple[float, str]] = []
    started = time.monotonic()
    while True:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append((time.monotonic() - started, data.decode("utf-8", "replace")))
        elif process.poll() is not None:
            break
    os.close(master)
    process.wait()
    return chunks


def draw(screen: pyte.Screen, elapsed: float, total: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    canvas = ImageDraw.Draw(image)
    for row in range(ROWS):
        line = screen.buffer.get(row, {})
        y = PAD_Y + row * LINE_H
        for col in range(COLS):
            cell = line.get(col)
            if cell is None or cell.data in ("", " "):
                continue
            colour = COLOURS.get(cell.fg, FG if cell.fg == "default" else FG)
            if cell.bold and cell.fg == "default":
                colour = (255, 255, 255)
            canvas.text((PAD_X + col * CELL_W, y), cell.data,
                        font=MONO_BOLD if cell.bold else MONO, fill=colour)
    # a thin progress rule, so the viewer can see how far through the take they are
    if total > 0:
        canvas.rectangle([0, HEIGHT - 6, int(WIDTH * min(elapsed / total, 1.0)), HEIGHT],
                         fill=(88, 166, 255))
    return image


def render(chunks: list[tuple[float, str]], destination: Path) -> float:
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.Stream(screen)
    total = chunks[-1][0] + 1.5 if chunks else 1.0
    frames = int(total * FPS)

    command = ["/opt/homebrew/bin/ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", str(destination)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    index = 0
    for frame in range(frames):
        now = frame / FPS
        while index < len(chunks) and chunks[index][0] <= now:
            stream.feed(chunks[index][1])
            index += 1
        encoder.stdin.write(draw(screen, now, total).tobytes())
        if frame % (FPS * 15) == 0:
            print(f"  {now:5.1f}s / {total:.0f}s", flush=True)
    encoder.stdin.close()
    encoder.wait()
    return total


def append_charts(terminal: Path, final: Path) -> None:
    """Tail the session with the notebook's chart cards, already rendered as frames."""
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/make_demo_video.py"),
                    "--frames"], cwd=ROOT, capture_output=True, check=True)
    listing = OUT_DIR / "_charts.txt"
    plan = [("scene10.png", 15), ("scene11.png", 20), ("scene12.png", 21),
            ("scene13.png", 11), ("scene14.png", 13)]
    lines = []
    for name, seconds in plan:
        lines += [f"file 'frames/{name}'", f"duration {seconds}"]
    lines.append(f"file 'frames/{plan[-1][0]}'")
    listing.write_text("\n".join(lines) + "\n")

    charts = OUT_DIR / "_charts.mp4"
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", listing.name, "-vf", f"fps={FPS},format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", charts.name],
                   cwd=OUT_DIR, capture_output=True, check=True)
    join = OUT_DIR / "_join.txt"
    join.write_text(f"file '{terminal.name}'\nfile '{charts.name}'\n")
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", join.name, "-c", "copy", "-movflags", "+faststart", final.name],
                   cwd=OUT_DIR, capture_output=True, check=True)
    for leftover in (listing, charts, join, terminal):
        leftover.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="re-render the saved session")
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    if args.replay and SESSION.exists():
        chunks = [(t, s) for t, s in json.loads(SESSION.read_text())]
        print(f"replaying {len(chunks)} chunks from {SESSION.name}")
    else:
        print("warming up (not recorded) ...", flush=True)
        subprocess.run(["./demo_drive.sh", "prep"], cwd=ROOT, capture_output=True)
        print("recording the session ...", flush=True)
        chunks = record(["./demo_drive.sh", "auto"])
        SESSION.write_text(json.dumps(chunks))
        print(f"  {len(chunks)} output chunks over {chunks[-1][0]:.0f}s")

    terminal = OUT_DIR / "_terminal.mp4"
    print("rendering frames ...", flush=True)
    seconds = render(chunks, terminal)
    print("appending the notebook charts ...", flush=True)
    final = OUT_DIR / "demo_session.mp4"
    append_charts(terminal, final)
    size = final.stat().st_size / 1_000_000
    print(f"wrote {final} ({size:.1f} MB; {seconds:.0f}s terminal + 80s charts)")


if __name__ == "__main__":
    main()
