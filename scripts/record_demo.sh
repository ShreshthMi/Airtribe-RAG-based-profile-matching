#!/usr/bin/env bash
# Record a real screen capture of the demo, then append the notebook's charts.
#
#   ./scripts/record_demo.sh
#
# Produces demo/demo_screencast.mp4: a literal screen recording of ./demo_drive.sh auto
# running in this terminal, followed by the chart cards rendered from the executed
# notebook. Roughly 3 to 4 minutes.
#
# PRIVACY: screencapture records the WHOLE DISPLAY. Everything visible is in the file.
# Close anything you would not submit, and turn on Do Not Disturb so notifications do
# not appear mid-take. The countdown below gives you time to do both.
#
# Requires macOS Screen Recording permission for this terminal application:
#   System Settings -> Privacy & Security -> Screen & System Audio Recording
#   enable your terminal app, then QUIT AND REOPEN it (the grant is read at launch).

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT_DIR=demo
RAW="$OUT_DIR/_raw_capture.mov"
TERMINAL_SEG="$OUT_DIR/_terminal.mp4"
CHART_SEG="$OUT_DIR/_charts.mp4"
FINAL="$OUT_DIR/demo_screencast.mp4"
FFMPEG=${FFMPEG:-/opt/homebrew/bin/ffmpeg}

bold=$(tput bold 2>/dev/null || true); reset=$(tput sgr0 2>/dev/null || true)
mkdir -p "$OUT_DIR"

# --- 1. is the permission actually granted? -------------------------------- #
probe="$OUT_DIR/_probe.mov"
rm -f "$probe"
/usr/sbin/screencapture -v -V 1 "$probe" >/dev/null 2>&1
if [ ! -s "$probe" ]; then
  rm -f "$probe"
  cat <<'MSG'
Screen Recording permission is not granted to this terminal.

screencapture exited without producing a file, which is how macOS reports a denied
TCC grant - there is no error message by design.

To grant it:
  1. System Settings -> Privacy & Security -> Screen & System Audio Recording
  2. Enable your terminal application (Terminal, iTerm, or the editor running this)
  3. QUIT the terminal app completely and reopen it. The grant is read at launch,
     so a reload is not enough.
  4. Re-run this script.

Nothing was recorded.
MSG
  exit 2
fi
rm -f "$probe"
echo "Screen Recording permission: granted"

# --- 2. warm up so the take does not open on a model load ------------------ #
echo "Warming the model and building the index (not recorded) ..."
./demo_drive.sh prep >/dev/null 2>&1 || { echo "prep failed"; exit 1; }

# --- 3. countdown ---------------------------------------------------------- #
clear
cat <<MSG
${bold}About to record the whole display.${reset}

  - Close anything you would not submit.
  - Turn on Do Not Disturb.
  - Make this window large; the demo renders into it.

Recording starts in
MSG
for n in 10 9 8 7 6 5 4 3 2 1; do printf '\r  %2d ' "$n"; sleep 1; done
clear

# --- 4. record ------------------------------------------------------------- #
rm -f "$RAW"
/usr/sbin/screencapture -v "$RAW" >/dev/null 2>&1 &
CAPTURE_PID=$!
sleep 2
AUTO=1 ./demo_drive.sh auto
sleep 2
kill -INT "$CAPTURE_PID" 2>/dev/null
wait "$CAPTURE_PID" 2>/dev/null
sleep 1
[ -s "$RAW" ] || { echo "capture produced no file"; exit 1; }
echo "captured $(du -h "$RAW" | cut -f1)"

# --- 5. normalise the capture, build the chart tail, join ------------------ #
echo "Encoding ..."
"$FFMPEG" -y -i "$RAW" \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p" \
  -c:v libx264 -preset medium -crf 20 -an "$TERMINAL_SEG" >/dev/null 2>&1 || exit 1

# The chart cards come from the executed notebook via the renderer, so they carry the
# real figures rather than screenshots of a browser.
.venv/bin/python scripts/make_demo_video.py --frames >/dev/null 2>&1 || exit 1
cat > "$OUT_DIR/_charts.txt" <<LIST
file 'frames/scene10.png'
duration 15
file 'frames/scene11.png'
duration 20
file 'frames/scene12.png'
duration 21
file 'frames/scene13.png'
duration 11
file 'frames/scene14.png'
duration 13
file 'frames/scene14.png'
LIST
( cd "$OUT_DIR" && "$FFMPEG" -y -f concat -safe 0 -i _charts.txt \
    -vf "fps=30,format=yuv420p" -c:v libx264 -preset medium -crf 20 \
    "$(basename "$CHART_SEG")" >/dev/null 2>&1 ) || exit 1

printf "file '%s'\nfile '%s'\n" "$(basename "$TERMINAL_SEG")" "$(basename "$CHART_SEG")" \
  > "$OUT_DIR/_join.txt"
( cd "$OUT_DIR" && "$FFMPEG" -y -f concat -safe 0 -i _join.txt -c copy \
    -movflags +faststart "$(basename "$FINAL")" >/dev/null 2>&1 ) || exit 1

rm -f "$RAW" "$TERMINAL_SEG" "$CHART_SEG" "$OUT_DIR/_charts.txt" "$OUT_DIR/_join.txt"
DURATION=$(/opt/homebrew/bin/ffprobe -v error -show_entries format=duration -of csv=p=0 "$FINAL" 2>/dev/null)
printf '\n%swrote %s%s  (%.0f seconds, %s)\n' "$bold" "$FINAL" "$reset" "${DURATION:-0}" "$(du -h "$FINAL" | cut -f1)"
