"""Resume loading through the Milestone 1 file-system tools.

Part A never touches the filesystem directly. Every resume is discovered with
Milestone 1's ``fs_tools.list_files`` and read with ``fs_tools.read_file``, so that
deliverable's sandboxing, content-sniffing format detection and never-raise error
contract carry over to the RAG pipeline unchanged. See
``milestone1_tools/PROVENANCE.md`` for what is vendored and why.

This module is the only adapter between the two projects. It translates three things:

* **Configuration.** ``fs_tools`` confines every path to ``$FS_ASSISTANT_ROOT`` (or the
  working directory when unset) and re-reads that variable on each call, so the root is
  set and restored around each batch instead of trusting where the process was started.
* **Error style.** The tools report failures as data (``{"success": False, "error": ...}``
  for reads, ``[{"error": ...}]`` for listings). A listing failure is a programming or
  configuration error and raises :class:`LoaderError`; a single unreadable resume is
  expected in a real corpus and becomes a :class:`SkippedResume` the caller can report.
* **Scope.** ``fs_tools`` also reads ``.csv``/``.json``; only resume formats are listed.

Note that ``list_files`` is deliberately **not** recursive and skips sub-folders, hidden
names and Office lock files (``~$...``), so the resume directory must be flat.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# The vendored copies live here; point MILESTONE1_TOOLS_DIR at a live checkout to use
# that instead (see milestone1_tools/PROVENANCE.md).
TOOLS_DIR_ENV_VAR = "MILESTONE1_TOOLS_DIR"
VENDORED_TOOLS_DIR = Path(__file__).resolve().parent / "milestone1_tools"

# sha256 of the verbatim Milestone 1 sources, checked by verify_vendored_tools().
EXPECTED_HASHES = {
    "fs_tools.py": "9d7ea00f75843ec863dfa722d7d0442a1a5651b39890c93acc62a803b8464e38",
    "extractors.py": "239a72f23c5ba5b8d731a0d89773c85ef58de49e973a1278c06277960ef84cda",
}

RESUME_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def tools_dir() -> Path:
    """Directory the Milestone 1 tools are imported from."""
    override = (os.environ.get(TOOLS_DIR_ENV_VAR) or "").strip()
    return Path(override).expanduser().resolve() if override else VENDORED_TOOLS_DIR


def _import_fs_tools():
    """Import Milestone 1's fs_tools from `tools_dir()`.

    The tools are written as top-level modules (``fs_tools`` does a flat
    ``import extractors``), so their directory goes on ``sys.path`` rather than being
    turned into a package - repackaging would mean editing Milestone 1 code.
    """
    directory = str(tools_dir())
    if not (Path(directory) / "fs_tools.py").is_file():
        raise LoaderError(
            f"Milestone 1 tools not found in {directory}. Expected fs_tools.py and "
            f"extractors.py there, or set {TOOLS_DIR_ENV_VAR} to a checkout that has them."
        )
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        import fs_tools  # noqa: PLC0415  (deliberately late: sys.path must be set first)
    except ImportError as exc:  # missing pypdf / python-docx, or a broken checkout
        raise LoaderError(f"Could not import Milestone 1 fs_tools from {directory}: {exc}") from exc
    return fs_tools


class LoaderError(Exception):
    """The resume directory could not be listed, or the Milestone 1 tools are unusable."""


@dataclass(frozen=True)
class LoadedResume:
    """One resume read through Milestone 1's ``read_file``."""

    path: Path          # absolute path on disk
    display_path: str   # "resumes/priya_sharma.md" - what lands in the index as resume_path
    text: str
    metadata: dict      # fs_tools metadata: name, extension, size_bytes, char_count, page_count, ...
    warning: str | None  # fs_tools warning, e.g. a PDF with no text layer on some pages


@dataclass(frozen=True)
class SkippedResume:
    """A file that was listed but could not be turned into usable text."""

    display_path: str
    reason: str


@contextlib.contextmanager
def sandbox_root(root: Path | str):
    """Confine the Milestone 1 tools to `root` for the duration of the block.

    ``fs_tools`` reads ``$FS_ASSISTANT_ROOT`` on every call, so this is the supported way
    to retarget it. The previous value is restored on exit. Not thread-safe, by nature of
    the environment being process-global.
    """
    previous = os.environ.get("FS_ASSISTANT_ROOT")
    os.environ["FS_ASSISTANT_ROOT"] = str(Path(root).expanduser().resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FS_ASSISTANT_ROOT", None)
        else:
            os.environ["FS_ASSISTANT_ROOT"] = previous


def verify_vendored_tools() -> list[str]:
    """Return a list of problems with the vendored tools; empty means they are verbatim.

    Guards the "unmodified copy" claim in PROVENANCE.md. Skipped when
    MILESTONE1_TOOLS_DIR points somewhere else, since a live checkout may legitimately
    have moved on.
    """
    directory = tools_dir()
    if directory != VENDORED_TOOLS_DIR:
        return []
    problems = []
    for filename, expected in EXPECTED_HASHES.items():
        target = directory / filename
        if not target.is_file():
            problems.append(f"{filename} is missing from {directory}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(
                f"{filename} does not match Milestone 1 commit fc7cf0c "
                f"(sha256 {actual[:12]}..., expected {expected[:12]}...). "
                f"Update the table in {directory.name}/PROVENANCE.md if the change is intended."
            )
    return problems


def _listing_error(entries: list[dict]) -> str | None:
    """fs_tools reports a listing failure as a one-element [{'error': ...}] list."""
    for entry in entries:
        if "error" in entry:
            return str(entry["error"])
    return None


def list_resumes(directory: Path | str) -> list[dict]:
    """List resume files in `directory` via Milestone 1's ``list_files``.

    Returns the tool's own entries ({"name", "path", "extension", "size_bytes",
    "modified"}), sorted by name. Raises :class:`LoaderError` if the directory cannot be
    listed - a missing directory is a setup mistake, not a per-file problem.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise LoaderError(f"Resume directory not found: {root}")
    fs_tools = _import_fs_tools()
    with sandbox_root(root):
        # Listing "" is the sandbox root itself, so every returned path is a bare
        # filename and nothing above the resume directory is reachable.
        entries = fs_tools.list_files("", extension=",".join(RESUME_EXTENSIONS))
    error = _listing_error(entries)
    if error:
        raise LoaderError(f"Could not list resumes in {root}: {error}")
    return entries


def load_resumes(directory: Path | str) -> tuple[list[LoadedResume], list[SkippedResume]]:
    """Load every resume in `directory`, returning (loaded, skipped).

    Unreadable or empty files are returned as skips with the tool's own message rather
    than raising, so one corrupt resume cannot abort an index build - but they are never
    discarded silently either.
    """
    root = Path(directory).expanduser().resolve()
    entries = list_resumes(root)
    fs_tools = _import_fs_tools()
    loaded: list[LoadedResume] = []
    skipped: list[SkippedResume] = []
    with sandbox_root(root):
        for entry in entries:
            name = entry["name"]
            display_path = f"{root.name}/{name}" if root.name else name
            result = fs_tools.read_file(name)
            if not result["success"]:
                skipped.append(SkippedResume(display_path, result["error"] or "unknown read error"))
                continue
            text = result["content"] or ""
            if not text.strip():
                reason = result["warning"] or "file contains no extractable text"
                skipped.append(SkippedResume(display_path, reason))
                continue
            loaded.append(LoadedResume(
                path=root / name,
                display_path=display_path,
                text=text,
                metadata=result["metadata"] or {},
                warning=result["warning"],
            ))
    return loaded, skipped


def iter_documents(directory: Path | str) -> Iterator[LoadedResume]:
    """Yield each readable resume in `directory`, reporting skips on stderr."""
    loaded, skipped = load_resumes(directory)
    for skip in skipped:
        print(f"[resume_loader] skipping {skip.display_path}: {skip.reason}", file=sys.stderr)
    yield from loaded


def load_text_file(path: Path | str) -> str:
    """Read one file (a job description, say) through Milestone 1's ``read_file``.

    Accepts the same formats as a resume, so a JD can be a .pdf or .docx as easily as a
    .txt. Raises :class:`LoaderError` on failure, since a named input file that cannot be
    read is always a hard error.
    """
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise LoaderError(f"File not found: {target}")
    fs_tools = _import_fs_tools()
    with sandbox_root(target.parent):
        result = fs_tools.read_file(target.name)
    if not result["success"]:
        raise LoaderError(f"Could not read {target}: {result['error']}")
    text = result["content"] or ""
    if not text.strip():
        raise LoaderError(f"{target} contains no extractable text ({result['warning'] or 'empty'})")
    return text
