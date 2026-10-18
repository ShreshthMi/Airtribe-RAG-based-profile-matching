"""
File-system tools (Milestone 1 stand-in).

Small, dependency-light helpers to discover and read resume files from disk.
Supports .txt, .md, .pdf (via pypdf) and .docx (via python-docx).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def list_files(directory: str | Path, extensions: set[str] | None = None, recursive: bool = True) -> list[Path]:
    """Return sorted paths of files under `directory` whose extension is supported."""
    exts = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts)


def read_file(path: str | Path) -> str:
    """Read a file and return its plain text content."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx  # python-docx

        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported file type: {path}")


def iter_documents(directory: str | Path) -> Iterator[tuple[Path, str]]:
    """Yield (path, text) for each readable resume in `directory`."""
    for path in list_files(directory):
        try:
            text = read_file(path)
        except Exception as exc:  # keep going if a single file is corrupt
            print(f"[file_tools] skipping {path}: {exc}")
            continue
        if text.strip():
            yield path, text


def relative_path(path: str | Path, base: str | Path | None = None) -> str:
    """Path relative to `base` (default: cwd) for nicer output; falls back to absolute."""
    try:
        return os.path.relpath(Path(path).resolve(), Path(base or os.getcwd()).resolve())
    except ValueError:
        return str(Path(path).resolve())
