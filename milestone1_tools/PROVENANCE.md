# Milestone 1 file-system tools (vendored)

`fs_tools.py` and `extractors.py` are **verbatim copies** of the Milestone 1 deliverable,
the LLM-Powered File System Assistant. They are the tools Part A loads every resume
through; `resume_loader.py` in the project root is the only adapter around them.

| | |
|---|---|
| Source repository | `LLM-Powered-File-System-Assistant` |
| Commit | `fc7cf0c1636205fc2ec4bbcbe2c726fdbd1290f9` (`fc7cf0c`, 2026-10-09) |
| Copied on | 2026-10-18 |
| `fs_tools.py` | sha256 `9d7ea00f75843ec863dfa722d7d0442a1a5651b39890c93acc62a803b8464e38` |
| `extractors.py` | sha256 `239a72f23c5ba5b8d731a0d89773c85ef58de49e973a1278c06277960ef84cda` |

## Why they are copied rather than imported

The assignment is submitted as one self-contained repository, so it cannot depend on a
sibling checkout existing at a particular path on the grader's machine. The copies are
unmodified, and `verify_vendored_tools()` in `resume_loader.py` re-checks the two hashes
above, so an accidental local edit is reported instead of silently diverging.

To run against the live Milestone 1 checkout instead of these copies, point an
environment variable at it - no code change:

```bash
export MILESTONE1_TOOLS_DIR=~/code/LLM-Powered-File-System-Assistant
python resume_rag.py --resumes resumes --db chroma_db --reset
```

## Refreshing the copies

```bash
M=~/code/LLM-Powered-File-System-Assistant
cp "$M/fs_tools.py" "$M/extractors.py" milestone1_tools/
shasum -a 256 milestone1_tools/fs_tools.py milestone1_tools/extractors.py   # update the table above
```

## Not a package, on purpose

`fs_tools.py` does a flat `import extractors`, so this directory is put on `sys.path` and
deliberately has no `__init__.py`. Making it a package would mean editing Milestone 1
code, which would cost the "verbatim" guarantee the hashes above provide.

## What the project relies on

* `read_file(filepath) -> {"success", "filepath", "content", "metadata", "warning", "error"}`
  - PDF via pypdf, DOCX including tables/headers/footers/text boxes, and plain text.
  Never raises; failures come back as `success: False` with a readable `error`.
* `list_files(directory, extension=None) -> list[dict]` - one `{"name", "path", ...}` per
  file, sorted by name. **Not recursive**; sub-folders, hidden names and Office lock files
  (`~$...`) are skipped. Errors arrive as a one-element `[{"error": ...}]` list.
* Every path is confined to `$FS_ASSISTANT_ROOT` (or the current directory when unset).
  `..`, absolute paths outside the root, `~` and symlink escapes are refused, and so are
  hidden files - which is why `resume_loader.py` sets that root explicitly per call rather
  than trusting the working directory.
