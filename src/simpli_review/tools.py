"""Tools the reviewer can call while it investigates a PR.

All read-only and confined to the checked-out repository: paths are
resolved under the repo root, `..`/absolute paths and `.git` are refused,
and env/secret-looking files are never returned. Every result is capped so
one call can't flood the context.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from simpli_review.index import RepoIndex, Symbol
from simpli_review.models import ReviewResult

MAX_READ_LINES = 400
SECRET_NAMES = (".env", ".pem", ".key", "id_rsa", "id_ed25519", ".p12", ".pfx")

_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file from the repository at the PR's head commit, with line numbers. "
        f"Returns at most {MAX_READ_LINES} lines; page with start_line.",
        "input_schema": {
            "type": "object",
            "properties": {"path": _STR, "start_line": _INT, "end_line": _INT},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_code",
        "description": "Regex search over every indexed file. Use it to find usages, similar code, "
        "config keys, or whether something is handled elsewhere. Optional glob narrows the paths.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": _STR, "path_glob": _STR},
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_definition",
        "description": "Where a function, method or class with this exact name is defined (code graph).",
        "input_schema": {
            "type": "object",
            "properties": {"name": _STR},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_callers",
        "description": "Functions anywhere in the repo that call this name (code graph). Use it to check "
        "whether a changed signature, return value or behaviour breaks a caller.",
        "input_schema": {
            "type": "object",
            "properties": {"name": _STR},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_directory",
        "description": "List the files and folders directly inside a directory ('' for the repo root).",
        "input_schema": {
            "type": "object",
            "properties": {"path": _STR},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "git_history",
        "description": "Recent commits touching a file (hash, date, subject) - to spot reverts, "
        "repeated fixes, or why code is the way it is.",
        "input_schema": {
            "type": "object",
            "properties": {"path": _STR, "limit": _INT},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_review",
        "description": "Submit the finished review. Call exactly once, as your final action, after you have "
        "verified every finding against the code.",
        "input_schema": ReviewResult.model_json_schema(),
    },
]


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, root: Path, index: RepoIndex) -> None:
        self.root = root.resolve()
        self.index = index
        self.files_read: set[str] = set()

    def run(self, name: str, args: dict[str, Any]) -> str:
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool {name!r}")
        result: str = handler(**args)
        return result

    # ---------- path safety ----------

    def _safe(self, path: str) -> Path:
        p = PurePosixPath(path.replace("\\", "/").strip())
        if p.is_absolute() or ".." in p.parts:
            raise ToolError("Paths must be relative to the repository root, without '..'.")
        if p.parts and p.parts[0] == ".git":
            raise ToolError("The .git directory is not readable.")
        if any(p.name == s or p.name.endswith(s) or p.name.startswith(".env") for s in SECRET_NAMES):
            raise ToolError("Secret-looking files are never read.")
        full = (self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise ToolError("Path escapes the repository.")
        return full

    # ---------- tools ----------

    def _t_read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        full = self._safe(path)
        if not full.is_file():
            raise ToolError(f"No such file: {path}")
        try:
            lines = full.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as e:
            raise ToolError(f"{path} is not a text file") from e
        start = max(1, start_line)
        end = min(len(lines), end_line or start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        self.files_read.add(path)
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
        more = (
            f"\n... ({len(lines) - end} more lines; call again with start_line={end + 1})"
            if end < len(lines)
            else ""
        )
        return f"{path} (lines {start}-{end} of {len(lines)})\n{body}{more}"

    def _t_search_code(self, pattern: str, path_glob: str | None = None) -> str:
        try:
            hits = self.index.search(pattern, path_glob)
        except Exception as e:  # re.error and friends
            raise ToolError(f"Bad pattern: {e}") from e
        if not hits:
            return "No matches."
        return "\n".join(f"{p}:{ln}: {text}" for p, ln, text in hits) + (
            "\n(first 40 matches; narrow the pattern or glob)" if len(hits) >= 40 else ""
        )

    def _t_find_definition(self, name: str) -> str:
        return _fmt_symbols(self.index.definitions(name)) or f"No definition named {name!r} in the index."

    def _t_find_callers(self, name: str) -> str:
        callers = self.index.callers_of(name, limit=25)
        return _fmt_symbols(callers) or f"No indexed function calls {name!r}."

    def _t_list_directory(self, path: str) -> str:
        full = self.root if path in ("", ".") else self._safe(path)
        if not full.is_dir():
            raise ToolError(f"No such directory: {path}")
        entries = sorted(full.iterdir(), key=lambda e: (e.is_file(), e.name))
        shown = [f"{e.name}/" if e.is_dir() else e.name for e in entries if e.name != ".git"][:200]
        return "\n".join(shown) or "(empty)"

    def _t_git_history(self, path: str, limit: int = 10) -> str:
        self._safe(path)
        try:
            n = max(1, min(limit, 30))
            out = subprocess.run(
                ["git", "log", f"-n{n}", "--follow", "--format=%h %ad %s", "--date=short", "--", path],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ToolError(f"git log failed: {e}") from e
        return out.stdout.strip() or "No history (shallow clone or new file)."


def _fmt_symbols(symbols: list[Symbol]) -> str:
    return "\n".join(f"{s.path}:{s.start}-{s.end}  {s.kind} {s.qualname}  |  {s.signature}" for s in symbols)
