"""A light code graph of the repository under review.

Every file is parsed into functions, classes and the relationships between
them (calls, imports), and that graph is queried for each changed function:
what it calls, who calls it, and similar functions (docs/HOW_IT_WORKS.md).
It lives in memory, with no database:

- Python is parsed with `ast` (exact definitions, line ranges, calls, imports).
- JavaScript/TypeScript use regexes (definitions, imports, call names) - good
  enough to find callers and callees, not a type checker.
- Everything else is still searchable as text.

Built once per run in memory; a few thousand files take seconds.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILE_BYTES = 400_000
SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
TEXT_SUFFIXES = SOURCE_SUFFIXES | {
    ".go", ".rb", ".java", ".kt", ".cs", ".php", ".rs", ".swift", ".sql", ".md", ".yml", ".yaml",
    ".json", ".toml", ".ini", ".cfg", ".sh", ".html", ".css", ".scss", ".vue", ".svelte", ".txt",
}  # fmt: skip
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".mypy_cache"}

_JS_DEF = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\*?\s+(?P<fn>[A-Za-z_$][\w$]*)"
    r"|class\s+(?P<cls>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>))"
)
_JS_METHOD = re.compile(
    r"^\s+(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*(?P<m>[A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?::[^{]+)?\{\s*$"
)
_JS_IMPORT = re.compile(
    r"""(?:import\s[^'"]*from\s*|import\s*\(\s*|require\s*\(\s*)['"](?P<mod>[^'"]+)['"]"""
)
_CALL = re.compile(r"(?<![\w$.])(?P<name>[A-Za-z_$][\w$]*)\s*\(")
_JS_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "function",
    "typeof",
    "await",
    "new",
    "super",
    "import",
    "require",
}


@dataclass(frozen=True)
class Symbol:
    path: str
    name: str
    qualname: str
    kind: str  # function | method | class
    start: int
    end: int
    signature: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass
class FileInfo:
    path: str
    text: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)


class RepoIndex:
    def __init__(self, root: Path, ignored: Callable[[str], bool] = lambda _p: False) -> None:
        self.root = root
        self.files: dict[str, FileInfo] = {}
        self.defs: dict[str, list[Symbol]] = defaultdict(list)
        self.calls: dict[str, set[str]] = {}  # symbol key -> names it calls
        self.callers: dict[str, set[str]] = defaultdict(set)  # name -> symbol keys calling it
        self._by_key: dict[str, Symbol] = {}
        for rel in self._walk(ignored):
            self._index_file(rel)

    # ---------- building ----------

    def _walk(self, ignored: Callable[[str], bool]) -> Iterable[str]:
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel = p.relative_to(self.root).as_posix()
            if any(part in SKIP_DIRS for part in rel.split("/")[:-1]) or ignored(rel):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield rel

    def _index_file(self, rel: str) -> None:
        try:
            text = (self.root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return
        info = FileInfo(rel, text)
        self.files[rel] = info
        suffix = Path(rel).suffix.lower()
        if suffix == ".py":
            self._index_python(info)
        elif suffix in SOURCE_SUFFIXES:
            self._index_js(info)
        for sym in info.symbols:
            self.defs[sym.name].append(sym)
            self._by_key[sym.key] = sym
            for name in self.calls.get(sym.key, ()):
                self.callers[name].add(sym.key)

    def _index_python(self, info: FileInfo) -> None:
        try:
            tree = ast.parse(info.text)
        except SyntaxError:
            return
        lines = info.text.splitlines()

        def visit(node: ast.AST, prefix: str, in_class: bool) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    qual = f"{prefix}{child.name}"
                    kind = (
                        "class" if isinstance(child, ast.ClassDef) else ("method" if in_class else "function")
                    )
                    end = getattr(child, "end_lineno", child.lineno) or child.lineno
                    sig = lines[child.lineno - 1].strip() if child.lineno - 1 < len(lines) else child.name
                    sym = Symbol(info.path, child.name, qual, kind, child.lineno, end, sig)
                    info.symbols.append(sym)
                    if kind != "class":
                        self.calls[sym.key] = _python_calls(child)
                    visit(child, f"{qual}.", isinstance(child, ast.ClassDef))
                elif isinstance(child, ast.Import):
                    info.imports.update(a.name for a in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module:
                    info.imports.add(child.module)

        visit(tree, "", False)

    def _index_js(self, info: FileInfo) -> None:
        lines = info.text.splitlines()
        starts: list[tuple[int, str, str]] = []
        for i, line in enumerate(lines, start=1):
            for m in _JS_IMPORT.finditer(line):
                info.imports.add(m.group("mod"))
            d = _JS_DEF.match(line)
            if d:
                name = d.group("fn") or d.group("cls") or d.group("var")
                starts.append((i, name, "class" if d.group("cls") else "function"))
                continue
            mm = _JS_METHOD.match(line)
            if mm and mm.group("m") not in _JS_KEYWORDS:
                starts.append((i, mm.group("m"), "method"))
        for idx, (start, name, kind) in enumerate(starts):
            end = _block_end(lines, start, starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines))
            sym = Symbol(info.path, name, name, kind, start, end, lines[start - 1].strip()[:200])
            info.symbols.append(sym)
            if kind != "class":
                # Include the definition line: a one-line arrow function calls
                # things there. Its own name is not a call it makes.
                body = "\n".join(lines[start - 1 : end])
                self.calls[sym.key] = {
                    m.group("name")
                    for m in _CALL.finditer(body)
                    if m.group("name") not in _JS_KEYWORDS and m.group("name") != name
                }

    # ---------- queries ----------

    def symbols_touching(self, path: str, lines: set[int]) -> list[Symbol]:
        info = self.files.get(path)
        if not info or not lines:
            return []
        hit = [s for s in info.symbols if s.kind != "class" and any(s.start <= ln <= s.end for ln in lines)]
        # innermost first: a method, not its whole class
        return sorted(hit, key=lambda s: s.end - s.start)

    def callers_of(self, name: str, limit: int = 8) -> list[Symbol]:
        keys = sorted(self.callers.get(name, ()))
        return [self._by_key[k] for k in keys[:limit] if k in self._by_key]

    def callees_of(self, sym: Symbol, limit: int = 8) -> list[Symbol]:
        out: list[Symbol] = []
        for name in sorted(self.calls.get(sym.key, ())):
            if name == sym.name:
                continue
            out.extend(self.defs.get(name, [])[:2])
            if len(out) >= limit:
                break
        return out[:limit]

    def similar_to(self, sym: Symbol, limit: int = 4) -> list[Symbol]:
        """Functions that probably follow the same pattern: same name elsewhere,
        or siblings in the same file sharing a name token (create_x / create_y)."""
        tokens = set(re.split(r"_|(?<=[a-z])(?=[A-Z])", sym.name.lower())) - {""}
        same_name = [s for s in self.defs.get(sym.name, []) if s.key != sym.key]
        siblings = [
            s
            for s in self.files[sym.path].symbols
            if s.key != sym.key
            and s.kind == sym.kind
            and tokens & (set(re.split(r"_|(?<=[a-z])(?=[A-Z])", s.name.lower())) - {""})
        ]
        return (same_name + siblings)[:limit]

    def definitions(self, name: str, limit: int = 10) -> list[Symbol]:
        return self.defs.get(name, [])[:limit]

    def search(
        self, pattern: str, path_glob: str | None = None, limit: int = 40
    ) -> list[tuple[str, int, str]]:
        from simpli_review.config import match_glob

        rx = re.compile(pattern)
        hits: list[tuple[str, int, str]] = []
        for path, info in self.files.items():
            if path_glob and not match_glob(path, path_glob):
                continue
            for i, line in enumerate(info.text.splitlines(), start=1):
                if rx.search(line):
                    hits.append((path, i, line.strip()[:200]))
                    if len(hits) >= limit:
                        return hits
        return hits

    def importers_of(self, path: str, limit: int = 10) -> list[str]:
        stem = Path(path).with_suffix("").as_posix()
        module = stem.replace("/", ".")
        tail = stem.rsplit("/", 1)[-1]
        out = []
        for other, info in self.files.items():
            if other == path:
                continue
            if any(
                imp == module or imp.endswith(f".{module}") or imp.endswith(f"/{tail}")
                for imp in info.imports
            ):
                out.append(other)
                if len(out) >= limit:
                    break
        return out


def _python_calls(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _block_end(lines: list[str], start: int, hard_stop: int) -> int:
    """End of a brace block starting on `start` (1-based); falls back to the
    line before the next definition."""
    depth = 0
    opened = False
    for i in range(start - 1, min(len(lines), hard_stop + 400)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return i + 1
        if not opened and i + 1 >= hard_stop:
            return hard_stop
    return max(start, hard_stop)
