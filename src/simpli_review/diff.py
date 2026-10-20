"""Unified-diff parsing.

GitHub only accepts an inline review comment on a line that appears in the
PR's diff (an added line or a context line, on the new side). Each file's
patch is parsed once into the set of commentable new-side lines, and the diff
shown to the model is annotated with those new-side line numbers so it can
cite lines GitHub will accept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FilePatch:
    path: str
    status: str  # added | modified | removed | renamed
    patch: str
    previous_path: str | None = None
    added: set[int] = field(default_factory=set)
    commentable: set[int] = field(default_factory=set)
    additions: int = 0
    deletions: int = 0

    def annotated(self) -> str:
        """The patch with the new-side line number in front of every line
        that can carry a comment ('+' added, ' ' context); removed lines get
        a blank gutter."""
        out: list[str] = []
        new_line = 0
        for raw in self.patch.splitlines():
            m = _HUNK.match(raw)
            if m:
                new_line = int(m.group(3))
                out.append(raw)
                continue
            if raw.startswith("-"):
                out.append(f"{'':>6} {raw}")
            elif raw.startswith("\\"):
                out.append(raw)
            else:
                out.append(f"{new_line:>6} {raw}")
                new_line += 1
        return "\n".join(out)


def parse_patch(
    path: str, patch: str, status: str = "modified", previous_path: str | None = None
) -> FilePatch:
    fp = FilePatch(path=path, status=status, patch=patch, previous_path=previous_path)
    new_line = 0
    in_hunk = False
    for raw in patch.splitlines():
        m = _HUNK.match(raw)
        if m:
            new_line = int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk or raw.startswith("\\"):
            continue
        if raw.startswith("+"):
            fp.added.add(new_line)
            fp.commentable.add(new_line)
            fp.additions += 1
            new_line += 1
        elif raw.startswith("-"):
            fp.deletions += 1
        else:
            fp.commentable.add(new_line)
            new_line += 1
    return fp


def split_unified_diff(text: str) -> list[FilePatch]:
    """A whole `git diff` (local mode) into per-file patches."""
    patches: list[FilePatch] = []
    path: str | None = None
    old_path: str | None = None
    status = "modified"
    body: list[str] = []

    def flush() -> None:
        if path is not None:
            patches.append(parse_patch(path, "\n".join(body), status, old_path if old_path != path else None))

    for line in text.splitlines():
        if line.startswith("diff --git "):
            flush()
            path, old_path, status, body = None, None, "modified", []
            m = re.match(r"diff --git a/(.+) b/(.+)$", line)
            if m:
                old_path, path = m.group(1), m.group(2)
        elif line.startswith("new file mode"):
            status = "added"
        elif line.startswith("deleted file mode"):
            status = "removed"
        elif line.startswith("rename from"):
            status = "renamed"
        elif line.startswith("+++ ") or line.startswith("--- ") or line.startswith("index "):
            continue
        elif path is not None and (body or line.startswith("@@")):
            body.append(line)
    flush()
    return patches


def nearest_commentable(fp: FilePatch, line: int, max_distance: int = 3) -> int | None:
    """The model sometimes cites a line just outside the hunk. Snap to the
    closest commentable line within a few lines, else None (comment moves to
    the summary instead of being dropped)."""
    if line in fp.commentable:
        return line
    for d in range(1, max_distance + 1):
        for candidate in (line - d, line + d):
            if candidate in fp.commentable:
                return candidate
    return None
