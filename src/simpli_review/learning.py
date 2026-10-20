"""Learning from how the team reacted to past review comments.

Prompting and model-judged severity scores do not cut noise reliably;
filtering a new comment that is similar to several comments the team already
downvoted does (docs/HOW_IT_WORKS.md). Similarity here is token overlap, which
needs no extra service and is enough for "the same kind of nit again". The
same history also goes into the prompt, so the reviewer sees what this team
accepts and rejects.

Protected categories (security, data loss, ...) are never suppressed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from simpli_review.models import PROTECTED, Finding
from simpli_review.render import FINDING_MARKER, parse_finding_marker

SIMILARITY = 0.45  # Jaccard over word sets
DOWNVOTES_TO_SUPPRESS = 3  # suppressed when similar to >= 3 downvoted comments
_WORD = re.compile(r"[a-z][a-z0-9_]{2,}")
_STOP = {"the", "and", "this", "that", "with", "for", "are", "not", "but", "can", "should", "when", "from"}


@dataclass
class PastComment:
    text: str
    category: str | None
    up: int
    down: int

    @property
    def words(self) -> set[str]:
        return words(self.text)


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def past_bot_comments(raw: list[dict[str, Any]]) -> list[PastComment]:
    """Only comments this reviewer wrote (they carry the hidden marker)."""
    out: list[PastComment] = []
    for c in raw:
        body = c.get("body") or ""
        if FINDING_MARKER not in body:
            continue
        meta = parse_finding_marker(body)
        reactions = c.get("reactions") or {}
        text = meta.get("title", "") + " " + _strip_markup(body)
        out.append(
            PastComment(text, meta.get("category"), int(reactions.get("+1", 0)), int(reactions.get("-1", 0)))
        )
    return out


@dataclass
class Feedback:
    downvoted: list[PastComment]
    upvoted: list[PastComment]

    @classmethod
    def from_history(cls, history: list[PastComment]) -> Feedback:
        return cls(
            downvoted=[c for c in history if c.down > c.up],
            upvoted=[c for c in history if c.up > c.down],
        )

    def suppresses(self, finding: Finding) -> bool:
        if finding.category in PROTECTED:
            return False
        fw = words(f"{finding.title} {finding.body}")
        similar = sum(1 for c in self.downvoted if similarity(fw, c.words) >= SIMILARITY)
        return similar >= DOWNVOTES_TO_SUPPRESS

    def prompt_section(self, limit: int = 8) -> str:
        if not self.downvoted and not self.upvoted:
            return ""
        lines = ["## How this team reacted to earlier review comments"]
        if self.downvoted:
            lines.append("They rejected (👎) comments like these - do not raise this kind of point again:")
            lines += [f"- {_short(c.text)}" for c in self.downvoted[:limit]]
        if self.upvoted:
            lines.append("They valued (👍) comments like these - this is the bar to aim for:")
            lines += [f"- {_short(c.text)}" for c in self.upvoted[:limit]]
        return "\n".join(lines)


def _strip_markup(body: str) -> str:
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    return re.sub(r"[*_`#>|]", " ", body)


def _short(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"
