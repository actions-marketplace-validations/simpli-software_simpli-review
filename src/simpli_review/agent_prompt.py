"""'Prompt for AI agents': a finding rewritten as plain-text instructions a coding agent can act on.

Rendered deterministically from the finding (path, line, severity, title, explanation, and the
suggestion if there is one), so it needs no model call and no schema field. Every comment gets its
own collapsed block; the summary gets one combined block for every open finding.

All text goes through scrub.redact first (emails, token-like strings and HTML comments removed),
and runs of three backticks are neutralised so the text cannot break out of its code fence.
"""

from __future__ import annotations

from dataclasses import dataclass

from simpli_review.models import Finding
from simpli_review.scrub import redact, scrub

MAX_BODY_CHARS = 1500
MAX_SUGGESTION_CHARS = 2000
MAX_TITLE_CHARS = 200
MAX_COMBINED_CHARS = 20_000
VERIFY = (
    "Then verify: re-read the changed code and its callers, run the project's tests and linters, "
    "and confirm the problem no longer occurs."
)


@dataclass
class AgentTask:
    """What a coding agent needs to act on one finding."""

    path: str
    line: int | None
    end_line: int | None
    severity: str
    title: str
    body: str | None  # None: the explanation lives in an earlier review comment
    suggestion: str | None = None

    @classmethod
    def from_finding(cls, f: Finding) -> AgentTask:
        return cls(f.path, f.line, f.end_line, f.severity.value, f.title, f.body, f.suggestion)


def _safe(text: str) -> str:
    return redact(text).replace("```", "'''").strip()


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _where(t: AgentTask) -> str:
    path = scrub(t.path, 300)
    if t.line is None:
        return f"In {path}"
    if t.end_line and t.end_line > t.line:
        return f"In {path} around lines {t.line}-{t.end_line}"
    return f"In {path} around line {t.line}"


def task_text(t: AgentTask) -> str:
    lines = [f"{_where(t)}: [{t.severity}] {scrub(t.title, MAX_TITLE_CHARS).rstrip('.')}."]
    if t.body is None:
        lines.append(
            "Problem: raised in an earlier review of this PR; read its review thread for the details."
        )
    else:
        lines.append("Problem: " + _cap(_safe(t.body), MAX_BODY_CHARS))
    if t.suggestion is not None:
        lines += [
            "Fix: replace the code there with the following, adjusting it to the surrounding code:",
            _cap(_safe(t.suggestion), MAX_SUGGESTION_CHARS),
        ]
    elif t.body is None:
        lines.append("Fix: resolve it as that thread describes, with the smallest change that does.")
    else:
        lines.append("Fix: make the smallest change that resolves the problem described above.")
    lines.append(VERIFY)
    return "\n".join(lines)


def _details(summary: str, text: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n```text\n{text}\n```\n\n</details>"


def task_details(t: AgentTask) -> str:
    """The collapsed block for one task (an inline comment, or a scan issue)."""
    return _details("Prompt for AI agents", task_text(t))


def finding_details(f: Finding) -> str:
    """The collapsed block that goes on one inline comment."""
    return task_details(AgentTask.from_finding(f))


def combined_text(tasks: list[AgentTask]) -> str:
    head = (
        f"Fix the following {len(tasks)} open finding(s) on this pull request. Handle them one at a time: "
        "for each, change only what is needed, then verify before moving to the next."
    )
    parts = [head]
    used = len(head)
    for i, t in enumerate(tasks):
        entry = f"{i + 1}. " + task_text(t).replace("\n", "\n   ")
        if used + len(entry) > MAX_COMBINED_CHARS:
            parts.append(f"({len(tasks) - i} more finding(s) left out for length; see the review comments.)")
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


def combined_details(tasks: list[AgentTask]) -> str:
    """One collapsed block listing every open finding; empty when there are none."""
    if not tasks:
        return ""
    return _details(f"Prompt for AI agents ({len(tasks)} open)", combined_text(tasks))
