"""GitHub issues for scan findings: fingerprints, de-duplication, the issue text, and the cap.

Rules (see README, "AI scan"):
- Only findings at or above `scan.minSeverity` become issues, at most `scan.maxIssues` per run.
- A finding whose fingerprint matches an open issue, or an issue closed as "not planned", is skipped.
  A match is the same fingerprint, or the same file with a similar title (a model words the same problem
  differently from one run to the next).
- Nothing is ever closed or commented on automatically: scans are not deterministic.
- All text is scrubbed of secret-looking values before it is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

from simpli_review.agent_prompt import AgentTask, task_details
from simpli_review.config import ScanConfig
from simpli_review.github import GitHub, GitHubError
from simpli_review.learning import similarity, words
from simpli_review.models import ScanFinding, Severity
from simpli_review.scrub import redact_secrets

log = logging.getLogger(__name__)

LABEL = "ai-scan"
LABEL_COLOR = "5319e7"
LABEL_DESCRIPTION = "Found by the scheduled AI security scan"
TITLE_PREFIX = "[AI scan] "
SCAN_MARKER = "<!-- simpli-review:scan"
SAME_FINDING = 0.6  # Jaccard over title words, same file (the PR reviewer uses the same threshold)

_SEVERITY_ORDER = {Severity.P0: 0, Severity.P1: 1, Severity.P2: 2}
_REF = re.compile(r"(?<![\w/.:-])([\w][\w./-]*\.\w+):(\d+)(?:-(\d+))?")
_MENTION = re.compile(r"(?<![\w`])@(?=\w)")

IssueStatus = Literal[
    "pending",  # will be opened
    "created",
    "dry_run",  # would be opened
    "duplicate_open",
    "duplicate_not_planned",
    "below_min_severity",
    "over_cap",
    "failed",
]


@dataclass
class Fingerprint:
    digest: str
    path: str
    words: list[str]


def fingerprint(path: str, title: str) -> Fingerprint:
    """Path plus the sorted, normalised words of the title. No line numbers, so it survives edits above."""
    tokens = sorted(words(title))
    digest = hashlib.sha256(f"{path}\n{' '.join(tokens)}".encode()).hexdigest()[:16]
    return Fingerprint(digest, path, tokens)


def marker(fp: Fingerprint) -> str:
    payload = json.dumps({"fp": fp.digest, "path": fp.path, "words": fp.words}, sort_keys=True)
    # "<" and ">" are escaped so a path or word can't close the HTML comment.
    payload = payload.replace("<", chr(92) + "u003c").replace(">", chr(92) + "u003e")
    return f"{SCAN_MARKER} {payload} -->"


def parse_marker(body: str) -> Fingerprint | None:
    m = re.search(re.escape(SCAN_MARKER) + r"\s*(\{.*?\})\s*-->", body or "", flags=re.S)
    if not m:
        return None
    try:
        meta = json.loads(m.group(1))
        return Fingerprint(str(meta["fp"]), str(meta["path"]), [str(w) for w in meta["words"]])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


@dataclass
class KnownIssue:
    fp: Fingerprint
    number: int
    state: str  # open | closed
    state_reason: str | None  # completed | not_planned | reopened | None


def known_issues(raw: list[dict[str, Any]]) -> list[KnownIssue]:
    out: list[KnownIssue] = []
    for i in raw:
        fp = parse_marker(i.get("body") or "")
        if fp:
            out.append(KnownIssue(fp, int(i["number"]), str(i.get("state")), i.get("state_reason")))
    return out


def match_known(fp: Fingerprint, known: list[KnownIssue]) -> IssueStatus | None:
    """The reason to skip a finding, or None when no open / not-planned issue is the same finding."""
    mine = set(fp.words)
    for k in known:
        suppresses = k.state == "open" or k.state_reason == "not_planned"
        if not suppresses:
            continue
        same = k.fp.digest == fp.digest or (
            k.fp.path == fp.path and similarity(mine, set(k.fp.words)) >= SAME_FINDING
        )
        if same:
            return "duplicate_open" if k.state == "open" else "duplicate_not_planned"
    return None


@dataclass
class IssueAction:
    finding: ScanFinding
    fp: Fingerprint
    status: IssueStatus = "pending"
    number: int | None = None
    url: str | None = None


@dataclass
class IssuePlan:
    actions: list[IssueAction] = field(default_factory=list)

    def count(self, status: IssueStatus) -> int:
        return sum(1 for a in self.actions if a.status == status)


def plan_issues(findings: list[ScanFinding], known: list[KnownIssue], scan: ScanConfig) -> IssuePlan:
    """Decide, for every finding, whether it becomes an issue. Pure: no GitHub calls."""
    plan = IssuePlan()
    seen_open: list[KnownIssue] = list(known)
    created = 0
    minimum = _SEVERITY_ORDER[Severity(scan.min_severity)]
    for f in sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], -f.confidence, f.path, f.line)):
        action = IssueAction(f, fingerprint(f.path, f.title))
        plan.actions.append(action)
        if _SEVERITY_ORDER[f.severity] > minimum:
            action.status = "below_min_severity"
            continue
        dup = match_known(action.fp, seen_open)
        if dup:
            action.status = dup
            continue
        if created >= scan.max_issues:
            action.status = "over_cap"
            continue
        created += 1
        # Two findings in one run that are the same problem become one issue.
        seen_open.append(KnownIssue(action.fp, 0, "open", None))
    return plan


# ---------- issue text ----------


def _safe_md(text: str) -> str:
    """Scrub secret-looking values and stop @mentions from pinging people."""
    return _MENTION.sub("@​", redact_secrets(text))


def issue_title(f: ScanFinding) -> str:
    title = " ".join(redact_secrets(f.title).split())
    return (TITLE_PREFIX + title)[:240]


def blob_url(
    server: str, repo: str, sha: str, path: str, line: int | None = None, end: int | None = None
) -> str:
    url = f"{server.rstrip('/')}/{repo}/blob/{sha}/{quote(path)}"
    if line:
        url += f"#L{line}" + (f"-L{end}" if end and end > line else "")
    return url


def _link_refs(text: str, *, server: str, repo: str, sha: str, files: set[str]) -> str:
    def link(m: re.Match[str]) -> str:
        path, a, b = m.group(1), int(m.group(2)), m.group(3)
        if path not in files:
            return m.group(0)
        return f"[{m.group(0)}]({blob_url(server, repo, sha, path, a, int(b) if b else None)})"

    return _REF.sub(link, text)


def issue_body(
    a: IssueAction,
    *,
    server: str,
    repo: str,
    sha: str,
    files: set[str],
) -> str:
    f = a.finding
    where = f"{f.path}:{f.line}" + (f"-{f.end_line}" if f.end_line and f.end_line > f.line else "")
    task = AgentTask(
        f.path,
        f.line,
        f.end_line,
        f.severity.value,
        _safe_md(f.title),
        _safe_md(f"{f.body}\n\nEvidence: {f.evidence}"),
    )
    evidence = _link_refs(_safe_md(f.evidence.strip()), server=server, repo=repo, sha=sha, files=files)
    parts = [
        f"**{f.severity.value}** · {f.category.value} · confidence {f.confidence}/5",
        "",
        f"**Where:** [{where}]({blob_url(server, repo, sha, f.path, f.line, f.end_line)}) "
        f"(at scanned commit `{sha[:7]}`)",
        "",
        "### What is wrong",
        _safe_md(f.body.strip()),
        "",
        "### Evidence",
        evidence,
        "",
        task_details(task),
        "",
        f"<sub>Found by an automated AI scan of `{sha[:7]}`. AI scans are not deterministic, so this may be "
        "a "
        "false positive or may not be raised again next time. Nothing closes this issue automatically. "
        "Close it as *not planned* if it is not a problem and it will not be raised again.</sub>",
        marker(a.fp),
    ]
    return "\n".join(parts)


# ---------- publishing ----------


def publish(
    gh: GitHub,
    plan: IssuePlan,
    *,
    server: str,
    repo: str,
    sha: str,
    files: set[str],
    dry_run: bool = False,
) -> None:
    """Open the issues the plan marks pending. A failure on one issue never stops the others."""
    pending = [a for a in plan.actions if a.status == "pending"]
    if not pending:
        return
    if dry_run:
        for a in pending:
            a.status = "dry_run"
        return
    try:
        gh.ensure_label(LABEL, LABEL_COLOR, LABEL_DESCRIPTION)
    except GitHubError as e:
        log.warning("could not create the %s label (%s); not opening issues", LABEL, e.status)
        for a in pending:
            a.status = "failed"
        return
    for a in pending:
        try:
            issue = gh.create_issue(
                issue_title(a.finding),
                issue_body(a, server=server, repo=repo, sha=sha, files=files),
                [LABEL],
            )
            a.status, a.number, a.url = "created", int(issue["number"]), str(issue.get("html_url") or "")
        except GitHubError as e:
            log.warning("could not open an issue (%s)", e.status)
            a.status = "failed"
