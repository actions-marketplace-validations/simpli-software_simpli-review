"""Merge blocking: decide whether this review should turn the Actions check red.

A review blocks when, with `blockOn` set to P0 or P1,

- any finding at or above that severity is open: newly posted (inline or in the summary), or an
  earlier finding on this PR that the per-PR memory still counts as open; or
- any check in `error` mode failed.

Findings the filters dropped (strictness, confidence, team feedback, duplicates) do not count.
A PR carrying the override label is never blocked; the summary then says so. Reviewer failures are
not handled here: they keep their old behaviour (a warning, exit 0).
"""

from __future__ import annotations

from simpli_review.agent_prompt import AgentTask
from simpli_review.checks import failed_error_checks
from simpli_review.config import Config
from simpli_review.learning import similarity, words
from simpli_review.memory import PriorFinding, SinceLast
from simpli_review.models import BlockDecision, CheckRow, Finding, Severity

_RANK = {Severity.P0: 0, Severity.P1: 1, Severity.P2: 2}
SAME_FINDING = 0.6  # same threshold finalize uses to call a new finding a repeat of a posted one


def _severity(value: str) -> Severity:
    try:
        return Severity(value)
    except ValueError:
        return Severity.P2


def open_tasks(new: list[Finding], still_open: list[PriorFinding]) -> list[AgentTask]:
    """Every open finding: this run's new ones, then earlier ones still open (repeats dropped)."""
    tasks = [AgentTask.from_finding(f) for f in new]
    seen = [(t.path, words(t.title)) for t in tasks]
    for p in still_open:
        pw = words(p.title)
        if any(path == p.path and similarity(pw, w) >= SAME_FINDING for path, w in seen):
            continue
        tasks.append(AgentTask(p.path, p.line, None, _severity(p.severity).value, p.title, None))
        seen.append((p.path, pw))
    return tasks


def still_open_findings(since: SinceLast | None) -> list[PriorFinding]:
    return since.still_open if since else []


def evaluate(
    config: Config,
    *,
    tasks: list[AgentTask],
    checks: list[CheckRow],
    labels: list[str],
) -> BlockDecision:
    reasons: list[str] = []
    if config.block_on != "none":
        limit = _RANK[Severity(config.block_on)]
        at_or_above = [t for t in tasks if _RANK[_severity(t.severity)] <= limit]
        if at_or_above:
            counts = ", ".join(
                f"{sum(1 for t in at_or_above if t.severity == s.value)} x {s.value}"
                for s in Severity
                if any(t.severity == s.value for t in at_or_above)
            )
            reasons.append(
                f"{len(at_or_above)} open finding(s) at or above {config.block_on} ({counts}), "
                f"new or still open from earlier reviews"
            )
    for row in failed_error_checks(checks):
        reasons.append(f"check `{row.name}` failed (error mode): {row.reason}")
    if not reasons:
        return BlockDecision(blocking=False)
    if config.override_label and config.override_label in labels:
        return BlockDecision(blocking=False, reasons=reasons, overridden_by=config.override_label)
    return BlockDecision(blocking=True, reasons=reasons)
