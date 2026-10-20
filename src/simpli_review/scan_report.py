"""The scan's outputs: the spend report (job summary and console), simpli-review-scan.json and .md.

Cost visibility is the point of this file: every run reports the model, effort, turns and tokens, and an
ESTIMATED cost in USD from the price table in pricing.py. The report is written even when the scan fails,
because a failed run still spent money.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from simpli_review.models import ScanFinding
from simpli_review.pricing import PRICES, PRICES_AS_OF, estimate_micros, format_usd
from simpli_review.scan_issues import IssueAction, IssuePlan

JSON_NAME = "simpli-review-scan.json"
MD_NAME = "simpli-review-scan.md"
ARTIFACT_NAME = "simpli-review-scan"
REPORT_VERSION = 1

Status = Literal["complete", "incomplete"]


@dataclass
class ScanReport:
    status: Status
    repository: str
    commit: str
    started_at: str  # ISO 8601, UTC
    duration_seconds: int
    model: str
    effort: str
    max_turns: int
    turns: int
    usage: dict[str, int]
    budget_tokens: int
    budget_nudged_at: int | None
    summary: str = ""
    not_covered: list[str] = field(default_factory=list)
    findings: list[ScanFinding] = field(default_factory=list)
    plan: IssuePlan = field(default_factory=IssuePlan)
    dropped_low_confidence: int = 0
    dropped_invalid: int = 0
    duplicates_in_run: int = 0
    error: str | None = None
    name: str = "AI scan"

    @property
    def total_tokens(self) -> int:
        return sum(self.usage.values())

    @property
    def cost_micros(self) -> int | None:
        return estimate_micros(self.model, self.usage)

    @property
    def cost_text(self) -> str:
        return format_usd(self.cost_micros)

    # ---------- JSON ----------

    def to_dict(self) -> dict[str, Any]:
        by_finding: dict[int, IssueAction] = {id(a.finding): a for a in self.plan.actions}
        findings: list[dict[str, Any]] = []
        for f in self.findings:
            a = by_finding.get(id(f))
            row = f.model_dump(mode="json")
            row["fingerprint"] = a.fp.digest if a else None
            row["issue"] = {"status": a.status, "number": a.number, "url": a.url} if a else None
            findings.append(row)
        price = PRICES.get(self.model)
        return {
            "version": REPORT_VERSION,
            "status": self.status,
            "error": self.error,
            "repository": self.repository,
            "commit": self.commit,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "effort": self.effort,
            "turns": {"used": self.turns, "max": self.max_turns},
            "usage": {**self.usage, "total": self.total_tokens},
            "budget": {
                "max_tokens": self.budget_tokens,
                "used_tokens": self.total_tokens,
                "told_to_submit_at_turn": self.budget_nudged_at,
            },
            "cost": {
                "is_estimate": True,
                "currency": "USD",
                "estimate_usd_micros": self.cost_micros,
                "estimate_usd": self.cost_text,
                "prices_as_of": PRICES_AS_OF,
                "price_cents_per_million_tokens": vars(price) if price else None,
            },
            "summary": self.summary,
            "not_covered": self.not_covered,
            "dropped": {
                "low_confidence": self.dropped_low_confidence,
                "invalid_location_or_evidence": self.dropped_invalid,
                "duplicate_in_run": self.duplicates_in_run,
            },
            "issues": {
                "created": self.plan.count("created"),
                "would_create": self.plan.count("dry_run"),
                "failed": self.plan.count("failed"),
                "skipped_duplicate_open": self.plan.count("duplicate_open"),
                "skipped_duplicate_not_planned": self.plan.count("duplicate_not_planned"),
                "over_cap": self.plan.count("over_cap"),
                "below_min_severity": self.plan.count("below_min_severity"),
            },
            "findings": findings,
        }

    # ---------- Markdown ----------

    def spend_lines(self) -> list[str]:
        """The plain-text spend report printed to the log."""
        lines = [
            f"model: {self.model}  effort: {self.effort}  turns: {self.turns}/{self.max_turns}",
            f"tokens: input {self.usage['input']:,}  output {self.usage['output']:,}  "
            f"cache read {self.usage['cache_read']:,}  cache write {self.usage['cache_write']:,}  "
            f"total {self.total_tokens:,} (budget {self.budget_tokens:,})",
            f"estimated cost (estimate, list prices as of {PRICES_AS_OF}): {self.cost_text}",
        ]
        return lines

    def markdown(self) -> str:
        u = self.usage
        head = "complete" if self.status == "complete" else "INCOMPLETE"
        out = [
            f"# {self.name}: {head}",
            "",
            f"Repository `{self.repository}` at commit `{self.commit[:7]}`, scanned {self.started_at} "
            f"({self.duration_seconds // 60} min {self.duration_seconds % 60} s).",
        ]
        if self.error:
            out += ["", f"> **The scan did not finish:** {self.error}. The spend below is real."]
        out += [
            "",
            "## Spend",
            "",
            "| | |",
            "|---|---|",
            f"| Model | `{self.model}` |",
            f"| Effort | {self.effort} |",
            f"| Turns used | {self.turns} of {self.max_turns} |",
            f"| Input tokens | {u['input']:,} |",
            f"| Output tokens | {u['output']:,} |",
            f"| Cache read tokens | {u['cache_read']:,} |",
            f"| Cache write tokens | {u['cache_write']:,} |",
            f"| Total tokens | {self.total_tokens:,} of a {self.budget_tokens:,} budget |",
            f"| **Estimated cost (estimate)** | **{self.cost_text}** |",
            "",
            f"The cost is an estimate: the tokens above priced at list prices as of {PRICES_AS_OF} "
            "(`src/simpli_review/pricing.py`). Check the Anthropic usage page for the real figure.",
        ]
        if self.budget_nudged_at is not None:
            out += [
                "",
                f"The token budget was nearly used after turn {self.budget_nudged_at}: the model was told to "
                "submit what it had. Raise `scan.maxInputTokens` to let it look further.",
            ]
        if self.status == "complete":
            out += ["", "## Summary", "", self.summary.strip() or "(none)"]
            if self.not_covered:
                out += ["", "Not covered: " + "; ".join(self.not_covered)]
        issues = self.plan
        out += [
            "",
            "## Findings",
            "",
            f"{len(self.findings)} kept. Dropped: {self.dropped_low_confidence} below minConfidence, "
            f"{self.dropped_invalid} without a real file:line or evidence, "
            f"{self.duplicates_in_run} repeated. "
            f"Issues: {issues.count('created')} opened, {issues.count('dry_run')} would open (dry run), "
            f"{issues.count('failed')} failed, "
            f"{issues.count('duplicate_open') + issues.count('duplicate_not_planned')} skipped as known, "
            f"{issues.count('over_cap')} over the cap, "
            f"{issues.count('below_min_severity')} below minSeverity.",
        ]
        if self.findings:
            out += ["", "| Severity | Where | Title | Conf. | Issue |", "|---|---|---|---|---|"]
            by_finding = {id(a.finding): a for a in issues.actions}
            for f in self.findings:
                out.append(
                    f"| {f.severity.value} | `{f.path}:{f.line}` | {_cell(f.title)} | {f.confidence} | "
                    f"{_issue_cell(by_finding.get(id(f)))} |"
                )
            for f in self.findings:
                out += [
                    "",
                    f"### {f.severity.value} {_cell(f.title)}",
                    "",
                    f"`{f.path}:{f.line}` · {f.category.value} · confidence {f.confidence}/5",
                    "",
                    f.body.strip(),
                    "",
                    f"Evidence: {f.evidence.strip()}",
                ]
        return "\n".join(out) + "\n"

    # ---------- files ----------

    def write(self, directory: Path) -> tuple[Path, Path]:
        j = directory / JSON_NAME
        m = directory / MD_NAME
        j.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        m.write_text(self.markdown(), encoding="utf-8")
        return j, m

    def append_step_summary(self, path: str) -> None:
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(self.markdown() + "\n")
        except OSError:
            pass  # the summary is a convenience; the files and the log carry the same numbers


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _issue_cell(a: IssueAction | None) -> str:
    if a is None:
        return ""
    if a.status == "created" and a.number:
        return f"#{a.number}"
    return {
        "dry_run": "would open",
        "failed": "failed",
        "duplicate_open": "already open",
        "duplicate_not_planned": "closed as not planned",
        "over_cap": "over cap",
        "below_min_severity": "report only",
    }.get(a.status, a.status)
