"""What a review produces. One shape for the model's output and for rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """P0 must not merge; P1 should be fixed before merge; P2 worth fixing."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class CommentType(StrEnum):
    logic = "logic"
    syntax = "syntax"
    style = "style"


class Category(StrEnum):
    """Categories in PROTECTED are never suppressed by learned feedback."""

    security = "security"
    data_loss = "data_loss"
    concurrency = "concurrency"
    resource_leak = "resource_leak"
    null_safety = "null_safety"
    input_validation = "input_validation"
    correctness = "correctness"
    error_handling = "error_handling"
    performance = "performance"
    maintainability = "maintainability"
    convention = "convention"


PROTECTED = frozenset(
    {
        Category.security,
        Category.data_loss,
        Category.concurrency,
        Category.resource_leak,
        Category.null_safety,
        Category.input_validation,
    }
)


class Finding(BaseModel):
    path: str = Field(description="Repository-relative path of the file the comment belongs on.")
    line: int = Field(
        description="Line number in the NEW version of the file; must be a changed or context line."
    )
    end_line: int | None = Field(
        default=None, description="Last line of a multi-line range on the new side, or null."
    )
    severity: Severity
    type: CommentType
    category: Category
    title: str = Field(description="One-line statement of the problem.")
    body: str = Field(description="Why it is wrong, the concrete failure scenario, and how to fix it.")
    evidence: str = Field(
        description="Where in the codebase you confirmed this (file:line references you actually read)."
    )
    suggestion: str | None = Field(
        default=None,
        description="Exact replacement code for line..end_line (new side), or null. No diff markers.",
    )
    confidence: int = Field(ge=1, le=5, description="5 = verified by reading the code paths involved.")
    rule_id: str | None = Field(default=None, description="Id of the team rule this enforces, if any.")


class FileNote(BaseModel):
    path: str
    summary: str = Field(description="What changed in this file, one sentence.")


class PreviousStatus(BaseModel):
    ref: int = Field(description="The [number] of the finding in 'Previous findings on this PR'.")
    status: Literal["fixed", "still_present", "dismissed"]
    note: str = Field(default="", description="One short sentence of evidence, or why it is dismissed.")


# off = not run, warning = shown in the summary, error = a failure blocks the merge (see blocking.py).
CheckMode = Literal["off", "warning", "error"]
CheckOutcome = Literal["pass", "fail", "inconclusive"]
MAX_CHECK_NAME = 50
MAX_CHECK_INSTRUCTIONS = 1000


class ModelCheck(BaseModel):
    """The model's verdict on one pre-merge check it was asked to evaluate."""

    name: str = Field(
        min_length=1, max_length=MAX_CHECK_NAME, description="Exactly the name of the check, as listed."
    )
    result: CheckOutcome = Field(
        description="pass, fail, or inconclusive when the available tools cannot settle it."
    )
    reason: str = Field(description="One line saying why. For a fail, name the concrete thing found.")


@dataclass
class CheckRow:
    """One line of the pre-merge checks table in the summary comment."""

    name: str
    result: CheckOutcome
    mode: CheckMode
    reason: str


@dataclass
class BlockDecision:
    """Whether a review should fail the Actions check, and why."""

    blocking: bool  # the check should fail
    reasons: list[str] = field(default_factory=list)  # why it would block, even when overridden
    overridden_by: str | None = None  # the PR label that skipped a block that would have happened


class ReviewResult(BaseModel):
    summary: str = Field(description="Plain-language summary of what the PR does and the main risks.")
    confidence_score: int = Field(
        ge=0, le=5, description="0-5 merge confidence; 5 = safe to merge, weighted by how risky the area is."
    )
    confidence_reason: str = Field(description="One or two sentences explaining the score.")
    files: list[FileNote]
    diagram: str | None = Field(
        default=None,
        description="A mermaid diagram (sequenceDiagram/flowchart/classDiagram/erDiagram) of the change, "
        "without the ``` fence, or null for trivial changes.",
    )
    findings: list[Finding]
    previous_findings: list[PreviousStatus] = Field(
        default_factory=list,
        description="Re-reviews only: one entry per finding listed under 'Previous findings on this PR'. "
        "Leave empty when that section is absent.",
    )
    checks: list[ModelCheck] = Field(
        default_factory=list,
        description="One entry per check listed under 'Pre-merge checks to evaluate'. "
        "Leave empty when that section is absent.",
    )


class ScanFinding(BaseModel):
    """A problem found by reading the code at one commit (no diff, so no changed-line rule)."""

    path: str = Field(description="Repository-relative path of the file where the problem is.")
    line: int = Field(ge=1, description="Line number in that file where the problem is.")
    end_line: int | None = Field(default=None, description="Last line of a range, or null.")
    severity: Severity
    category: Category
    title: str = Field(description="One-line statement of the problem. No secret values.")
    body: str = Field(
        description="Why it is a problem, the concrete failure or attack scenario, and how to fix it. "
        "Never quote a secret value; say where it is."
    )
    evidence: str = Field(
        description="The file:line references you actually read that prove it, e.g. 'app/auth.py:42 ...'."
    )
    confidence: int = Field(ge=1, le=5, description="5 = verified by reading every code path involved.")
    rule_id: str | None = Field(default=None, description="Id of the team rule this enforces, if any.")


class ScanResult(BaseModel):
    summary: str = Field(description="Two or three sentences: what was covered, and the overall risk.")
    not_covered: list[str] = Field(
        default_factory=list,
        description="Areas of the repository you did not get to, so the next scan can look there.",
    )
    findings: list[ScanFinding]
