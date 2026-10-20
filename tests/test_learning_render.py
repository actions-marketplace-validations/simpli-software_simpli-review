from __future__ import annotations

from simpli_review.learning import Feedback, PastComment, past_bot_comments
from simpli_review.models import Category, CommentType, FileNote, Finding, ReviewResult, Severity
from simpli_review.render import (
    SUMMARY_MARKER,
    finding_marker,
    inline_body,
    parse_finding_marker,
    summary_body,
)


def _f(title: str, body: str, category: Category = Category.maintainability, **kw: object) -> Finding:
    base = dict(path="a.py", line=3, severity=Severity.P2, type=CommentType.logic, category=category,
                title=title, body=body, evidence="a.py:3", confidence=4)  # fmt: skip
    base.update(kw)
    return Finding.model_validate(base)


NIT = "Consider extracting this helper into a separate utility function for readability"


def _nits(n: int, down: int = 2, up: int = 0) -> list[PastComment]:
    return [PastComment(NIT, "maintainability", up=up, down=down) for _ in range(n)]


def test_suppressed_when_similar_to_three_downvoted():
    finding = _f("Extract helper", "Consider extracting this helper into a separate utility function")
    assert Feedback.from_history(_nits(3)).suppresses(finding)
    assert not Feedback.from_history(_nits(2)).suppresses(finding)


def test_upvotes_outweighing_downvotes_do_not_count():
    finding = _f("Extract helper", "Consider extracting this helper into a separate utility function")
    assert not Feedback.from_history(_nits(5, down=1, up=3)).suppresses(finding)


def test_protected_categories_are_never_suppressed():
    finding = _f("Extract helper", "Consider extracting this helper into a separate utility function",
                 category=Category.security)  # fmt: skip
    assert not Feedback.from_history(_nits(10)).suppresses(finding)


def test_past_bot_comments_reads_only_our_markers_and_reactions():
    ours = inline_body(_f("Unchecked None", "x may be None here"))
    raw = [
        {"body": ours, "reactions": {"+1": 0, "-1": 2}},
        {"body": "a human comment", "reactions": {"-1": 5}},
    ]
    (only,) = past_bot_comments(raw)
    assert only.down == 2 and only.category == "maintainability" and "Unchecked None" in only.text
    section = Feedback.from_history([only]).prompt_section()
    assert "rejected" in section and "Unchecked None" in section


def test_marker_round_trip():
    f = _f('Title with "quotes" -->', "b")
    meta = parse_finding_marker(f"text\n{finding_marker(f)}")
    assert meta["title"] == f.title and meta["severity"] == "P2"


def test_inline_body_has_badge_suggestion_and_evidence():
    body = inline_body(_f("Off by one", "loop skips last item", suggestion="for i in range(n + 1):",
                          severity=Severity.P1, rule_id="loops"))  # fmt: skip
    assert body.startswith("**🟠 P1 · Logic** — Off by one")
    assert "```suggestion\nfor i in range(n + 1):\n```" in body
    assert "Team rule: `loops`" in body and "<summary>Evidence</summary>" in body


def test_summary_body_sections():
    result = ReviewResult(
        summary="Adds cents conversion.", confidence_score=3, confidence_reason="One bug.",
        files=[FileNote(path="a.py", summary="uses to_cents | rounding")],
        diagram="flowchart LR\n A-->B", findings=[],
    )  # fmt: skip
    posted = [_f("Bug", "b", severity=Severity.P1)]
    unplaced = [_f("Elsewhere", "far away", path="b.py", line=90)]
    out = summary_body(name="Acme Review", result=result, posted=posted, unplaced=unplaced, suppressed=2,
                       head_sha="abcdef1234", files_reviewed=1, include_diagram=True, review_count=3,
                       mention="@acme-review")  # fmt: skip
    assert out.startswith(SUMMARY_MARKER) and "## Acme Review" in out
    assert "**Confidence: 3/5** 🟡" in out
    assert "| `a.py` | uses to_cents \\| rounding | 🟠 P1 |" in out
    assert "`b.py:90` — Elsewhere" in out
    assert "```mermaid\nflowchart LR" in out
    assert "review #3 · 2 suppressed" in out and "`@acme-review`" in out
