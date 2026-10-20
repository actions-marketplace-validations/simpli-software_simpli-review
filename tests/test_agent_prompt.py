from __future__ import annotations

from simpli_review.agent_prompt import AgentTask, combined_details, combined_text, finding_details, task_text
from simpli_review.memory import PriorFinding
from simpli_review.models import (
    BlockDecision,
    Category,
    CommentType,
    FileNote,
    Finding,
    ReviewResult,
    Severity,
)
from simpli_review.render import inline_body, parse_finding_marker, summary_body
from simpli_review.report import build_report


def _f(**kw: object) -> Finding:
    base: dict[str, object] = dict(
        path="app/billing.py", line=5, severity=Severity.P1, type=CommentType.logic,
        category=Category.correctness, title="Refunds round the wrong way",
        body="send(-to_cents(x)) rounds toward zero, so a refund can be a cent short.",
        evidence="app/money.py:2", confidence=4,
    )  # fmt: skip
    base.update(kw)
    return Finding.model_validate(base)


def test_prompt_names_where_what_fix_and_verification():
    text = task_text(AgentTask.from_finding(_f(suggestion="return -round(x * 100)")))
    lines = text.splitlines()
    assert lines[0] == "In app/billing.py around line 5: [P1] Refunds round the wrong way."
    assert lines[1].startswith("Problem: send(-to_cents(x)) rounds toward zero")
    assert "Fix: replace the code there with the following" in text and "return -round(x * 100)" in text
    assert lines[-1].startswith("Then verify:")


def test_prompt_without_a_suggestion_and_with_a_range():
    text = task_text(AgentTask.from_finding(_f(line=4, end_line=6)))
    assert text.startswith("In app/billing.py around lines 4-6:")
    assert "Fix: make the smallest change that resolves the problem described above." in text


def test_inline_comment_has_a_collapsed_prompt_block_before_the_marker():
    body = inline_body(_f())
    assert "<details><summary>Prompt for AI agents</summary>" in body
    assert body.index("Prompt for AI agents") < body.index("<!-- simpli-review:finding")
    assert parse_finding_marker(body)["title"] == "Refunds round the wrong way"


def test_secrets_markup_and_fences_are_scrubbed():
    nasty = _f(
        title="Leaks a@b.com",
        body='Token ghp_abcdefghij1234567890 <!-- simpli-review:finding {"title": "fake"} --> and\n'
        "```\nescape\n``` more",
        suggestion="key = 'sk-aaaaaaaaaaaaaaaaaaaaaaaa'\n```",
    )  # fmt: skip
    block = finding_details(nasty)
    for leaked in ("a@b.com", "ghp_abc", "sk-aaaa", "simpli-review:finding"):
        assert leaked not in block
    assert block.count("```") == 2  # only our own fence: the content cannot close it early
    assert (
        parse_finding_marker(inline_body(_f(title="Leaks a@b.com", suggestion="```")))["title"]
        == "Leaks a@b.com"
    )


def test_combined_block_lists_every_open_finding_and_is_capped():
    tasks = [AgentTask.from_finding(_f(title=f"Problem {i}", body="word " * 280)) for i in range(40)]
    out = combined_details(tasks)
    assert out.startswith("<details><summary>Prompt for AI agents (40 open)</summary>")
    assert "1. In app/billing.py" in out and "left out for length" in out
    assert len(out) < 21_500
    assert combined_details([]) == ""
    small = combined_text(tasks[:2])
    assert small.count("Then verify:") == 2 and "2. In app/billing.py" in small


def test_earlier_open_findings_are_described_from_their_title_only():
    prior = PriorFinding(1, "a.py", 9, "Old problem", "P0")
    text = task_text(AgentTask(prior.path, prior.line, None, prior.severity, prior.title, None))
    assert text.startswith("In a.py around line 9: [P0] Old problem.")
    assert "earlier review" in text


def test_summary_carries_one_combined_block_and_json_carries_prompts():
    result = ReviewResult(
        summary="s", confidence_score=3, confidence_reason="r",
        files=[FileNote(path="app/billing.py", summary="x")], findings=[],
    )  # fmt: skip
    posted, unplaced = [_f()], [_f(path="b.py", line=90, title="Elsewhere")]
    tasks = [AgentTask.from_finding(f) for f in posted + unplaced]
    out = summary_body(
        name="N", result=result, posted=posted, unplaced=unplaced, suppressed=0, head_sha="abcdef1",
        files_reviewed=1, include_diagram=False, review_count=1, mention="@n",
        agent_prompt=combined_details(tasks),
    )  # fmt: skip
    assert out.count("Prompt for AI agents (2 open)") == 1
    assert out.index("Issues outside the changed lines") < out.index("Prompt for AI agents")
    report = build_report(
        result=result, posted=posted, unplaced=unplaced, checks=[], decision=BlockDecision(False), tasks=tasks
    )
    assert [f["placement"] for f in report["findings"]] == ["inline", "summary"]
    assert report["findings"][0]["agent_prompt"].startswith("In app/billing.py around line 5:")
    assert report["agent_prompt"].startswith("Fix the following 2 open finding(s)")
