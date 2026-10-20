from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from simpli_review.cli import decide_trigger, default_mention
from simpli_review.config import Config, load_config
from simpli_review.diff import parse_patch
from simpli_review.index import RepoIndex
from simpli_review.models import ReviewResult
from simpli_review.render import inline_body
from simpli_review.reviewer import (
    PullRequest,
    ReviewFailed,
    Settings,
    build_user_message,
    finalize,
    run_review,
    to_github_comments,
)
from simpli_review.tools import Toolbox
from tests.conftest import FakeClient, message, text, tool_use

PATCH = "@@ -4,3 +4,3 @@\n def charge(amount):\n-    cents = amount * 100\n+    cents = to_cents(amount)\n     return send(cents)"  # noqa: E501
PR = PullRequest(title="Use to_cents", body="Fix rounding", author="dale", base_ref="main", head_ref="fix",
                 head_sha="abc1234")  # fmt: skip


def _finding(**kw: Any) -> dict[str, Any]:
    base = {"path": "app/billing.py", "line": 5, "end_line": None, "severity": "P1", "type": "logic",
            "category": "correctness", "title": "Negative amounts round the wrong way", "body": "b",
            "evidence": "app/money.py:2", "suggestion": None, "confidence": 4, "rule_id": None}  # fmt: skip
    base.update(kw)
    return base


def _review(findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": "s",
        "confidence_score": 4,
        "confidence_reason": "r",
        "files": [{"path": "app/billing.py", "summary": "x"}],
        "diagram": None,
        "findings": findings,
    }


def _setup(repo: Path) -> tuple[list[dict[str, Any]], Toolbox]:
    patches = [parse_patch("app/billing.py", PATCH)]
    idx = RepoIndex(repo)
    user = build_user_message(
        pr=PR, patches=patches, index=idx, config=load_config(repo), repo_root=repo, feedback=None
    )
    return user, Toolbox(repo, idx)


def test_user_message_has_repo_context_graph_and_annotated_diff(repo: Path):
    user, _ = _setup(repo)
    repo_block, pr_block = user[0]["text"], user[1]["text"]
    assert "Context file: CLAUDE.md" in repo_block and "integer cents" in repo_block
    assert "### function charge (app/billing.py:4-6) - changed" in pr_block
    assert "called by: Checkout.pay" in pr_block and "calls: to_cents" in pr_block
    assert "app/billing.py is imported by: app/api.py" in pr_block
    assert "     5 +    cents = to_cents(amount)" in pr_block


def test_loop_runs_tools_then_returns_submitted_review(repo: Path):
    user, tb = _setup(repo)
    client = FakeClient([
        message(tool_use("t1", "read_file", {"path": "app/money.py"}),
                tool_use("t2", "find_callers", {"name": "to_cents"})),
        message(tool_use("t3", "submit_review", _review([_finding()]))),
    ])  # fmt: skip
    result, turns, usage = run_review(client, Settings(), user, tb)  # type: ignore[arg-type]
    assert turns == 2 and usage["input"] == 200 and usage["cache_read"] == 100
    assert result.findings[0].title == "Negative amounts round the wrong way"
    second = client.requests[1]
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]  # both results in ONE user message
    assert "def to_cents(dollars):" in results[0]["content"]
    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["output_config"] == {"effort": "high"} and first["cache_control"] == {"type": "ephemeral"}
    assert all(t["eager_input_streaming"] for t in first["tools"])


def test_bad_tool_input_and_invalid_review_are_returned_as_errors(repo: Path):
    user, tb = _setup(repo)
    client = FakeClient([
        message(tool_use("t1", "read_file", {"file": "x"}),
                tool_use("t2", "read_file", {"path": "../escape"})),
        message(tool_use("t3", "submit_review", {"summary": "missing everything"})),
        message(tool_use("t4", "submit_review", _review([]))),
    ])  # fmt: skip
    result, turns, _ = run_review(client, Settings(), user, tb)  # type: ignore[arg-type]
    assert turns == 3 and result.findings == []
    r1 = client.requests[1]["messages"][-1]["content"]
    assert all(r["is_error"] for r in r1)
    assert "missing ['path']" in r1[0]["content"] and "relative to the repository" in r1[1]["content"]
    r2 = client.requests[2]["messages"][-1]["content"]
    assert r2[0]["is_error"] and "did not match the schema" in r2[0]["content"]


def test_nudges_once_then_fails_without_submit(repo: Path):
    user, tb = _setup(repo)
    client = FakeClient(
        [message(text("Looks fine."), stop="end_turn"), message(text("Done."), stop="end_turn")]
    )
    with pytest.raises(ReviewFailed, match="without calling submit_review"):
        run_review(client, Settings(), user, tb)  # type: ignore[arg-type]
    assert client.requests[1]["messages"][-1]["content"].startswith("Call submit_review")


def test_refusal_and_turn_limit(repo: Path):
    user, tb = _setup(repo)
    with pytest.raises(ReviewFailed, match="declined"):
        run_review(FakeClient([message(stop="refusal")]), Settings(), user, tb)  # type: ignore[arg-type]
    looping = FakeClient([message(tool_use(f"t{i}", "list_directory", {"path": ""})) for i in range(3)])
    with pytest.raises(ReviewFailed, match="no result after 3 turns"):
        run_review(looping, Settings(max_turns=3), user, tb)  # type: ignore[arg-type]
    last = looping.requests[-1]["messages"][-1]["content"]
    assert last[-1]["type"] == "text" and "turns left" in last[-1]["text"]


def test_finalize_places_snaps_filters_dedupes_and_caps():
    patches = [parse_patch("app/billing.py", PATCH)]  # commentable: 4, 5, 6
    result = ReviewResult.model_validate(_review([
        _finding(line=5, suggestion="    cents = to_cents(abs(amount))"),
        _finding(line=8, title="Snapped from line eight", suggestion="x"),   # -> 6, suggestion dropped
        _finding(line=40, title="Far outside the diff"),                     # -> summary
        _finding(path="app/other.py", title="Not in this PR"),               # -> summary
        _finding(title="Low confidence guess", confidence=2),                # filtered
        _finding(title="Style nit", type="style", severity="P2"),            # filtered at strictness 2
        _finding(title="Already posted problem here"),                       # duplicate
    ]))  # fmt: skip
    existing = [{"path": "app/billing.py", "body": inline_body(result.findings[6])}]
    out = finalize(result, patches=patches, config=Config(), feedback=None, existing_comments=existing)
    assert [(f.title, f.line) for f in out.posted] == [
        ("Negative amounts round the wrong way", 5), ("Snapped from line eight", 6)]  # fmt: skip
    assert out.posted[0].suggestion and out.posted[1].suggestion is None
    assert {f.title for f in out.unplaced} == {"Far outside the diff", "Not in this PR"}
    assert (out.filtered, out.duplicates) == (2, 1)

    capped = finalize(
        result, patches=patches, config=Config(maxComments=1), feedback=None, existing_comments=[]
    )
    assert len(capped.posted) == 1 and "Snapped from line eight" in {f.title for f in capped.unplaced}


def test_multiline_comment_shape():
    patches = [parse_patch("app/billing.py", PATCH)]
    result = ReviewResult.model_validate(_review([_finding(line=4, end_line=6, suggestion="a\nb\nc")]))
    (f,) = finalize(result, patches=patches, config=Config(), feedback=None, existing_comments=[]).posted
    (c,) = to_github_comments([f], lambda _f: "body")
    assert c == {"path": "app/billing.py", "line": 6, "start_line": 4, "side": "RIGHT", "start_side": "RIGHT",
                 "body": "body"}  # fmt: skip


def _comment_event(body: str, assoc: str = "MEMBER", on_pr: bool = True) -> dict[str, Any]:
    issue: dict[str, Any] = {"number": 7, **({"pull_request": {}} if on_pr else {})}
    return {"issue": issue, "comment": {"body": body, "author_association": assoc}}


def test_triggers():
    cfg = Config()
    assert decide_trigger("pull_request", {"action": "opened", "pull_request": {"number": 3}}, cfg, "@r") == (
        3, False, "")  # fmt: skip
    assert decide_trigger("pull_request", {"action": "synchronize", "pull_request": {"number": 3}}, cfg,
                          "@r")[0] is None  # fmt: skip
    assert decide_trigger("pull_request", {"action": "synchronize", "pull_request": {"number": 3}},
                          Config(autoReview=["open", "push"]), "@r")[0] == 3  # fmt: skip
    assert decide_trigger("issue_comment", _comment_event("please @R review"), cfg, "@r") == (7, True, "")
    assert decide_trigger("issue_comment", _comment_event("@r", assoc="NONE"), cfg, "@r")[0] is None
    assert decide_trigger("issue_comment", _comment_event("@r", on_pr=False), cfg, "@r")[0] is None
    assert decide_trigger("issue_comment", _comment_event("thanks"), cfg, "@r")[0] is None


def test_default_mention_comes_from_the_configured_name():
    assert default_mention("Acme Code Review") == "@acme-code-review"
