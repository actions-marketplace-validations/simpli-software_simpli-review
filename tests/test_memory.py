from __future__ import annotations

from pathlib import Path
from typing import Any

from simpli_review.config import load_config
from simpli_review.diff import parse_patch
from simpli_review.index import RepoIndex
from simpli_review.memory import (
    MAX_PRIOR,
    PriorFinding,
    collect_prior_findings,
    compute_since_last,
    prompt_section,
    scrub,
    since_last_block,
    summary_findings,
)
from simpli_review.models import (
    Category,
    CommentType,
    Finding,
    PreviousStatus,
    ReviewResult,
    Severity,
)
from simpli_review.render import SUMMARY_MARKER, inline_body, summary_body
from simpli_review.reviewer import PullRequest, build_user_message
from tests.test_reviewer import PATCH


def _finding(title: str = "Rounds the wrong way", path: str = "a.py") -> Finding:
    return Finding(
        path=path, line=5, severity=Severity.P1, type=CommentType.logic, category=Category.correctness,
        title=title, body="b", evidence="e", confidence=4,
    )  # fmt: skip


def _comment(cid: int, title: str = "Rounds the wrong way", **kw: Any) -> dict[str, Any]:
    return {
        "id": cid, "path": "a.py", "line": 5, "created_at": f"2026-10-0{cid % 9 + 1}T00:00:00Z",
        "body": inline_body(_finding(title)), "reactions": {"+1": 0, "-1": 0}, **kw,
    }  # fmt: skip


def _reply(cid: int, parent: int, body: str, kind: str = "User") -> dict[str, Any]:
    return {
        "id": cid, "in_reply_to_id": parent, "body": body, "created_at": "2026-10-09T00:00:00Z",
        "user": {"type": kind},
    }  # fmt: skip


def test_collects_own_findings_with_state_and_replies():
    comments = [
        _comment(1, "First"),
        _comment(2, "Second", reactions={"+1": 0, "-1": 2}),
        _comment(3, "Third"),
        {"id": 4, "path": "a.py", "line": 1, "body": "a human review comment", "user": {"type": "User"}},
        _reply(10, 1, "Intended: we round half up here, see the ADR."),
        _reply(11, 1, "bot noise", kind="Bot"),
    ]
    prior = collect_prior_findings(comments, resolved_ids={3})
    by_title = {p.title: p for p in prior}
    assert set(by_title) == {"First", "Second", "Third"}  # the human comment and replies are not findings
    assert by_title["First"].replies == ["Intended: we round half up here, see the ADR."]
    assert by_title["Second"].downvoted and not by_title["Second"].resolved
    assert by_title["Third"].resolved
    assert [p.ref for p in prior] == [1, 2, 3]


def test_falls_back_to_original_line_for_outdated_comments():
    c = _comment(1)
    c["line"] = None
    c["original_line"] = 42
    assert collect_prior_findings([c], set())[0].line == 42


def test_cap_keeps_most_recent():
    comments = [_comment(i, f"Finding {i}") for i in range(1, 41)]
    for i, c in enumerate(comments):
        c["created_at"] = f"2026-10-01T00:{i:02d}:00Z"
    prior = collect_prior_findings(comments, set())
    assert len(prior) == MAX_PRIOR
    assert prior[0].title == "Finding 40" and prior[-1].title == "Finding 11"


def test_scrub_removes_emails_tokens_markers_and_truncates():
    text = "mail bob@example.com token ghp_abcdefghijklmnop1234 <!-- hidden --> " + "word " * 60
    out = scrub(text)
    assert "bob@example.com" not in out and "ghp_" not in out and "hidden" not in out
    assert "[email]" in out and "[redacted]" in out and len(out) <= 200 and out.endswith("…")


def test_reply_text_is_scrubbed_in_collection():
    comments = [_comment(1), _reply(2, 1, "ping me at dale@example.com")]
    assert collect_prior_findings(comments, set())[0].replies == ["ping me at [email]"]


def _summary_with_unplaced() -> str:
    result = ReviewResult(summary="s", confidence_score=3, confidence_reason="r", files=[], findings=[])
    far = _finding("Far away problem", path="b.py").model_copy(update={"line": 90})
    return summary_body(
        name="N", result=result, posted=[], unplaced=[far], suppressed=0, head_sha="abcdef1",
        files_reviewed=1, include_diagram=False, review_count=1, mention="@n",
    )  # fmt: skip


def test_summary_only_findings_are_recovered():
    assert summary_findings(_summary_with_unplaced()) == [("P1", "b.py", 90, "Far away problem")]
    prior = collect_prior_findings([_comment(1)], set(), _summary_with_unplaced())
    assert [(p.title, p.comment_id is None) for p in prior] == [
        ("Rounds the wrong way", False),
        ("Far away problem", True),
    ]
    assert summary_findings("no such section") == []


def test_prompt_section_lists_findings_state_and_replies():
    prior = [
        PriorFinding(1, "a.py", 5, "Rounds the wrong way", "P1", resolved=True, replies=["done, thanks"]),
        PriorFinding(2, "b.py", None, "Leaky handle", "P0"),
    ]
    text = prompt_section(prior)
    assert text.startswith("## Previous findings on this PR")
    assert "[1] P1 `a.py:5` - Rounds the wrong way (thread resolved)" in text
    assert 'reply: "done, thanks"' in text
    assert "[2] P0 `b.py` - Leaky handle (open)" in text
    assert "previous_findings" in text and "untrusted" in text
    assert prompt_section([]) == ""


def test_prompt_section_goes_into_pr_block_only_when_present(repo: Path):
    patches = [parse_patch("app/billing.py", PATCH)]
    pr = PullRequest("t", "b", "me", "main", "feat", "abc1234")
    kw: dict[str, Any] = {
        "pr": pr, "patches": patches, "index": RepoIndex(repo), "config": load_config(repo),
        "repo_root": repo, "feedback": None,
    }  # fmt: skip
    plain = build_user_message(**kw)
    assert "Previous findings" not in plain[0]["text"] + plain[1]["text"]
    with_memory = build_user_message(**kw, prior=[PriorFinding(1, "a.py", 5, "T", "P1")])
    assert "Previous findings" in with_memory[1]["text"]  # volatile block, so the repo block stays cacheable
    assert with_memory[0] == plain[0]


def _statuses(**by_ref: str) -> list[PreviousStatus]:
    return [PreviousStatus(ref=int(k[1:]), status=v) for k, v in by_ref.items()]  # type: ignore[arg-type]


def test_since_last_counts_and_block():
    prior = [
        PriorFinding(1, "a.py", 5, "One", "P1"),
        PriorFinding(2, "a.py", 9, "Two", "P2"),
        PriorFinding(3, "b.py", 1, "Three", "P1", resolved=True),
        PriorFinding(4, "c.py", 2, "Four", "P0"),  # model did not report it
    ]
    s = compute_since_last(prior, _statuses(r1="fixed", r2="still_present"), new_findings=2)
    assert s is not None
    assert (s.fixed, s.dismissed, [p.title for p in s.still_open], s.new) == (1, 1, ["Two", "Four"], 2)
    block = since_last_block(s)
    assert "### Since last review" in block and "- Fixed: 1" in block and "- Dismissed: 1" in block
    assert "- Still open (2):\n  - `a.py:9` Two\n  - `c.py:2` Four" in block
    assert "- New findings: 2" in block


def test_since_last_empty_and_none_open():
    assert compute_since_last([], [], 3) is None and since_last_block(None) == ""
    s = compute_since_last([PriorFinding(1, "a.py", 5, "One", "P1")], _statuses(r1="fixed"), 0)
    assert "- Still open: none" in since_last_block(s) and "- New findings: 0" in since_last_block(s)


def test_since_last_list_is_capped():
    prior = [PriorFinding(i, "a.py", i, f"T{i}", "P2") for i in range(1, 16)]
    block = since_last_block(compute_since_last(prior, [], 0))
    assert "Still open (15)" in block and "and 5 more" in block and "T11" not in block


def test_summary_body_includes_block_only_when_given():
    result = ReviewResult(summary="s", confidence_score=4, confidence_reason="r", files=[], findings=[])
    kw: dict[str, Any] = {
        "name": "N", "result": result, "posted": [], "unplaced": [], "suppressed": 0, "head_sha": "abcdef1",
        "files_reviewed": 1, "include_diagram": False, "review_count": 2, "mention": "@n",
    }  # fmt: skip
    assert "Since last review" not in summary_body(**kw)
    out = summary_body(**kw, since_last="### Since last review\n\n- Fixed: 1\n")
    assert out.startswith(SUMMARY_MARKER) and "### Since last review\n\n- Fixed: 1" in out


def test_resolved_comment_ids_reads_graphql_threads():
    import httpx

    from simpli_review.github import GitHub

    pages = [
        {"hasNextPage": True, "endCursor": "c1"},
        {"hasNextPage": False, "endCursor": None},
    ]
    nodes = [
        [{"isResolved": True, "comments": {"nodes": [{"databaseId": 11}, {"databaseId": 12}]}}],
        [{"isResolved": False, "comments": {"nodes": [{"databaseId": 21}]}}],
    ]
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        calls.append(json.loads(request.content))
        i = len(calls) - 1
        body = {
            "data": {
                "repository": {"pullRequest": {"reviewThreads": {"pageInfo": pages[i], "nodes": nodes[i]}}}
            }
        }
        return httpx.Response(200, json=body)

    client = httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(handler))
    gh = GitHub("t", "acme/app", client=client)
    assert gh.resolved_comment_ids(7) == {11, 12}
    assert calls[0]["variables"] == {"o": "acme", "n": "app", "p": 7, "c": None}
    assert calls[1]["variables"]["c"] == "c1"
