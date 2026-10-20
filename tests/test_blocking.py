from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import anthropic
import pytest

from simpli_review import cli
from simpli_review.blocking import evaluate, open_tasks
from simpli_review.config import Config
from simpli_review.github import GitHubError
from simpli_review.memory import PriorFinding
from simpli_review.models import (
    BlockDecision,
    Category,
    CheckRow,
    CommentType,
    FileNote,
    Finding,
    ReviewResult,
    Severity,
)
from simpli_review.render import block_notice, inline_body, summary_body
from tests.conftest import FakeClient, message, tool_use
from tests.test_reviewer import PATCH, _finding, _review


def _f(sev: str = "P1", title: str = "Refunds round the wrong way", path: str = "a.py") -> Finding:
    return Finding(
        path=path, line=5, severity=Severity(sev), type=CommentType.logic, category=Category.correctness,
        title=title, body="b", evidence="e", confidence=4,
    )  # fmt: skip


def _decide(
    block_on: str = "P1",
    new: list[Finding] | None = None,
    prior: list[PriorFinding] | None = None,
    checks: list[CheckRow] | None = None,
    labels: list[str] | None = None,
) -> BlockDecision:
    cfg = Config.model_validate({"blockOn": block_on})
    tasks = open_tasks(new or [], prior or [])
    return evaluate(cfg, tasks=tasks, checks=checks or [], labels=labels or [])


def test_nothing_blocks_by_default():
    assert not _decide("none", new=[_f("P0")]).blocking


def test_block_on_p0_and_p1_thresholds():
    assert _decide("P0", new=[_f("P0")]).blocking
    assert not _decide("P0", new=[_f("P1"), _f("P2")]).blocking
    assert _decide("P1", new=[_f("P1")]).blocking
    assert _decide("P1", new=[_f("P0")]).blocking
    assert not _decide("P1", new=[_f("P2")]).blocking
    d = _decide("P1", new=[_f("P0"), _f("P1", "Another problem entirely"), _f("P2", "Minor")])
    assert d.reasons == [
        "2 open finding(s) at or above P1 (1 x P0, 1 x P1), new or still open from earlier reviews"
    ]


def test_still_open_earlier_findings_count_and_repeats_are_not_counted_twice():
    prior = [PriorFinding(1, "a.py", 5, "Refunds round the wrong way", "P1"),
             PriorFinding(2, "b.py", 9, "Totally separate", "P0")]  # fmt: skip
    d = _decide("P1", new=[_f("P1")], prior=prior)
    assert d.blocking and d.reasons[0].startswith("2 open finding(s)")
    assert _decide("P1", prior=[PriorFinding(1, "b.py", None, "Old", "P2")]).blocking is False


def test_error_mode_failures_block_but_warnings_and_inconclusive_do_not():
    rows = [
        CheckRow("tests", "fail", "error", "No test for the new branch."),
        CheckRow("docs", "fail", "warning", "x"),
        CheckRow("scope", "inconclusive", "error", "x"),
        CheckRow("lint", "pass", "error", "x"),
    ]
    d = _decide("none", checks=rows)
    assert d.blocking and d.reasons == ["check `tests` failed (error mode): No test for the new branch."]
    assert not _decide("none", checks=rows[1:]).blocking


def test_override_label_skips_the_block_and_keeps_the_reasons():
    d = _decide("P1", new=[_f("P0")], labels=["review-override"])
    assert not d.blocking and d.overridden_by == "review-override" and d.reasons
    assert not _decide("none", labels=["review-override"]).reasons  # nothing to override
    cfg = Config.model_validate({"blockOn": "P1", "overrideLabel": "ship-it"})
    tasks = open_tasks([_f("P0")], [])
    assert evaluate(cfg, tasks=tasks, checks=[], labels=["review-override"]).blocking
    assert not evaluate(cfg, tasks=tasks, checks=[], labels=["ship-it"]).blocking


def test_notices_say_why():
    blocked = block_notice(
        BlockDecision(True, ["1 open finding(s) at or above P1 (1 x P1)"]), "re-review", "review-override"
    )
    assert blocked.startswith("### ⛔ Merge blocked")
    assert (
        "- 1 open finding(s) at or above P1" in blocked
        and "`re-review`" in blocked
        and "`review-override`" in blocked
    )
    skipped = block_notice(BlockDecision(False, ["check `x` failed"], "review-override"))
    assert (
        skipped.startswith("### Merge block overridden")
        and "`review-override` label is on this PR" in skipped
    )
    assert "check `x` failed" in skipped


# ---------- the whole GitHub run ----------


class _GH:
    calls: ClassVar[dict[str, Any]] = {}
    labels: ClassVar[list[str]] = []
    body: ClassVar[str] = "Fix rounding"
    review_comments: ClassVar[list[dict[str, Any]]] = []
    base_files: ClassVar[dict[str, str]] = {}  # the base commit's tree: the only trusted settings source
    changed_files: ClassVar[list[dict[str, Any]]] = []
    api_error: ClassVar[int | None] = None  # status code every contents-API call fails with
    client: ClassVar[FakeClient | None] = None  # the scripted Anthropic client of the last run

    def __init__(self, *_: Any) -> None:
        pass

    def pull(self, number: int) -> dict[str, Any]:
        return {
            "title": "Use to_cents", "body": _GH.body, "user": {"login": "dale"}, "draft": False,
            "base": {"ref": "main", "sha": BASE_SHA}, "head": {"ref": "fix", "sha": "abc1234"},
            "labels": [{"name": n} for n in _GH.labels],
        }  # fmt: skip

    def commit_exists(self, sha: str) -> None:
        if _GH.api_error:
            raise GitHubError(_GH.api_error, "boom")
        assert sha == BASE_SHA

    def file_at(self, path: str, ref: str) -> str | None:
        if _GH.api_error:
            raise GitHubError(_GH.api_error, "boom")
        assert ref == BASE_SHA
        return _GH.base_files.get(path)

    def pull_files(self, number: int) -> list[dict[str, Any]]:
        return _GH.changed_files or [{"filename": "app/billing.py", "patch": PATCH, "status": "modified"}]

    def recent_repo_review_comments(self) -> list[dict[str, Any]]:
        return []

    def pull_review_comments(self, number: int) -> list[dict[str, Any]]:
        return _GH.review_comments

    def resolved_comment_ids(self, number: int) -> set[int]:
        return set()

    def issue_comments(self, number: int) -> list[dict[str, Any]]:
        return []

    def set_status_reaction(self, number: int, content: str, me: str | None) -> None:
        _GH.calls.setdefault("reactions", []).append(content)

    def create_review(self, number: int, sha: str, body: str, comments: list[dict[str, Any]]) -> None:
        _GH.calls["inline"] = comments

    def upsert_issue_comment(self, number: int, marker: str, body: str) -> None:
        _GH.calls["summary"] = body

    def remove_label(self, number: int, label: str) -> None:
        pass


BASE_SHA = "b" * 40


def _run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    *,
    config: dict[str, Any] | None = None,
    base_files: dict[str, str] | None = None,
    head_files: dict[str, str] | None = None,
    changed_files: list[dict[str, Any]] | None = None,
    api_error: int | None = None,
    review: dict[str, Any] | None = None,
    responses: list[Any] | None = None,
    labels: list[str] | None = None,
    body: str = "Fix rounding",
    review_comments: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
    event_name: str = "pull_request",
    event_body: dict[str, Any] | None = None,
) -> int:
    _GH.calls, _GH.labels, _GH.body = {}, labels or [], body
    _GH.review_comments = review_comments or []
    # Settings come from the base commit; whatever is in the checked-out workspace is the PR's head.
    _GH.base_files = dict(base_files or {})
    _GH.changed_files, _GH.api_error = changed_files or [], api_error
    if config is not None:
        _GH.base_files[".simpli-review/config.json"] = json.dumps(config)
    for rel, text in (head_files or {}).items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    event = repo / "event.json"
    event.write_text(
        json.dumps(
            event_body or {"action": "opened", "pull_request": {"number": 7, "base": {"sha": BASE_SHA}}}
        ),
        encoding="utf-8",
    )
    env = {
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_REPOSITORY": "a/b",
        "GITHUB_WORKSPACE": str(repo),
        "GITHUB_TOKEN": "x",
        "INPUT_NAME": "Code Review",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    script = (
        responses
        if responses is not None
        else [message(tool_use("t1", "submit_review", review or _review([])))]
    )
    monkeypatch.setattr(cli, "GitHub", _GH)
    _GH.client = FakeClient(script)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: _GH.client)
    return cli._github_run(dry_run)


def test_a_blocking_finding_turns_the_check_red_after_everything_is_posted(monkeypatch, repo, capsys):
    code = _run(monkeypatch, repo, config={"blockOn": "P1"}, review=_review([_finding()]))
    assert code == 1
    assert _GH.calls["reactions"] == ["eyes", "+1"] and len(_GH.calls["inline"]) == 1
    summary = _GH.calls["summary"]
    assert "### ⛔ Merge blocked" in summary and "1 open finding(s) at or above P1" in summary
    assert "`review-override`" in summary
    assert "::error::Code Review is blocking the merge" in capsys.readouterr().out
    assert (
        "Prompt for AI agents (1 open)" in summary
        and "Prompt for AI agents</summary>" in _GH.calls["inline"][0]["body"]
    )


def test_override_label_passes_the_check_and_the_summary_says_who(monkeypatch, repo):
    code = _run(
        monkeypatch, repo, config={"blockOn": "P1"}, review=_review([_finding()]), labels=["review-override"]
    )
    assert code == 0
    assert (
        "### Merge block overridden" in _GH.calls["summary"]
        and "`review-override` label is on this PR" in _GH.calls["summary"]
    )


def test_default_config_never_blocks(monkeypatch, repo):
    assert _run(monkeypatch, repo, review=_review([_finding(severity="P0")])) == 0
    assert "Merge blocked" not in _GH.calls["summary"]


def test_dry_run_does_not_fail_the_job(monkeypatch, repo, capsys):
    assert _run(monkeypatch, repo, config={"blockOn": "P1"}, review=_review([_finding()]), dry_run=True) == 0
    assert "Merge blocked" in capsys.readouterr().out and "summary" not in _GH.calls


def test_reviewer_failure_fails_closed_when_blocking(monkeypatch, repo):
    refusal = [message(stop="refusal")]
    assert _run(monkeypatch, repo, config={"blockOn": "P0"}, responses=refusal) == 1
    assert "could not finish" in _GH.calls["summary"] and _GH.calls["reactions"] == ["eyes", "confused"]
    assert "Merging is blocked until a review completes" in _GH.calls["summary"]


def test_reviewer_failure_passes_without_blocking_or_with_override(monkeypatch, repo):
    refusal = [message(stop="refusal")]
    assert _run(monkeypatch, repo, responses=refusal) == 0
    assert "Merging is blocked" not in _GH.calls["summary"]
    labels = ["review-override"]
    assert _run(monkeypatch, repo, config={"blockOn": "P0"}, responses=refusal, labels=labels) == 0


def test_failed_error_check_blocks_and_warning_does_not(monkeypatch, repo):
    custom = {"name": "tests", "instructions": "New branches have tests.", "mode": "error"}
    verdicts = [
        {"name": "tests", "result": "fail", "reason": "No test for refund()."},
        {"name": "declared-scope", "result": "fail", "reason": "Undeclared env var."},
    ]
    review = {**_review([]), "checks": verdicts}
    assert _run(monkeypatch, repo, config={"checks": [custom]}, review=review) == 1
    summary = _GH.calls["summary"]
    assert "| `tests` | ❌ Fail | error | No test for refund(). |" in summary
    assert "| `declared-scope` | ❌ Fail | warning | Undeclared env var. |" in summary
    assert "check `tests` failed (error mode)" in summary and "declared-scope` failed" not in summary
    assert _run(monkeypatch, repo, config={"checks": [{**custom, "mode": "warning"}]}, review=review) == 0


def test_pr_template_check_uses_the_repo_template_and_the_pr_body(monkeypatch, repo):
    template = {".github/pull_request_template.md": "## Summary\n\n## Tests\n"}
    config = {"builtinChecks": {"pr-template": "error", "declared-scope": "off"}}
    assert _run(monkeypatch, repo, config=config, base_files=template, body="## Summary\nDone.") == 1
    assert "| `pr-template` | ❌ Fail | error | Missing: Tests. |" in _GH.calls["summary"]
    body = "## Summary\nDone.\n## Tests\nNone"
    assert _run(monkeypatch, repo, config=config, base_files=template, body=body) == 0
    assert "| `pr-template` | ✅ Pass | error |" in _GH.calls["summary"]


def test_an_earlier_finding_that_is_still_open_blocks_and_a_fixed_one_does_not(monkeypatch, repo):
    earlier = [{"id": 11, "path": "app/billing.py", "line": 5, "created_at": "2026-10-01T00:00:00Z",
                "body": inline_body(_f("P0", "Refunds round the wrong way", "app/billing.py")),
                "reactions": {}}]  # fmt: skip
    still = {**_review([]), "previous_findings": [{"ref": 1, "status": "still_present"}]}
    fixed = {**_review([]), "previous_findings": [{"ref": 1, "status": "fixed"}]}
    assert _run(monkeypatch, repo, config={"blockOn": "P1"}, review=still, review_comments=earlier) == 1
    assert "Prompt for AI agents (1 open)" in _GH.calls["summary"]
    assert _run(monkeypatch, repo, config={"blockOn": "P1"}, review=fixed, review_comments=earlier) == 0
    assert "Merge blocked" not in _GH.calls["summary"]


def test_filtered_findings_do_not_block(monkeypatch, repo):
    low = _finding(severity="P0", confidence=2)  # dropped by minConfidence
    assert _run(monkeypatch, repo, config={"blockOn": "P0"}, review=_review([low])) == 0


def test_summary_without_any_gate_matches_the_old_layout():
    result = ReviewResult(
        summary="s",
        confidence_score=4,
        confidence_reason="r",
        files=[FileNote(path="a", summary="x")],
        findings=[],
    )
    out = summary_body(name="N", result=result, posted=[], unplaced=[], suppressed=0, head_sha="abcdef1",
                       files_reviewed=1, include_diagram=False, review_count=1, mention="@n")  # fmt: skip
    assert "Pre-merge" not in out and "Prompt for AI agents" not in out and "Merge" not in out


# ---------- simpli-review local ----------


def _branch(repo: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *args], cwd=repo,
                       check=True, capture_output=True)  # fmt: skip

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "initial")
    git("checkout", "-q", "-b", "feat")
    billing = repo / "app" / "billing.py"
    old, new = "cents = to_cents(amount)", "cents = to_cents(abs(amount))"
    billing.write_text(billing.read_text().replace(old, new), encoding="utf-8")
    git("commit", "-q", "-am", "Use abs")
    return repo


def _local(monkeypatch: pytest.MonkeyPatch, repo: Path, review: dict[str, Any], *flags: str) -> int:
    monkeypatch.setattr(
        anthropic, "Anthropic", lambda: FakeClient([message(tool_use("t1", "submit_review", review))])
    )
    return cli.main(["local", "--base", "main", "--repo", str(repo), *flags])


def test_local_json_is_one_object_on_stdout_and_exits_nonzero_when_blocking(monkeypatch, repo, capsys):
    (repo / ".simpli-review").mkdir()
    (repo / ".simpli-review" / "config.json").write_text(json.dumps({"blockOn": "P1"}), encoding="utf-8")
    _branch(repo)  # the config is committed on main, the base of the review
    verdict = {"name": "declared-scope", "result": "fail", "reason": "Adds an env var."}
    review = {
        **_review([_finding(line=5, suggestion="    cents = to_cents(abs(amount))")]),
        "checks": [verdict],
    }
    code = _local(monkeypatch, repo, review, "--json")
    out, err = capsys.readouterr()
    data = json.loads(out)  # raises unless stdout is exactly one JSON object
    assert code == 1 and data["blocking"] is True and data["reasons"][0].startswith("1 open finding(s)")
    assert data["summary"] == "s" and data["confidence_score"] == 4
    (f,) = data["findings"]
    assert (f["severity"], f["path"], f["line"], f["title"]) == (
        "P1",
        "app/billing.py",
        5,
        "Negative amounts round the wrong way",
    )
    assert f["body"] == "b" and f["placement"] == "inline"
    assert (
        f["agent_prompt"].startswith("In app/billing.py around line 5: [P1]")
        and "Then verify" in f["agent_prompt"]
    )
    assert data["checks"] == [
        {"name": "declared-scope", "result": "fail", "mode": "warning", "reason": "Adds an env var."}
    ]
    assert "turn 1" in err and "turn 1" not in out


def test_local_json_exits_zero_when_clear(monkeypatch, repo, capsys):
    _branch(repo)
    assert (
        _local(monkeypatch, repo, _review([_finding(severity="P0")]), "--json") == 0
    )  # blockOn defaults to none
    assert json.loads(capsys.readouterr().out)["blocking"] is False


def test_local_without_json_keeps_the_human_output(monkeypatch, repo, capsys):
    _branch(repo)
    assert _local(monkeypatch, repo, _review([_finding()])) == 0
    out = capsys.readouterr().out
    assert (
        out.startswith("turn 1\n<!-- simpli-review:summary -->\n## Code review")
        and "--- app/billing.py:5" in out
    )
    assert "Pre-merge checks" in out and not out.lstrip().startswith("{")


def test_local_human_mode_also_fails_when_blocking(monkeypatch, repo, capsys):
    (repo / ".simpli-review").mkdir()
    (repo / ".simpli-review" / "config.json").write_text(json.dumps({"blockOn": "P1"}), encoding="utf-8")
    _branch(repo)
    assert _local(monkeypatch, repo, _review([_finding()])) == 1
    assert "### ⛔ Merge blocked" in capsys.readouterr().out


def test_local_json_reports_errors_and_empty_diffs_as_json(monkeypatch, repo, capsys):
    assert cli.main(["local", "--base", "nope", "--repo", str(repo), "--json"]) == 2  # not a git repo
    out, err = capsys.readouterr()
    assert "Cannot find a merge base" in json.loads(out)["error"] and "Cannot find a merge base" in err
    _branch(repo)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True, capture_output=True)
    assert cli.main(["local", "--base", "main", "--repo", str(repo), "--json"]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert (
        empty["findings"] == [] and empty["blocking"] is False and empty["summary"] == "No changes to review."
    )
