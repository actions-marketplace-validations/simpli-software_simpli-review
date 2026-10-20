from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from simpli_review import cli
from simpli_review.cli import decide_trigger, is_re_review_label
from simpli_review.config import Config
from simpli_review.github import GitHub, GitHubError
from simpli_review.models import ReviewResult
from simpli_review.render import summary_body


def _labeled(label: str | None, number: int = 5) -> dict[str, Any]:
    event: dict[str, Any] = {"action": "labeled", "pull_request": {"number": number}}
    if label is not None:
        event["label"] = {"name": label}
    return event


def test_re_review_label_is_a_manual_review():
    assert decide_trigger("pull_request", _labeled("re-review"), Config(), "@r") == (5, True, "")
    assert decide_trigger("pull_request_target", _labeled("re-review"), Config(), "@r") == (5, True, "")


def test_other_labels_are_not_handled():
    number, manual, why = decide_trigger("pull_request", _labeled("bug"), Config(), "@r")
    assert number is None and not manual and "re-review" in why
    assert decide_trigger("pull_request", _labeled(None), Config(), "@r")[0] is None
    # Case matters: GitHub label names are matched exactly.
    assert decide_trigger("pull_request", _labeled("Re-Review"), Config(), "@r")[0] is None


def test_label_name_is_configurable():
    cfg = Config(reReviewLabel="please-review")
    assert decide_trigger("pull_request", _labeled("please-review"), cfg, "@r")[0] == 5
    assert decide_trigger("pull_request", _labeled("re-review"), cfg, "@r")[0] is None


def test_labeled_ignores_auto_review_setting():
    # `labeled` must not depend on autoReview ("open"/"push"): it is a manual request.
    assert decide_trigger("pull_request", _labeled("re-review"), Config(autoReview=[]), "@r")[0] == 5


def test_other_pull_request_actions_still_follow_auto_review():
    opened = {"action": "opened", "pull_request": {"number": 3}}
    assert decide_trigger("pull_request", opened, Config(), "@r") == (3, False, "")
    assert decide_trigger("pull_request", {**opened, "action": "unlabeled"}, Config(), "@r")[0] is None
    assert decide_trigger("pull_request", {**opened, "action": "synchronize"}, Config(), "@r")[0] is None


def test_comment_trigger_still_works_for_old_workflows():
    event = {
        "issue": {"number": 7, "pull_request": {}},
        "comment": {"body": "@r please", "author_association": "MEMBER"},
    }
    assert decide_trigger("issue_comment", event, Config(), "@r") == (7, True, "")


def test_is_re_review_label_only_for_labeled_pull_request_events():
    assert is_re_review_label("pull_request", _labeled("re-review"), Config())
    assert not is_re_review_label("pull_request", {**_labeled("re-review"), "action": "opened"}, Config())
    assert not is_re_review_label("issue_comment", _labeled("re-review"), Config())
    assert not is_re_review_label("pull_request", _labeled("bug"), Config())


def _gh(handler: Any) -> GitHub:
    client = httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(handler))
    return GitHub("t", "acme/app", client=client)


def test_remove_label_deletes_and_url_quotes_the_name():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.raw_path.decode()))
        return httpx.Response(200, json=[])

    _gh(handler).remove_label(5, "needs review/now")
    assert seen == [("DELETE", "/repos/acme/app/issues/5/labels/needs%20review%2Fnow")]


def test_remove_label_ignores_missing_label_but_raises_other_errors():
    _gh(lambda r: httpx.Response(404, json={"message": "Label does not exist"})).remove_label(5, "re-review")
    with pytest.raises(GitHubError):
        _gh(lambda r: httpx.Response(403, json={"message": "forbidden"})).remove_label(5, "re-review")


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, event: dict[str, Any], name: str = "pull_request"):
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", name)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/app")
    monkeypatch.setenv("GITHUB_TOKEN", "x")


class _Recorder:
    removed: ClassVar[list[tuple[int, str]]] = []

    def __init__(self, *_: Any) -> None:
        pass

    def remove_label(self, number: int, label: str) -> None:
        _Recorder.removed.append((number, label))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    _Recorder.removed = []
    monkeypatch.setattr(cli, "GitHub", _Recorder)
    return _Recorder


def test_label_is_removed_after_a_normal_run(monkeypatch, tmp_path, recorder):
    _env(monkeypatch, tmp_path, _labeled("re-review", 9))
    monkeypatch.setattr(cli, "_github_run", lambda dry_run, loaded=None: 0)
    assert cli.github_main(dry_run=False) == 0
    assert recorder.removed == [(9, "re-review")]


def test_label_is_removed_even_when_the_run_raises(monkeypatch, tmp_path, recorder):
    _env(monkeypatch, tmp_path, _labeled("re-review", 9))

    def boom(dry_run: bool, loaded: Any = None) -> int:
        raise RuntimeError("review blew up")

    monkeypatch.setattr(cli, "_github_run", boom)
    with pytest.raises(RuntimeError):
        cli.github_main(dry_run=False)
    assert recorder.removed == [(9, "re-review")]


def test_label_is_kept_on_dry_run_and_for_other_events(monkeypatch, tmp_path, recorder):
    _env(monkeypatch, tmp_path, _labeled("re-review", 9))
    monkeypatch.setattr(cli, "_github_run", lambda dry_run, loaded=None: 0)
    cli.github_main(dry_run=True)
    _env(monkeypatch, tmp_path, _labeled("bug", 9))
    cli.github_main(dry_run=False)
    _env(monkeypatch, tmp_path, {"action": "opened", "pull_request": {"number": 9}})
    cli.github_main(dry_run=False)
    assert recorder.removed == []


def test_label_removal_failure_does_not_fail_the_run(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, _labeled("re-review", 9))

    class Failing(_Recorder):
        def remove_label(self, number: int, label: str) -> None:
            raise GitHubError(500, "boom")

    monkeypatch.setattr(cli, "GitHub", Failing)
    monkeypatch.setattr(cli, "_github_run", lambda dry_run, loaded=None: 0)
    assert cli.github_main(dry_run=False) == 0


def test_summary_footer_names_the_label_when_given():
    result = ReviewResult(summary="s", confidence_score=4, confidence_reason="r", files=[], findings=[])
    kw: dict[str, Any] = {
        "name": "N", "result": result, "posted": [], "unplaced": [], "suppressed": 0, "head_sha": "abcdef1",
        "files_reviewed": 1, "include_diagram": False, "review_count": 2, "mention": "@n",
    }  # fmt: skip
    assert "add the `re-review` label to re-run" in summary_body(**kw, re_review_label="re-review")
    assert "comment `@n` to re-run" in summary_body(**kw)
