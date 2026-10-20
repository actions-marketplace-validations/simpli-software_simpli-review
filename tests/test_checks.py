from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from simpli_review.checks import build_rows, pr_template_row, prompt_section, read_pr_template
from simpli_review.config import Config, load_config
from simpli_review.models import ModelCheck
from simpli_review.reviewer import Settings, build_user_message, run_review
from simpli_review.tools import TOOLS
from tests.conftest import FakeClient, message, tool_use
from tests.test_reviewer import PR, _finding, _review, _setup

TEMPLATE = """\
## Summary
<!-- Describe what this PR does. ## Not a real heading -->

## Why
Explain why.

## Tests
- [ ] Added tests

## Migrations
None
"""

FILLED = """\
## Summary
Moves billing to integer cents.

## Why
Rounding bugs on refunds.

## Tests
- [x] Added tests

## Migrations
None
"""


def _row(body: str, template: str = TEMPLATE) -> tuple[str, str]:
    row = pr_template_row(template, body, "warning")
    return row.result, row.reason


def test_pr_template_passes_when_every_section_is_filled():
    result, reason = _row(FILLED)
    assert result == "pass" and "All 4" in reason


def test_pr_template_names_missing_and_empty_sections():
    body = "## Summary\n<!-- still the comment -->\n\n## Why\nExplain why.\n\n## Tests\n- [ ] Added tests\n"
    result, reason = _row(body)
    assert result == "fail"
    assert "Missing: Migrations" in reason
    assert "Summary" in reason and "Why" in reason and "Tests" in reason  # empty / still the template


@pytest.mark.parametrize("filler", ["TODO", "tbd", "- TODO", "TBD.", "", "-", "- [ ]"])
def test_todo_and_blank_content_is_empty(filler: str):
    body = FILLED.replace("Moves billing to integer cents.", filler)
    result, reason = _row(body)
    assert result == "fail" and "Summary" in reason


def test_none_counts_as_filled_even_when_the_template_shows_it():
    result, _ = _row(FILLED.replace("Rounding bugs on refunds.", "None"))
    assert result == "pass"
    result, _ = _row(FILLED.replace("Rounding bugs on refunds.", "n/a"))
    assert result == "pass"


def test_unchanged_checklist_and_edited_checklist():
    unticked = FILLED.replace("- [x] Added tests", "- [ ] Added tests")
    assert _row(unticked)[0] == "fail"
    assert (
        _row(FILLED.replace("- [x] Added tests", "- [ ] Added tests\n- [x] Also checked staging"))[0]
        == "pass"
    )


def test_headings_match_loosely_but_must_exist():
    body = FILLED.replace("## Why", "### **why**:")
    assert _row(body)[0] == "pass"
    assert _row("")[1].startswith("Missing: Summary, Why, Tests, Migrations")


def test_headings_in_comments_and_code_fences_are_not_required():
    template = "## One\n```\n## Not a heading\n```\n\n<!--\n## Hidden\n-->\n"
    assert _row("## One\nDone.", template)[0] == "pass"


def test_template_without_headings_passes():
    assert _row("anything", "Please describe your change.\n")[0] == "pass"


def test_nested_subheadings_count_as_content():
    template = "## Details\n"
    assert _row("## Details\n### Sub\ntext", template)[0] == "pass"
    assert _row("## Details\n\n## Other\nx", template)[0] == "fail"


def test_template_is_read_from_the_reviewed_repo(tmp_path: Path):
    assert read_pr_template(tmp_path) is None
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "pull_request_template.md").write_text(TEMPLATE, encoding="utf-8")
    assert read_pr_template(tmp_path) == TEMPLATE


# ---------- rows ----------


def _cfg(**kw: Any) -> Config:
    return Config.model_validate(kw)


def test_rows_for_default_config_without_a_template():
    rows = build_rows(Config(), [], None, "body")
    assert [(r.name, r.result, r.mode) for r in rows] == [("declared-scope", "inconclusive", "warning")]
    assert "did not report" in rows[0].reason


def test_rows_combine_builtin_template_and_model_verdicts():
    cfg = _cfg(checks=[{"name": "no-secrets", "instructions": "No secrets in the diff.", "mode": "error"}])
    reported = [
        ModelCheck(name="declared-scope", result="fail", reason="Adds a migration the description omits."),
        ModelCheck(name="NO-SECRETS", result="pass", reason="Nothing found."),
        ModelCheck(name="invented", result="fail", reason="ignored"),
        ModelCheck(name="no-secrets", result="fail", reason="second report is ignored"),
    ]
    rows = build_rows(cfg, reported, TEMPLATE, "## Summary\nx")
    assert [(r.name, r.result, r.mode) for r in rows] == [
        ("pr-template", "fail", "warning"),
        ("declared-scope", "fail", "warning"),
        ("no-secrets", "pass", "error"),
    ]


def test_modes_off_leave_the_row_out_and_local_reviews_skip_the_template():
    cfg = _cfg(builtinChecks={"pr-template": "off", "declared-scope": "off"})
    assert build_rows(cfg, [], TEMPLATE, "") == []
    rows = build_rows(Config(), [], TEMPLATE, None)  # local: no PR description
    assert [r.name for r in rows] == ["declared-scope"]
    off = _cfg(checks=[{"name": "zz-hidden", "instructions": "y", "mode": "off"}])
    assert "zz-hidden" not in prompt_section(off)


def test_model_reason_is_one_scrubbed_line():
    reported = [
        ModelCheck(name="declared-scope", result="fail", reason="mail me a@b.com\nkey ghp_abcdefghij12345")
    ]
    (row,) = build_rows(Config(), reported, None, "")
    assert "\n" not in row.reason and "a@b.com" not in row.reason and "ghp_" not in row.reason


# ---------- config ----------


def test_defaults_are_safe(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert cfg.checks == [] and cfg.block_on == "none" and cfg.override_label == "review-override"
    assert cfg.builtin_checks.pr_template == "warning" and cfg.builtin_checks.declared_scope == "warning"


def test_config_reads_checks_and_builtin_modes(tmp_path: Path):
    folder = tmp_path / ".simpli-review"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "builtinChecks": {"pr-template": "error", "declared-scope": "off"},
        "checks": [{"name": "tests", "instructions": "Changed code has tests.", "mode": "error"},
                   {"name": "docs", "instructions": "Docs updated."}],
        "blockOn": "P1", "overrideLabel": "skip-review",
    }), encoding="utf-8")  # fmt: skip
    cfg = load_config(tmp_path)
    assert (cfg.builtin_checks.pr_template, cfg.builtin_checks.declared_scope) == ("error", "off")
    assert [(c.name, c.mode) for c in cfg.checks] == [("tests", "error"), ("docs", "warning")]
    assert cfg.block_on == "P1" and cfg.override_label == "skip-review"


@pytest.mark.parametrize(
    "bad",
    [
        {"checks": [{"name": "x" * 51, "instructions": "ok"}]},
        {"checks": [{"name": "ok", "instructions": "y" * 1001}]},
        {"checks": [{"name": "ok", "instructions": "y", "mode": "fatal"}]},
        {"checks": [{"name": "a", "instructions": "y"}, {"name": "A", "instructions": "z"}]},
        {"checks": [{"name": "pr-template", "instructions": "y"}]},
        {"builtinChecks": {"pr-template": "blocking"}},
        {"blockOn": "P2"},
    ],
)
def test_invalid_check_config_is_rejected(bad: dict[str, Any]):
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_limits_are_inclusive():
    Config.model_validate({"checks": [{"name": "x" * 50, "instructions": "y" * 1000}]})


# ---------- the prompt and the submit_review schema ----------


def test_checks_are_listed_in_the_cacheable_repo_block(repo: Path):
    cfg = _cfg(checks=[{"name": "no-eval", "instructions": "No   eval()\nanywhere.", "mode": "error"}])
    user = build_user_message(pr=PR, patches=[], index=_setup(repo)[1].index, config=cfg, repo_root=repo,
                              feedback=None)  # fmt: skip
    block = user[0]["text"]
    assert "## Pre-merge checks to evaluate" in block
    assert "- `declared-scope`: Compare the PR description with the diff." in block
    assert "- `no-eval`: No eval() anywhere." in block
    assert "Pre-merge checks" not in user[1]["text"]
    quiet = _cfg(builtinChecks={"declared-scope": "off"})
    off = build_user_message(pr=PR, patches=[], index=_setup(repo)[1].index, config=quiet, repo_root=repo,
                             feedback=None)  # fmt: skip
    assert "Pre-merge checks" not in off[0]["text"]


def test_submit_review_schema_has_a_checks_array():
    schema = next(t for t in TOOLS if t["name"] == "submit_review")["input_schema"]
    assert schema["properties"]["checks"]["type"] == "array"
    item = schema["$defs"]["ModelCheck"]
    assert set(item["properties"]) == {"name", "result", "reason"}
    assert item["properties"]["result"]["enum"] == ["pass", "fail", "inconclusive"]
    assert item["properties"]["name"]["maxLength"] == 50
    assert "checks" not in schema["required"]  # optional: a missing check shows as inconclusive


def test_submitted_checks_are_validated_like_the_rest_of_the_schema(repo: Path):
    user, tb = _setup(repo)
    good = {
        **_review([]),
        "checks": [{"name": "declared-scope", "result": "pass", "reason": "Nothing extra."}],
    }
    bad_result = {**_review([]), "checks": [{"name": "declared-scope", "result": "maybe", "reason": "?"}]}
    bad_name = {**_review([]), "checks": [{"name": "n" * 51, "result": "pass", "reason": "?"}]}
    missing = {**_review([]), "checks": [{"name": "declared-scope", "result": "pass"}]}
    client = FakeClient([
        message(tool_use("t1", "submit_review", bad_result)),
        message(tool_use("t2", "submit_review", bad_name)),
        message(tool_use("t3", "submit_review", missing)),
        message(tool_use("t4", "submit_review", good)),
    ])  # fmt: skip
    result, turns, _ = run_review(client, Settings(), user, tb)  # type: ignore[arg-type]
    assert turns == 4 and [c.result for c in result.checks] == ["pass"]
    for i in (1, 2, 3):
        (r,) = client.requests[i]["messages"][-1]["content"]
        assert r["is_error"] and "did not match the schema" in r["content"]


def test_a_review_without_checks_still_validates(repo: Path):
    user, tb = _setup(repo)
    result, _, _ = run_review(
        FakeClient([message(tool_use("t1", "submit_review", _review([_finding()])))]),  # type: ignore[arg-type]
        Settings(), user, tb,
    )  # fmt: skip
    assert result.checks == []
