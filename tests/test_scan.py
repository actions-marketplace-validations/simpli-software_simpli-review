"""Scan mode: budget, cost, fingerprints and de-duplication, the issue cap, the report files, scrubbing.

All with the scripted fake Claude client and a fake GitHub: no API calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from simpli_review.config import Config, ScanConfig, load_config
from simpli_review.github import GitHub, GitHubError
from simpli_review.index import RepoIndex
from simpli_review.models import ScanFinding, ScanResult
from simpli_review.pricing import PRICES, estimate_micros, format_usd
from simpli_review.reviewer import LoopState, ReviewFailed, Settings, agent_loop
from simpli_review.scan import (
    SCAN_TOOLS,
    ScanConfigError,
    build_scan_message,
    finalize_scan,
    scan_main,
    scan_settings,
)
from simpli_review.scan_issues import (
    LABEL,
    LABEL_DESCRIPTION,
    fingerprint,
    issue_body,
    issue_title,
    known_issues,
    marker,
    match_known,
    parse_marker,
    plan_issues,
    publish,
)
from simpli_review.scan_map import repo_map
from simpli_review.scrub import redact_secrets
from simpli_review.tools import Toolbox
from tests.conftest import FakeClient, message, text, tool_use

SHA = "a" * 40
ROOT = Path(__file__).resolve().parent.parent
# Obviously fake values, shaped like the secrets the scrubber must remove.
FAKE_API_KEY = "sk-fakefakefakefake12345678"
FAKE_WEBHOOK_SECRET = "whsec_fakefakefake1234567890"
FAKE_PASSWORD = "hunter2hunter2"


def finding(**kw: Any) -> dict[str, Any]:
    base = {
        "path": "app/billing.py", "line": 5, "end_line": None, "severity": "P1", "category": "security",
        "title": "Charge amount is not validated", "body": "A negative amount refunds the customer.",
        "evidence": "app/billing.py:5 passes amount through; app/money.py:2 never checks the sign",
        "confidence": 4, "rule_id": None,
    }  # fmt: skip
    base.update(kw)
    return base


def scan_result(*findings: dict[str, Any]) -> dict[str, Any]:
    return {"summary": "Covered billing.", "not_covered": ["web/"], "findings": list(findings)}


def sf(**kw: Any) -> ScanFinding:
    return ScanFinding.model_validate(finding(**kw))


def ls(i: int) -> Any:
    return tool_use(f"t{i}", "list_directory", {"path": ""})


# ---------- cost arithmetic ----------


def test_price_table_and_cost_arithmetic():
    # Opus 5.5: $4 in, $20 out, cache read $0.20, cache write 1.25x = $5 (per million tokens).
    assert PRICES["claude-opus-5-5"].cache_write == 500 and PRICES["claude-opus-5-5"].cache_read == 20
    usage = {"input": 1_000_000, "output": 100_000, "cache_read": 2_000_000, "cache_write": 500_000}
    assert estimate_micros("claude-opus-5-5", usage) == 4_000_000 + 2_000_000 + 400_000 + 2_500_000
    assert format_usd(estimate_micros("claude-opus-5-5", usage)) == "$8.90"
    # Sonnet 5.5: $2 in, $10 out, cache read $0.20, cache write $2.50.
    assert estimate_micros("claude-sonnet-5-5", usage) == 2_000_000 + 1_000_000 + 400_000 + 1_250_000
    # One output token on Opus is $0.00002, i.e. 20 millionths of a dollar.
    assert estimate_micros("claude-opus-5-5", {"output": 1}) == 20


def test_cost_rounding_and_unknown_model():
    assert estimate_micros("claude-made-up", {"input": 10}) is None
    assert "unknown" in format_usd(None)
    assert format_usd(4_999) == "$0.00" and format_usd(5_000) == "$0.01"
    assert format_usd(12_345_000) == "$12.35"  # whole cents, half up
    assert estimate_micros("claude-opus-5-5", {}) == 0


# ---------- budget ----------


def _run(repo: Path, responses: list[Any], budget: int | None, max_turns: int = 10):
    client = FakeClient(responses)
    state = LoopState()
    settings = Settings(max_turns=max_turns, budget_tokens=budget)
    error: ReviewFailed | None = None
    result: ScanResult | None = None
    try:
        result = agent_loop(
            client, settings, system="s", tools=SCAN_TOOLS, submit_name="submit_scan",  # type: ignore[arg-type]
            parse=ScanResult.model_validate, user_content=[{"type": "text", "text": "go"}],
            toolbox=Toolbox(repo, RepoIndex(repo)), state=state,
        )  # fmt: skip
    except ReviewFailed as e:
        error = e
    return client, state, result, error


def test_budget_nudges_then_accepts_a_submit(repo: Path):
    tok = (300, 0, 0, 0)
    client, state, result, error = _run(
        repo,
        [
            message(ls(1), tokens=tok),
            message(ls(2), tokens=tok),
            message(tool_use("t3", "submit_scan", scan_result()), tokens=tok),
        ],
        budget=1000,
    )
    assert error is None and result is not None and result.summary == "Covered billing."
    # After turn 2: 600 used, and two more turns of about 300 would pass 1000, so it was told to submit.
    assert state.budget_nudged_at == 2 and state.turns == 3 and state.total_tokens == 900
    assert all(
        b["type"] != "text" for b in client.requests[1]["messages"][-1]["content"]
    )  # none after turn 1
    last = client.requests[2]["messages"][-1]["content"]
    assert last[-1]["type"] == "text" and "call submit_scan now" in last[-1]["text"]
    assert "600 of 1,000" in last[-1]["text"]


def test_budget_stops_after_the_nudged_turn(repo: Path):
    tok = (300, 0, 0, 0)
    client, state, _, error = _run(repo, [message(ls(i), tokens=tok) for i in range(1, 6)], budget=1000)
    assert error is not None and "token budget (1,000) was reached" in str(error)
    assert len(client.requests) == 3  # turn 3 was the one allowed after the nudge; there was no turn 4
    assert state.usage["input"] == 900  # the spend survives the failure


def test_budget_counts_cache_and_output_tokens_not_just_input(repo: Path):
    each = (10, 90, 150, 50)  # 300 in all, but only 10 of them are plain input
    submit = message(tool_use("t3", "submit_scan", scan_result()), tokens=each)
    _, state, result, error = _run(
        repo, [message(ls(1), tokens=each), message(ls(2), tokens=each), submit], budget=1000
    )
    assert error is None and result is not None
    assert state.budget_nudged_at == 2 and state.total_tokens == 900


def test_no_budget_means_no_nudge(repo: Path):
    _, state, result, error = _run(
        repo,
        [message(ls(1), tokens=(300_000, 0, 0, 0)), message(tool_use("t2", "submit_scan", scan_result()))],
        budget=None,
    )
    assert error is None and result is not None and state.budget_nudged_at is None


def test_turns_left_note_names_the_scan_submit_tool(repo: Path):
    client, _, _, error = _run(repo, [message(ls(i)) for i in range(1, 4)], budget=None, max_turns=3)
    assert error is not None and "no result after 3 turns" in str(error)
    last = client.requests[2]["messages"][-1]["content"]
    assert "turns left" in last[-1]["text"] and "submit_scan" in last[-1]["text"]


# ---------- settings and config ----------


def test_scan_config_defaults_and_overrides(tmp_path: Path):
    d = Config().scan
    assert (d.max_turns, d.effort, d.max_input_tokens) == (60, "high", 3_000_000)
    assert (d.max_issues, d.min_severity) == (5, "P1")
    (tmp_path / ".simpli-review").mkdir()
    (tmp_path / ".simpli-review" / "config.json").write_text(
        '{"scan": {"maxTurns": 20, "effort": "medium", "maxInputTokens": 500000, "maxIssues": 2, '
        '"minSeverity": "P0"}}',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    s = scan_settings(cfg, {})
    assert (s.max_turns, s.effort, s.budget_tokens) == (20, "medium", 500_000)
    assert cfg.scan.max_issues == 2 and cfg.scan.min_severity == "P0"


def test_action_inputs_override_effort_and_turns_and_empty_means_unset():
    cfg = Config()
    env = {"INPUT_EFFORT": "xhigh", "INPUT_MAX_TURNS": "12", "INPUT_MODEL": "claude-sonnet-5-5"}
    s = scan_settings(cfg, env)
    assert (s.effort, s.max_turns, s.model) == ("xhigh", 12, "claude-sonnet-5-5")
    s = scan_settings(cfg, {"INPUT_EFFORT": "", "INPUT_MAX_TURNS": " "})
    assert (s.effort, s.max_turns, s.model) == ("high", 60, "claude-opus-5-5")
    for bad in ({"INPUT_EFFORT": "turbo"}, {"INPUT_MAX_TURNS": "many"}, {"INPUT_MAX_TURNS": "0"}):
        with pytest.raises(ScanConfigError):
            scan_settings(cfg, bad)


# ---------- the map ----------


@pytest.fixture
def risky_repo(repo: Path) -> Path:
    files = {
        "app/auth/session.py": "def login(user):\n    return user\n",
        "app/routes/orders.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "\n"
            '@router.post("/orders")\n'
            "def create_order(body):\n    return body\n"
            "\n"
            '@router.get("/orders/{order_id}")\n'
            "def get_order(order_id):\n    return order_id\n"
        ),
        "app/webhooks/stripe_events.py": "def handle(event):\n    return event\n",
        "migrations/versions/0001_orders.sql": "CREATE TABLE orders (id uuid);\n",
        "app/util.py": "def helper():\n    return 1\n",
        "web/server.ts": "app.post('/hook', (req, res) => res.send('ok'));\n",
    }
    for rel, body in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return repo


def test_repo_map_puts_risky_code_first(risky_repo: Path):
    m = repo_map(RepoIndex(risky_repo))
    first = m.split("## Entry points")[0]
    listed = [ln.split(" ")[1] for ln in first.splitlines() if ln.startswith("- ")]
    session, hook, orders = "app/auth/session.py", "app/webhooks/stripe_events.py", "app/routes/orders.py"
    assert listed[0] == session
    assert listed.index(session) < listed.index(hook) < listed.index(orders)
    assert "app/util.py" not in listed  # nothing risky about it: only in the full file list
    assert "app/routes/orders.py:4  @router.post" in m and "app/routes/orders.py:8  @router.get" in m
    assert "web/server.ts:1  app.post('/hook'" in m
    assert "## Migrations" in m and "- migrations/versions/0001_orders.sql" in m
    assert "## All indexed files" in m and "app/: " in m and "util.py (3)" in m


def test_scan_message_has_context_files_and_map(risky_repo: Path):
    idx = RepoIndex(risky_repo)
    cfg = load_config(risky_repo)
    msg = build_scan_message(repo="o/r", commit=SHA, ref="main", index=idx, config=cfg, repo_root=risky_repo)
    assert "Context file: CLAUDE.md" in msg[0]["text"]
    assert f"commit {SHA}" in msg[1]["text"] and "Read these first" in msg[1]["text"]
    assert "SECRET=do-not-read" not in msg[0]["text"] + msg[1]["text"]  # secret files are never indexed


# ---------- finalize: evidence, confidence, scrubbing ----------


def test_finalize_drops_low_confidence_bad_locations_and_missing_evidence(repo: Path):
    r = ScanResult.model_validate(scan_result(
        finding(),
        finding(title="Too unsure", confidence=2),
        finding(title="No such file", path="app/nope.py"),
        finding(title="Past the end of the file", line=999),
        finding(title="Escapes the repo", path="../outside.py"),
        finding(title="No evidence cited", evidence="I looked and it seems wrong"),
        finding(title="Charge amount is not validated"),  # repeated
    ))  # fmt: skip
    out = finalize_scan(r, index=RepoIndex(repo), repo_root=repo, config=Config())
    assert [f.title for f in out.findings] == ["Charge amount is not validated"]
    assert (out.dropped_low_confidence, out.dropped_invalid, out.duplicates) == (1, 4, 1)


def test_finalize_orders_by_severity_then_confidence_and_honours_min_confidence(repo: Path):
    r = ScanResult.model_validate(scan_result(
        finding(title="hardening gap high", severity="P2", confidence=5),
        finding(title="session fixation moderate", confidence=3),
        finding(title="tenant leak critical", severity="P0", confidence=3),
        finding(title="order lookup unscoped sure", confidence=5),
    ))  # fmt: skip
    out = finalize_scan(r, index=RepoIndex(repo), repo_root=repo, config=Config(minConfidence=3))
    assert [f.title for f in out.findings] == [
        "tenant leak critical",
        "order lookup unscoped sure",
        "session fixation moderate",
        "hardening gap high",
    ]
    out = finalize_scan(r, index=RepoIndex(repo), repo_root=repo, config=Config(minConfidence=5))
    assert [f.title for f in out.findings] == [
        "order lookup unscoped sure",
        "hardening gap high",
    ] and out.dropped_low_confidence == 2


def test_finalize_scrubs_secret_values(repo: Path):
    r = ScanResult.model_validate(scan_result(finding(
        title=f"Hard-coded key {FAKE_API_KEY} in settings",
        body=f'The code sets password = "{FAKE_PASSWORD}" and mails a@b.com.',
        evidence="app/billing.py:5 api_key: fakevalue1234",
    )))  # fmt: skip
    f = finalize_scan(r, index=RepoIndex(repo), repo_root=repo, config=Config()).findings[0]
    blob = f.title + f.body + f.evidence
    for secret in (FAKE_API_KEY, FAKE_PASSWORD, "a@b.com", "fakevalue1234"):
        assert secret not in blob
    assert "app/billing.py:5" in f.evidence


def test_redact_secrets_is_idempotent():
    once = redact_secrets(f'db_password = "{FAKE_PASSWORD}" and token: ghp_fakefakefake1234')
    assert FAKE_PASSWORD not in once and "ghp_fakefakefake1234" not in once
    assert redact_secrets(once) == once


# ---------- fingerprints and de-duplication ----------


def test_fingerprint_ignores_line_numbers_word_order_and_case():
    a = fingerprint("app/billing.py", "Charge amount is not validated")
    assert a == fingerprint("app/billing.py", "Not validated: the CHARGE amount")  # same words
    assert a.digest != fingerprint("app/other.py", "Charge amount is not validated").digest
    assert a.digest != fingerprint("app/billing.py", "Refund amount is not validated").digest
    assert len(a.digest) == 16


def test_marker_round_trip_and_cannot_close_the_comment():
    fp = fingerprint("a/b-->c.py", "Title with <script> and --> in it")
    body = "text\n" + marker(fp)
    assert body.count("-->") == 1  # only the real terminator
    parsed = known_issues([{"number": 7, "state": "open", "state_reason": None, "body": body}])
    assert parsed[0].fp == fp and parsed[0].number == 7
    assert known_issues([{"number": 8, "state": "open", "body": "no marker here"}]) == []


def _known(f: ScanFinding, state: str, reason: str | None = None, title: str | None = None):
    fp = fingerprint(f.path, title or f.title)
    return known_issues([{"number": 1, "state": state, "state_reason": reason, "body": marker(fp)}])


def test_open_and_not_planned_issues_suppress_but_completed_ones_do_not():
    f = sf()
    fp = fingerprint(f.path, f.title)
    assert match_known(fp, _known(f, "open")) == "duplicate_open"
    assert match_known(fp, _known(f, "closed", "not_planned")) == "duplicate_not_planned"
    assert match_known(fp, _known(f, "closed", "completed")) is None
    assert match_known(fp, _known(f, "closed", None)) is None


def test_a_reworded_title_in_the_same_file_is_the_same_finding():
    f = sf(title="Charge amount is not validated before the payment call")
    known = _known(f, "open", title="Charge amount not validated before payment call")
    assert match_known(fingerprint(f.path, f.title), known) == "duplicate_open"
    other = sf(title="Webhook signature is checked after parsing the body")
    assert match_known(fingerprint(other.path, other.title), known) is None
    elsewhere = sf(path="app/money.py", title=f.title)
    assert match_known(fingerprint(elsewhere.path, elsewhere.title), known) is None


# ---------- the issue plan and the cap ----------


def _findings(n: int, severity: str = "P1") -> list[ScanFinding]:
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    return [
        sf(title=f"{names[i]} endpoint leaks records", severity=severity, path=f"app/m{i}.py")
        for i in range(n)
    ]


def test_issue_cap_and_min_severity():
    fs = [*_findings(7), sf(title="Weak default", severity="P2", path="app/z.py")]
    plan = plan_issues(fs, [], ScanConfig())
    assert (
        plan.count("pending") == 5 and plan.count("over_cap") == 2 and plan.count("below_min_severity") == 1
    )
    two = [sf(severity="P0"), sf(title="Another thing wrong", severity="P1")]
    plan = plan_issues(two, [], ScanConfig(minSeverity="P0"))
    assert [a.status for a in plan.actions] == ["pending", "below_min_severity"]
    assert plan_issues(_findings(3), [], ScanConfig(maxIssues=0)).count("over_cap") == 3


def test_worst_findings_get_the_issues_when_over_the_cap():
    low = sf(title="lower one", severity="P1", confidence=3, path="app/low.py")
    top = sf(title="critical one", severity="P0", confidence=3, path="app/top.py")
    mid = sf(title="middle one", severity="P1", confidence=5, path="app/mid.py")
    plan = plan_issues([low, mid, top], [], ScanConfig(maxIssues=2))
    assert {a.finding.title: a.status for a in plan.actions} == {
        "critical one": "pending", "middle one": "pending", "lower one": "over_cap"
    }  # fmt: skip


def test_duplicates_do_not_use_up_the_cap_and_repeats_in_one_run_collapse():
    fs = _findings(3)
    known = _known(fs[0], "open") + _known(fs[1], "closed", "not_planned")
    plan = plan_issues(fs, known, ScanConfig(maxIssues=1))
    assert [a.status for a in plan.actions] == ["duplicate_open", "duplicate_not_planned", "pending"]
    twin = plan_issues([fs[0], fs[0].model_copy(update={"line": 9})], [], ScanConfig())
    assert [a.status for a in twin.actions] == ["pending", "duplicate_open"]


# ---------- issue text and publishing ----------


class FakeGH:
    def __init__(self, issues: list[dict[str, Any]] | None = None, fail_create: bool = False) -> None:
        self.issues = issues or []
        self.fail_create = fail_create
        self.labels: list[tuple[str, str, str]] = []
        self.created: list[dict[str, Any]] = []

    def labeled_issues(self, label: str, limit: int = 1000) -> list[dict[str, Any]]:
        assert label == LABEL
        return self.issues

    def ensure_label(self, name: str, color: str, description: str) -> None:
        self.labels.append((name, color, description))

    def create_issue(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        if self.fail_create:
            raise GitHubError(403, "Resource not accessible")
        self.created.append({"title": title, "body": body, "labels": labels})
        n = 100 + len(self.created)
        return {"number": n, "html_url": f"https://github.com/o/r/issues/{n}"}


def test_issue_title_body_links_prompt_marker_and_scrubbing():
    f = sf(
        title=f"Key {FAKE_API_KEY} logged on failure",
        body=f'Logs `password = "{FAKE_PASSWORD}"`. cc @dale and @acme/team. Fix: stop logging it.',
        end_line=7,
    )
    plan = plan_issues([f], [], ScanConfig())
    files = {"app/billing.py", "app/money.py"}
    body = issue_body(plan.actions[0], server="https://github.com", repo="o/r", sha=SHA, files=files)
    assert issue_title(f).startswith("[AI scan] ") and "sk-fake" not in issue_title(f)
    assert f"https://github.com/o/r/blob/{SHA}/app/billing.py#L5-L7" in body
    assert "(at scanned commit `aaaaaaa`)" in body
    assert f"(https://github.com/o/r/blob/{SHA}/app/money.py#L2)" in body  # evidence refs are linked
    assert "**P1** · security · confidence 4/5" in body
    assert "<summary>Prompt for AI agents</summary>" in body and "In app/billing.py around lines 5-7" in body
    assert FAKE_PASSWORD not in body and "sk-fake" not in body
    assert "@dale" not in body and "@acme/team" not in body  # mentions are defused
    assert body.rstrip().endswith("-->") and parse_marker(body) == plan.actions[0].fp
    assert "Nothing closes this issue automatically" in body


def test_publish_creates_the_label_and_issues_and_survives_failures():
    plan = plan_issues(_findings(2), [], ScanConfig())
    gh = FakeGH()
    publish(gh, plan, server="https://github.com", repo="o/r", sha=SHA, files=set())  # type: ignore[arg-type]
    assert gh.labels == [("ai-scan", "5319e7", "Found by the scheduled AI security scan")]
    assert len(LABEL_DESCRIPTION) <= 100  # GitHub's limit for a label description
    assert [c["labels"] for c in gh.created] == [["ai-scan"], ["ai-scan"]]
    assert [(a.status, a.number) for a in plan.actions] == [("created", 101), ("created", 102)]

    plan = plan_issues(_findings(2), [], ScanConfig())
    publish(FakeGH(fail_create=True), plan, server="s", repo="o/r", sha=SHA, files=set())  # type: ignore[arg-type]
    assert [a.status for a in plan.actions] == ["failed", "failed"]


def test_dry_run_opens_nothing():
    plan = plan_issues(_findings(2), [], ScanConfig())
    gh = FakeGH()
    publish(gh, plan, server="s", repo="o/r", sha=SHA, files=set(), dry_run=True)  # type: ignore[arg-type]
    assert gh.created == [] and gh.labels == [] and plan.count("dry_run") == 2


def test_a_finding_seen_again_leaves_its_open_issue_alone():
    f = sf()
    issue = {"number": 3, "state": "open", "state_reason": None, "body": marker(fingerprint(f.path, f.title))}
    gh = FakeGH(issues=[issue])
    plan = plan_issues([f], known_issues(gh.labeled_issues(LABEL)), ScanConfig())
    publish(gh, plan, server="s", repo="o/r", sha=SHA, files=set())  # type: ignore[arg-type]
    assert plan.actions[0].status == "duplicate_open"
    assert gh.created == [] and gh.labels == []
    # The GitHub client has no way to comment on or close an issue, so a scan cannot do either.
    assert not [n for n in dir(GitHub) if "close" in n or "comment_issue" in n or "update_issue" in n]


# ---------- end to end: the report files, the job summary, the log ----------


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"GITHUB_REPOSITORY": "o/r", "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), **extra}


def _report(repo: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((repo / "simpli-review-scan.json").read_text(encoding="utf-8"))
    return data


def test_scan_writes_report_files_summary_and_spend(
    git_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    gh = FakeGH()
    second = finding(title="Refund path skips the same check", severity="P2", line=10)
    client = FakeClient([
        message(tool_use("t1", "read_file", {"path": "app/money.py"}), tokens=(1000, 200, 500, 300)),
        message(tool_use("t2", "submit_scan", scan_result(finding(), second)), tokens=(2000, 400, 1500, 0)),
    ])  # fmt: skip
    code = scan_main(repo=str(git_repo), env=_env(tmp_path), client=client, gh=gh)  # type: ignore[arg-type]
    assert code == 0
    out = capsys.readouterr().out
    assert "model: claude-opus-5-5  effort: high  turns: 2/60" in out
    assert "input 3,000  output 600  cache read 2,000  cache write 300  total 5,900" in out
    # 3000 x $4 + 600 x $20 + 2000 x $0.20 + 300 x $5 per million = $0.0259
    assert "estimated cost (estimate, list prices as of 2026-10-06): $0.03" in out

    data = _report(git_repo)
    assert data["status"] == "complete" and data["model"] == "claude-opus-5-5" and data["effort"] == "high"
    assert data["turns"] == {"used": 2, "max": 60}
    assert data["usage"] == {
        "input": 3000,
        "output": 600,
        "cache_read": 2000,
        "cache_write": 300,
        "total": 5900,
    }
    assert data["cost"]["is_estimate"] is True and data["cost"]["estimate_usd_micros"] == 25_900
    assert data["cost"]["estimate_usd"] == "$0.03" and data["cost"]["currency"] == "USD"
    assert data["budget"] == {"max_tokens": 3_000_000, "used_tokens": 5900, "told_to_submit_at_turn": None}
    assert re.fullmatch(r"[0-9a-f]{40}", data["commit"]) and data["repository"] == "o/r"
    assert data["duration_seconds"] >= 0 and data["started_at"].endswith("+00:00")
    statuses = {f["title"]: f["issue"]["status"] for f in data["findings"]}
    assert statuses == {
        "Charge amount is not validated": "created",
        "Refund path skips the same check": "below_min_severity",
    }
    assert data["issues"]["created"] == 1 and data["issues"]["below_min_severity"] == 1

    md = (git_repo / "simpli-review-scan.md").read_text(encoding="utf-8")
    assert "Estimated cost (estimate)" in md and "| Turns used | 2 of 60 |" in md and "#101" in md
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == md + "\n"
    assert [c["title"] for c in gh.created] == ["[AI scan] Charge amount is not validated"]


def test_budget_failure_still_writes_the_report_and_the_cost(
    git_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    (git_repo / ".simpli-review").mkdir()
    (git_repo / ".simpli-review" / "config.json").write_text(
        '{"scan": {"maxInputTokens": 10000}}', encoding="utf-8"
    )
    client = FakeClient([message(ls(i), tokens=(3000, 100, 0, 0)) for i in range(1, 6)])
    gh = FakeGH()
    code = scan_main(repo=str(git_repo), env=_env(tmp_path), client=client, gh=gh)  # type: ignore[arg-type]
    assert code == 1
    assert (
        "::error::The scan did not finish: the token budget (10,000) was reached" in capsys.readouterr().out
    )
    data = _report(git_repo)
    assert data["status"] == "incomplete" and data["findings"] == [] and "token budget" in data["error"]
    assert data["usage"]["input"] == 9000 and data["cost"]["estimate_usd_micros"] > 0
    assert data["budget"]["told_to_submit_at_turn"] == 2
    assert "INCOMPLETE" in (git_repo / "simpli-review-scan.md").read_text(encoding="utf-8")
    assert gh.created == []


def test_scan_dedupes_against_existing_issues_and_respects_the_cap(git_repo: Path, tmp_path: Path):
    (git_repo / ".simpli-review").mkdir()
    (git_repo / ".simpli-review" / "config.json").write_text('{"scan": {"maxIssues": 1}}', encoding="utf-8")
    known = sf(title="Charge amount is not validated")
    marked = marker(fingerprint(known.path, known.title))
    gh = FakeGH(issues=[{"number": 3, "state": "closed", "state_reason": "not_planned", "body": marked}])
    client = FakeClient([message(tool_use("t1", "submit_scan", scan_result(
        finding(),
        finding(title="Refund endpoint trusts client currency", path="app/api.py", line=2),
        finding(title="Money helper truncates cents", path="app/money.py", line=2),
    )))])  # fmt: skip
    assert scan_main(repo=str(git_repo), env=_env(tmp_path), client=client, gh=gh) == 0  # type: ignore[arg-type]
    by = {f["title"]: f["issue"]["status"] for f in _report(git_repo)["findings"]}
    assert by["Charge amount is not validated"] == "duplicate_not_planned"
    assert sorted(by.values()) == ["created", "duplicate_not_planned", "over_cap"]
    assert len(gh.created) == 1


def test_scan_dry_run_opens_no_issues_and_says_so(git_repo: Path, tmp_path: Path):
    gh = FakeGH()
    client = FakeClient([message(tool_use("t1", "submit_scan", scan_result(finding())))])
    code = scan_main(repo=str(git_repo), dry_run=True, env=_env(tmp_path), client=client, gh=gh)  # type: ignore[arg-type]
    assert code == 0
    assert _report(git_repo)["issues"]["would_create"] == 1 and gh.created == [] and gh.labels == []


def test_report_and_issues_never_contain_secret_values(git_repo: Path, tmp_path: Path):
    gh = FakeGH()
    leaky = finding(
        title=f"Payment key {FAKE_API_KEY} committed",
        body=f'settings.py has STRIPE_SECRET = "{FAKE_WEBHOOK_SECRET}". Contact ops@example.com.',
        evidence=f"app/billing.py:5 api_key = '{FAKE_PASSWORD}'",
    )
    client = FakeClient([message(tool_use("t1", "submit_scan", scan_result(leaky)))])
    assert scan_main(repo=str(git_repo), env=_env(tmp_path), client=client, gh=gh) == 0  # type: ignore[arg-type]
    everything = (
        (git_repo / "simpli-review-scan.json").read_text(encoding="utf-8")
        + (git_repo / "simpli-review-scan.md").read_text(encoding="utf-8")
        + (tmp_path / "summary.md").read_text(encoding="utf-8")
        + json.dumps(gh.created)
    )
    for secret in (FAKE_API_KEY, FAKE_WEBHOOK_SECRET, FAKE_PASSWORD, "ops@example.com"):
        assert secret not in everything
    assert len(gh.created) == 1


def test_bad_effort_input_is_a_setup_error(
    git_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    env = _env(tmp_path, INPUT_EFFORT="turbo")
    assert scan_main(repo=str(git_repo), env=env, client=FakeClient([]), gh=FakeGH()) == 2  # type: ignore[arg-type]
    assert "effort must be one of" in capsys.readouterr().out


def test_refusal_is_reported_as_incomplete_with_spend(git_repo: Path, tmp_path: Path):
    client = FakeClient([message(text("no"), stop="refusal", tokens=(500, 0, 0, 0))])
    assert scan_main(repo=str(git_repo), env=_env(tmp_path), client=client, gh=FakeGH()) == 1  # type: ignore[arg-type]
    data = _report(git_repo)
    assert data["status"] == "incomplete" and data["usage"]["input"] == 500 and "declined" in data["error"]


# ---------- the example workflow and the action ----------


def _load(name: str) -> dict[Any, Any]:
    loaded: dict[Any, Any] = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
    return loaded


def test_example_workflow_is_hardened():
    wf = _load("examples/ai-scan.yml")
    on = wf.get("on", wf.get(True))
    assert on["schedule"] == [{"cron": "17 9 * * 1"}]
    assert set(on["workflow_dispatch"]["inputs"]) == {"effort", "max-turns"}
    # Monthly: cron fires every Monday, the `when` job lets only the first Monday (or a manual run) through.
    assert wf["permissions"] == {}
    assert wf["concurrency"]["group"] and wf["concurrency"]["cancel-in-progress"] is False
    gate = wf["jobs"]["when"]
    assert "permissions" not in gate and gate["outputs"]["run"]
    job = wf["jobs"]["scan"]
    assert job["needs"] == "when" and "needs.when.outputs.run" in job["if"]
    assert job["permissions"] == {"contents": "read", "issues": "write"}
    assert job["timeout-minutes"] == 45
    checkout, scan, upload = job["steps"]
    for step in (checkout, scan, upload):
        assert not re.search(r"@v\d", step["uses"]), f"{step['uses']} is referenced by tag"
    assert checkout["with"]["persist-credentials"] is False and checkout["with"]["fetch-depth"] > 1
    assert scan["with"]["mode"] == "scan" and scan["uses"] == "simpli-software/simpli-review@<commit-sha>"
    for step in (checkout, upload):
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"])
    assert upload["with"]["name"] == "simpli-review-scan"
    paths = upload["with"]["path"]
    assert "simpli-review-scan.json" in paths and "simpli-review-scan.md" in paths
    raw = (ROOT / "examples/ai-scan.yml").read_text(encoding="utf-8")
    assert re.search(r"simpli-review@\S+ # v\d", raw)  # a version comment follows the hash


def test_action_has_mode_input_defaulting_to_review_and_passes_inputs_via_env():
    action = _load("action.yml")
    assert action["inputs"]["mode"]["default"] == "review"
    assert action["inputs"]["name"].get("required") is not True
    run_step = action["runs"]["steps"][-1]
    assert run_step["env"]["INPUT_MODE"] == "${{ inputs.mode }}"
    assert "${{" not in run_step["run"]  # no expression is pasted into the shell script


def test_api_error_text_includes_the_api_message() -> None:
    from simpli_review.scrub import api_error_text

    class FakeBadRequest(Exception):
        status_code = 400

        def __init__(self) -> None:
            super().__init__()
            self.body = {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "tools.6: bad schema"},
            }

    text = api_error_text(FakeBadRequest())
    assert "HTTP 400" in text and "invalid_request_error: tools.6: bad schema" in text
    assert api_error_text(ValueError("x")) == "Anthropic API error (ValueError)"


def test_every_workflow_reference_is_pinned_by_hash_or_a_placeholder():
    sources = [ROOT / "action.yml", ROOT / "examples/code-review.yml", ROOT / ".github/workflows/ci.yml"]
    for path in sources:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*-?\s*uses:\s*(\S+)\s*(#.*)?$", line)
            if not m:
                continue
            ref, comment = m.group(1), m.group(2) or ""
            assert re.fullmatch(r"[\w./-]+@([0-9a-f]{40}|<commit-sha>)", ref), f"{path.name}: {ref}"
            assert re.search(r"# v\d", comment), f"{path.name}: {ref} has no version comment"


def test_action_has_marketplace_branding():
    assert _load("action.yml")["branding"] == {"icon": "eye", "color": "purple"}
