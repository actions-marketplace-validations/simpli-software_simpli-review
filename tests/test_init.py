"""`simpli-review init`: no network and no real gh; the API and gh are fakes."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from simpli_review import init as init_mod
from simpli_review.cli import main
from simpli_review.config import DEFAULT_IGNORE, Config
from simpli_review.init import (
    Answers,
    InitOptions,
    PinError,
    Proc,
    build_config,
    detect_ignore_patterns,
    make_http_get,
    parse_ref,
    render_ai_scan,
    render_code_review,
    resolve_latest,
    run_init,
    yaml_scalar,
)

SHA = "a" * 40
TAG_SHA = "b" * 40
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
UP = init_mod.UPSTREAM


def fake_api(*, annotated: bool = False, tag: str = "v1.4.2") -> Callable[[str], Any]:
    def get(path: str) -> Any:
        if path == f"/repos/{UP}/releases/latest":
            return {"tag_name": tag}
        if path == f"/repos/{UP}/git/ref/tags/{tag}":
            return {
                "object": {"type": "tag" if annotated else "commit", "sha": TAG_SHA if annotated else SHA}
            }
        if annotated and path == f"/repos/{UP}/git/tags/{TAG_SHA}":
            return {"object": {"type": "commit", "sha": SHA}}
        raise PinError(f"unexpected path {path}")

    return get


def offline(path: str) -> Any:
    raise PinError("could not reach the GitHub API (URLError)")


class FakeGh:
    """Stands in for the gh CLI and records every call."""

    def __init__(
        self,
        *,
        authed: bool = True,
        labels: tuple[str, ...] = (),
        secrets: tuple[str, ...] = (),
        org_secrets: tuple[str, ...] = (),
    ) -> None:
        self.authed, self.labels, self.secrets, self.org_secrets = authed, set(labels), secrets, org_secrets
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], cwd: Path) -> Proc:
        self.calls.append(args)
        match args[1:3]:
            case ["auth", "status"]:
                return Proc(0 if self.authed else 1, "", "")
            case ["label", "list"]:
                return Proc(0, json.dumps([{"name": n} for n in sorted(self.labels)]), "")
            case ["label", "create"]:
                self.labels.add(args[3])
                return Proc(0, "", "")
            case ["secret", "list"]:
                names = self.org_secrets if "--org" in args else self.secrets
                return Proc(0, json.dumps([{"name": n} for n in names]), "")
            case ["repo", "view"]:
                return Proc(0, "acme\n", "")
        return Proc(1, "", "unexpected")

    def verbs(self) -> set[str]:
        return {" ".join(c[1:3]) for c in self.calls}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def go(
    root: Path,
    *,
    gh: FakeGh | None = None,
    http_get: Callable[[str], Any] | None = None,
    answers: list[str] | None = None,
    which: Callable[[str], str | None] = lambda name: "/usr/bin/gh",
    **opts: Any,
) -> tuple[int, str]:
    out = io.StringIO()
    feed = iter(answers or [])

    def input_fn(prompt: str) -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    opts.setdefault("yes", answers is None)
    code = run_init(
        InitOptions(root=root, **opts),
        input_fn=input_fn,
        http_get=http_get or fake_api(),
        run_cmd=gh or FakeGh(),
        which=which,
        env={},
        out=out,
    )
    return code, out.getvalue()


def read_yaml(root: Path, rel: str) -> dict[Any, Any]:
    doc = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


# ---------- templates ----------


def _steps(doc: dict[Any, Any]) -> list[dict[str, Any]]:
    return [s for job in doc["jobs"].values() for s in job["steps"]]


def _normalise(doc: dict[Any, Any], heading: str | None) -> dict[Any, Any]:
    """Blank the parts init fills in, so the rest must match the example exactly."""
    if heading is not None:
        doc["name"] = "H"
    for step in _steps(doc):
        if str(step.get("uses", "")).startswith(UP + "@"):
            step["uses"] = "PINNED"
            if heading is not None:
                step["with"]["name"] = "H"
    return doc


def test_code_review_matches_the_example() -> None:
    generated = yaml.safe_load(render_code_review(Answers(), init_mod.placeholder_pin()))
    example = yaml.safe_load((EXAMPLES / "code-review.yml").read_text(encoding="utf-8"))
    assert _normalise(generated, "x") == _normalise(example, "x")


def test_ai_scan_matches_the_example_and_is_first_monday_monthly() -> None:
    text = render_ai_scan(init_mod.placeholder_pin())
    generated = yaml.safe_load(text)
    example = yaml.safe_load((EXAMPLES / "ai-scan.yml").read_text(encoding="utf-8"))
    assert _normalise(generated, None) == _normalise(example, None)
    assert generated["permissions"] == {}
    assert set(generated["jobs"]) == {"when", "scan"}
    assert generated["jobs"]["scan"]["needs"] == "when"
    assert generated["jobs"]["scan"]["permissions"] == {"contents": "read", "issues": "write"}


def test_every_third_party_action_is_pinned_by_hash() -> None:
    for text in (
        render_code_review(Answers(), init_mod.placeholder_pin()),
        render_ai_scan(init_mod.placeholder_pin()),
    ):
        for step in _steps(yaml.safe_load(text)):
            uses = step.get("uses")
            if uses and not uses.startswith(UP + "@"):
                assert len(uses.split("@")[1]) == 40


# ---------- pinning ----------


def test_resolve_lightweight_tag() -> None:
    pin = resolve_latest(fake_api())
    assert (pin.sha, pin.comment, pin.resolved) == (SHA, "v1.4.2", True)
    assert pin.uses == f"{UP}@{SHA} # v1.4.2"


def test_resolve_dereferences_annotated_tag() -> None:
    assert resolve_latest(fake_api(annotated=True)).sha == SHA


def test_resolve_rejects_odd_tag_names() -> None:
    with pytest.raises(PinError):
        resolve_latest(fake_api(tag="v1 # evil"))


def test_resolve_rejects_non_commit_target() -> None:
    def get(path: str) -> Any:
        if path.endswith("releases/latest"):
            return {"tag_name": "v1.0.0"}
        return {"object": {"type": "tree", "sha": SHA}}

    with pytest.raises(PinError):
        resolve_latest(get)


def test_parse_ref() -> None:
    pin = parse_ref(f"someone/fork@{SHA.upper()}")
    assert (pin.repo, pin.sha) == ("someone/fork", SHA)
    for bad in ("someone/fork", "someone/fork@v1", f"a b/c@{SHA}", f"/x@{SHA}"):
        with pytest.raises(init_mod.InitError):
            parse_ref(bad)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()


class _FakeOpener:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def open(self, req: Any, timeout: float) -> _FakeResponse:
        self.requests.append(req)
        return _FakeResponse(b'{"tag_name": "v1.0.0"}')


@pytest.mark.parametrize(
    ("env", "expect_auth"),
    [
        ({}, False),
        ({"GITHUB_TOKEN": "t0k"}, True),
        ({"GH_TOKEN": "t0k"}, True),
        ({"GITHUB_TOKEN": "  "}, False),
    ],
)
def test_token_is_sent_only_when_set(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expect_auth: bool
) -> None:
    opener = _FakeOpener()
    monkeypatch.setattr(init_mod.urllib.request, "build_opener", lambda *h: opener)
    assert make_http_get(env)("/repos/x/y/releases/latest") == {"tag_name": "v1.0.0"}
    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert ("authorization" in headers) is expect_auth
    assert opener.requests[0].full_url.startswith("https://api.github.com/")


# ---------- the command ----------


def test_defaults_write_the_files(root: Path) -> None:
    code, out = go(root)
    assert code == 0
    review = read_yaml(root, ".github/workflows/code-review.yml")
    assert review["name"] == "Code review"
    assert _steps(review)[1]["uses"] == f"{UP}@{SHA}"
    assert f"{UP}@{SHA} # v1.4.2" in (root / ".github/workflows/code-review.yml").read_text(encoding="utf-8")
    scan = read_yaml(root, ".github/workflows/ai-scan.yml")
    assert scan["permissions"] == {}
    cfg = json.loads((root / ".simpli-review/config.json").read_text(encoding="utf-8"))
    assert cfg == {"blockOn": "P0", "disabledLabels": ["no-review"]}
    assert Config.model_validate(cfg).block_on == "P0"
    assert "Pinned" in out
    assert "review-override" in out and "maintainers" in out


def test_files_use_lf_line_endings(root: Path) -> None:
    go(root)
    for rel in (
        ".github/workflows/code-review.yml",
        ".github/workflows/ai-scan.yml",
        ".simpli-review/config.json",
    ):
        data = (root / rel).read_bytes()
        assert b"\r" not in data
        assert data.endswith(b"\n")


def test_offline_writes_placeholder_and_explains(root: Path) -> None:
    code, out = go(root, http_get=offline)
    assert code == 0
    text = (root / ".github/workflows/code-review.yml").read_text(encoding="utf-8")
    assert f"{UP}@<commit-sha> # v1.x" in text
    assert "TODO: replace <commit-sha>" in text
    assert "Could not pin" in out and "git/ref/tags/<tag>" in out
    assert "Replace <commit-sha>" in out
    yaml.safe_load(text)


def test_ref_pins_that_build_without_api(root: Path) -> None:
    code, _ = go(root, ref=f"{UP}@{SHA}", http_get=offline)
    assert code == 0
    assert f"{UP}@{SHA} # pinned with --ref" in (root / ".github/workflows/code-review.yml").read_text(
        encoding="utf-8"
    )


def test_bad_ref_and_not_a_repo(tmp_path: Path, root: Path) -> None:
    code, out = go(root, ref="nope")
    assert code == 2 and "--ref" in out
    assert not (root / ".github").exists()
    code, out = go(tmp_path / "missing")
    assert code == 2 and "repository" in out


def test_existing_files_are_never_overwritten_without_force(root: Path) -> None:
    wf = root / ".github/workflows/code-review.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text("name: mine\non: push\njobs: {}\n", encoding="utf-8")
    code, out = go(root)
    assert code == 0
    assert wf.read_text(encoding="utf-8") == "name: mine\non: push\njobs: {}\n"
    assert "skipped" in out and "-name: mine" in out and "--force" in out
    assert (root / ".github/workflows/ai-scan.yml").exists()


def test_force_overwrites_and_rerun_is_unchanged(root: Path) -> None:
    wf = root / ".github/workflows/code-review.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text("old\n", encoding="utf-8")
    go(root, force=True)
    assert yaml.safe_load(wf.read_text(encoding="utf-8"))["name"] == "Code review"
    _, out = go(root)
    assert out.count("unchanged") == 3


def test_dry_run_writes_nothing_and_prints_content(root: Path) -> None:
    gh = FakeGh()
    code, out = go(root, dry_run=True, gh=gh)
    assert code == 0
    assert not (root / ".github").exists() and not (root / ".simpli-review").exists()
    assert "would-create" in out and "| name: Code review" in out and '"blockOn": "P0"' in out
    assert gh.calls == []


def test_labels_created_and_existing_skipped(root: Path) -> None:
    gh = FakeGh(labels=("re-review",), secrets=("ANTHROPIC_API_KEY",))
    _, out = go(root, gh=gh)
    assert gh.labels == {"re-review", "review-override", "no-review"}
    assert "review-override, no-review created" in out and "re-review already there" in out
    assert "ANTHROPIC_API_KEY secret exists" in out


def test_no_labels_flag(root: Path) -> None:
    gh = FakeGh(secrets=("ANTHROPIC_API_KEY",))
    _, out = go(root, gh=gh, no_labels=True)
    assert "label create" not in gh.verbs() and "label list" not in gh.verbs()
    assert "gh label create" not in out
    cfg = json.loads((root / ".simpli-review/config.json").read_text(encoding="utf-8"))
    assert "disabledLabels" not in cfg


def test_secret_missing_prints_command_and_never_sets_it(root: Path) -> None:
    gh = FakeGh()
    _, out = go(root, gh=gh)
    assert "gh secret set ANTHROPIC_API_KEY" in out
    assert not any(c[1:3] == ["secret", "set"] for c in gh.calls)
    assert not any("--body" in c or "-b" in c for c in gh.calls)


def test_org_secret_counts(root: Path) -> None:
    _, out = go(root, gh=FakeGh(org_secrets=("ANTHROPIC_API_KEY",)))
    assert "ANTHROPIC_API_KEY secret exists" in out


def test_without_gh_commands_are_printed(root: Path) -> None:
    _, out = go(root, which=lambda name: None)
    assert "gh is not installed" in out
    assert "gh label create re-review" in out and "gh label create no-review" in out
    assert "gh secret set ANTHROPIC_API_KEY" in out


def test_gh_not_signed_in(root: Path) -> None:
    gh = FakeGh(authed=False)
    _, out = go(root, gh=gh)
    assert "not signed in" in out
    assert gh.verbs() == {"auth status"}


# ---------- questions ----------


def test_interactive_answers_shape_the_config(root: Path) -> None:
    answers = [
        "Review bot",  # heading
        "3",  # strictness
        "p1",  # block on
        "off",  # pr-template
        "err",  # declared-scope (unique prefix)
        "n",  # scan
        "dist/**, *.lock",  # ignore patterns
        "We use   FastAPI.",  # instructions
    ]
    code, out = go(root, answers=answers)
    assert code == 0
    cfg = json.loads((root / ".simpli-review/config.json").read_text(encoding="utf-8"))
    assert cfg == {
        "strictness": 3,
        "blockOn": "P1",
        "builtinChecks": {"pr-template": "off", "declared-scope": "error"},
        "ignorePatterns": "dist/**\n*.lock",
        "instructions": "We use FastAPI.",
        "disabledLabels": ["no-review"],
    }
    assert read_yaml(root, ".github/workflows/code-review.yml")["name"] == "Review bot"
    assert not (root / ".github/workflows/ai-scan.yml").exists()
    assert "Scan budget" not in out


def test_scan_budget_is_written_when_changed(root: Path) -> None:
    answers = ["", "", "none", "", "", "yes", "1,500,000", "", ""]
    go(root, answers=answers)
    cfg = json.loads((root / ".simpli-review/config.json").read_text(encoding="utf-8"))
    assert cfg == {"scan": {"maxInputTokens": 1_500_000}, "disabledLabels": ["no-review"]}


def test_enter_accepts_every_default(root: Path) -> None:
    code, _ = go(root, answers=["", "", "", "", "", "", "", "", ""])
    assert code == 0
    cfg = json.loads((root / ".simpli-review/config.json").read_text(encoding="utf-8"))
    assert cfg == {"blockOn": "P0", "disabledLabels": ["no-review"]}


def test_invalid_answer_is_asked_again(root: Path) -> None:
    code, out = go(root, answers=["", "9", "2"])
    assert code == 0
    assert "choose one of" in out


def test_three_invalid_answers_stop_the_command(root: Path) -> None:
    code, out = go(root, answers=["", "9", "x", "7"])
    assert code == 2 and "no valid answer" in out
    assert not (root / ".github").exists()


@pytest.mark.parametrize("heading", ['Review: "bot" #1', "yes", "Ünï review", "a\\b"])
def test_odd_headings_round_trip(root: Path, heading: str) -> None:
    go(root, answers=[heading] + [""] * 8)
    assert read_yaml(root, ".github/workflows/code-review.yml")["name"] == heading
    step = _steps(read_yaml(root, ".github/workflows/code-review.yml"))[1]
    assert step["with"]["name"] == heading


def test_heading_with_expression_is_rejected(root: Path) -> None:
    code, out = go(root, answers=["${{ secrets.X }}", "${{ a }}", "${{ b }}"])
    assert code == 2 and "no valid answer" in out


def test_yaml_scalar_quotes_when_needed() -> None:
    assert yaml_scalar("Code review") == "Code review"
    assert yaml_scalar("true") == '"true"'
    assert yaml_scalar("a: b") == '"a: b"'


# ---------- pieces ----------


def test_detects_lockfiles_and_build_dirs(root: Path) -> None:
    (root / "Cargo.lock").write_text("", encoding="utf-8")
    (root / "go.sum").write_text("", encoding="utf-8")
    (root / "target").mkdir()
    (root / "web").mkdir()
    (root / "web" / ".next").mkdir()
    (root / "web" / "package-lock.json").write_text("", encoding="utf-8")
    found = detect_ignore_patterns(root)
    assert found[: len(DEFAULT_IGNORE.splitlines())] == DEFAULT_IGNORE.splitlines()
    assert "go.sum" in found and "target/**" in found and ".next/**" in found
    assert "Cargo.lock" not in found  # already covered by *.lock
    assert found.count("package-lock.json") == 1


def test_default_answers_give_an_empty_diff_from_defaults() -> None:
    assert (
        build_config(
            Answers(ignore_patterns=tuple(DEFAULT_IGNORE.splitlines()), block_on="none"), labels=False
        )
        == {}
    )


def test_cli_wiring(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(init_mod, "make_http_get", lambda env: fake_api())
    assert main(["init", "--yes", "--dry-run", "--repo", str(root)]) == 0
    assert "would-create" in capsys.readouterr().out
    assert main(["init", "--non-interactive", "--dry-run", "--no-labels", "--repo", str(root)]) == 0
