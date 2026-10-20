"""`simpli-review init`: one-command setup, run from the root of the repository being set up.

It asks a few questions (Enter accepts the default), then writes:

    .github/workflows/code-review.yml   the review workflow (same as examples/code-review.yml)
    .github/workflows/ai-scan.yml       the monthly AI scan, if chosen (same as examples/ai-scan.yml)
    .simpli-review/config.json          only the fields that differ from the defaults

It pins `simpli-software/simpli-review` by commit hash from the latest release (public GitHub API, no
token unless GITHUB_TOKEN or GH_TOKEN is set), creates the PR labels and checks for the
ANTHROPIC_API_KEY secret through `gh` when it is installed and signed in. It never reads, prints or sets
a secret value. Pure standard library apart from the PyYAML and pydantic the package already needs
(to parse the generated files back).
"""

from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import yaml

from simpli_review.config import CONFIG_DIR, DEFAULT_IGNORE, Config, match_glob
from simpli_review.init_templates import AI_SCAN_TEMPLATE, CODE_REVIEW_TEMPLATE

UPSTREAM = "simpli-software/simpli-review"
API_BASE = "https://api.github.com"
API_HOST = "api.github.com"
HTTP_TIMEOUT_S = 10
MAX_RESPONSE_BYTES = 1_000_000
PLACEHOLDER_SHA = "<commit-sha>"
PLACEHOLDER_TAG = "v1.x"
REF_COMMENT = "pinned with --ref"
SECRET_NAME = "ANTHROPIC_API_KEY"

CODE_REVIEW_PATH = ".github/workflows/code-review.yml"
AI_SCAN_PATH = ".github/workflows/ai-scan.yml"
CONFIG_PATH = f"{CONFIG_DIR}/config.json"

DEFAULT_HEADING = "Code review"
DEFAULT_SCAN_BUDGET = 3_000_000
MIN_SCAN_BUDGET = 10_000  # the floor config.ScanConfig enforces
MAX_HEADING = 60
MAX_INSTRUCTIONS = 2000
MAX_TRIES = 3
DIFF_LINES = 20

# (name, description, colour). `no-review` only does something once config has disabledLabels.
LABELS: tuple[tuple[str, str, str], ...] = (
    ("re-review", "Ask the reviewer for another review", "1d76db"),
    ("review-override", "Let this PR through despite blocking findings (maintainers only)", "d93f0b"),
    ("no-review", "Skip the automated review on this PR", "cfd3d7"),
)

LOCKFILES = (
    "yarn.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "bun.lockb",
    "npm-shrinkwrap.json",
    "poetry.lock",
    "Pipfile.lock",
    "uv.lock",
    "pdm.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    "Gemfile.lock",
    "packages.lock.json",
    "Podfile.lock",
    "pubspec.lock",
    "mix.lock",
)
BUILD_DIRS = (
    "dist",
    "build",
    "out",
    "target",
    "coverage",
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".gradle",
    "__pycache__",
)
_SKIP_SUBDIRS = frozenset({".git", *BUILD_DIRS})

_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REF_RE = re.compile(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9a-fA-F]{40})")
_PLAIN_HEADING_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _./()-]*")
_YAML_WORDS = frozenset({"true", "false", "yes", "no", "on", "off", "null", "y", "n"})
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

InputFn = Callable[[str], str]
HttpGet = Callable[[str], Any]


class InitError(Exception):
    """A problem the user can fix; the message is printed and the command exits 2."""


class PinError(Exception):
    """The latest release could not be resolved to a commit."""


class Proc(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


RunCmd = Callable[[list[str], Path], Proc]


@dataclass(frozen=True)
class InitOptions:
    root: Path
    yes: bool = False
    force: bool = False
    dry_run: bool = False
    no_labels: bool = False
    ref: str | None = None


@dataclass(frozen=True)
class Answers:
    heading: str = DEFAULT_HEADING
    strictness: int = 2
    block_on: str = "P0"
    pr_template: str = "warning"
    declared_scope: str = "warning"
    ai_scan: bool = True
    scan_budget: int = DEFAULT_SCAN_BUDGET
    ignore_patterns: tuple[str, ...] = ()
    instructions: str = ""


@dataclass(frozen=True)
class Pin:
    repo: str
    sha: str
    comment: str
    resolved: bool

    @property
    def uses(self) -> str:
        return f"{self.repo}@{self.sha} # {self.comment}"


@dataclass
class FileResult:
    path: str
    status: str  # created, overwritten, unchanged, skipped, would-create, would-overwrite, would-skip
    diff: list[str] = field(default_factory=list)


# ---------- detection ----------


def default_ignore_patterns() -> list[str]:
    return DEFAULT_IGNORE.splitlines()


def detect_ignore_patterns(root: Path) -> list[str]:
    """The built-in ignore list plus any lockfiles and build directories found in the repository.

    `ignorePatterns` replaces the built-in list, so the prefill starts from it and only adds. The
    repository root and its first-level directories are looked at.
    """
    patterns = default_ignore_patterns()
    dirs = [root]
    with contextlib.suppress(OSError):
        dirs += sorted(p for p in root.iterdir() if p.is_dir() and p.name not in _SKIP_SUBDIRS)
    seen_locks: set[str] = set()
    seen_dirs: set[str] = set()
    for d in dirs:
        for name in LOCKFILES:
            if name not in seen_locks and (d / name).is_file():
                seen_locks.add(name)
                if not any(match_glob(name, p) for p in patterns):
                    patterns.append(name)
        for name in BUILD_DIRS:
            if name not in seen_dirs and (d / name).is_dir():
                seen_dirs.add(name)
                glob = f"{name}/**"
                if glob not in patterns:
                    patterns.append(glob)
    return patterns


# ---------- questions ----------


def _choice[T](options: Mapping[str, T]) -> Callable[[str], T]:
    """Case-insensitive match on the whole word or a unique prefix."""

    def parse(raw: str) -> T:
        key = raw.strip().lower()
        if key in options:
            return options[key]
        hits = [k for k in options if k.startswith(key)] if key else []
        if len(hits) == 1:
            return options[hits[0]]
        raise ValueError("choose one of: " + ", ".join(options))

    return parse


def _parse_heading(raw: str) -> str:
    text = " ".join(raw.split())
    if not text or len(text) > MAX_HEADING:
        raise ValueError(f"enter 1 to {MAX_HEADING} characters")
    if _CONTROL_RE.search(raw) or "${{" in text:
        raise ValueError("no control characters or ${{ }} expressions")
    return text


def _parse_budget(raw: str) -> int:
    digits = raw.replace(",", "").replace("_", "").replace(" ", "")
    if not digits.isdigit() or int(digits) < MIN_SCAN_BUDGET:
        raise ValueError(f"enter a whole number of tokens, at least {MIN_SCAN_BUDGET:,}")
    return int(digits)


def _parse_patterns(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() in ("none", "-"):
        return ()
    items = [p.strip() for chunk in raw.splitlines() for p in chunk.split(",")]
    items = [p for p in items if p]
    for p in items:
        if _CONTROL_RE.search(p) or len(p) > 200 or p.startswith("#"):
            raise ValueError(f"not a usable pattern: {p[:40]!r}")
    return tuple(items)


def _parse_instructions(raw: str) -> str:
    text = " ".join(raw.split())
    if len(text) > MAX_INSTRUCTIONS or _CONTROL_RE.search(raw):
        raise ValueError(f"one line, at most {MAX_INSTRUCTIONS} characters")
    return "" if text.lower() == "none" else text


class _Asker:
    def __init__(self, input_fn: InputFn, out: TextIO, interactive: bool) -> None:
        self.input_fn = input_fn
        self.out = out
        self.interactive = interactive

    def ask[T](self, label: str, default_text: str, default: T, parse: Callable[[str], T]) -> T:
        if not self.interactive:
            return default
        for _ in range(MAX_TRIES):
            try:
                raw = self.input_fn(f"{label} [{default_text}]: ").strip()
            except EOFError:
                print("", file=self.out)
                return default
            if not raw:
                return default
            try:
                return parse(raw)
            except ValueError as e:
                print(f"  {e}", file=self.out)
        raise InitError(f"no valid answer to: {label}")


_YES_NO = {"y": True, "yes": True, "n": False, "no": False}
_MODES = {"off": "off", "warning": "warning", "error": "error"}
_BLOCK = {"none": "none", "p0": "P0", "p1": "P1"}
_STRICT = {"1": 1, "2": 2, "3": 3}


def ask_questions(input_fn: InputFn, out: TextIO, root: Path, *, interactive: bool) -> Answers:
    a = _Asker(input_fn, out, interactive)
    detected = tuple(detect_ignore_patterns(root))
    if interactive:
        print("Press Enter to accept the value in brackets.\n", file=out)
    heading = a.ask("Review heading name", DEFAULT_HEADING, DEFAULT_HEADING, _parse_heading)
    strictness = a.ask("Strictness 1-3 (1 = everything, 3 = P0 only)", "2", 2, _choice(_STRICT))
    block_on = a.ask("Block merges on (none, P0, P1)", "P0", "P0", _choice(_BLOCK))
    pr_template = a.ask("Check pr-template (off, warning, error)", "warning", "warning", _choice(_MODES))
    declared_scope = a.ask(
        "Check declared-scope (off, warning, error)", "warning", "warning", _choice(_MODES)
    )
    ai_scan = a.ask("Add the monthly AI scan, first Monday (yes/no)", "yes", True, _choice(_YES_NO))
    budget = DEFAULT_SCAN_BUDGET
    if ai_scan:
        budget = a.ask(
            "Scan budget in tokens", f"{DEFAULT_SCAN_BUDGET:,}", DEFAULT_SCAN_BUDGET, _parse_budget
        )
    if interactive:
        print("\nIgnore patterns (built-in list plus what was found in this repository):", file=out)
        print("  " + ", ".join(detected), file=out)
    patterns = a.ask(
        "Ignore patterns, comma list ('none' clears)", "keep the list above", detected, _parse_patterns
    )
    instructions = a.ask("Extra instructions for the reviewer, one line", "none", "", _parse_instructions)
    return Answers(
        heading=heading,
        strictness=strictness,
        block_on=block_on,
        pr_template=pr_template,
        declared_scope=declared_scope,
        ai_scan=ai_scan,
        scan_budget=budget,
        ignore_patterns=patterns,
        instructions=instructions,
    )


# ---------- pinning ----------


def parse_ref(ref: str) -> Pin:
    m = _REF_RE.fullmatch(ref.strip())
    if not m:
        raise InitError("--ref must look like owner/repo@<40-character commit sha>")
    return Pin(repo=m.group(1), sha=m.group(2).lower(), comment=REF_COMMENT, resolved=True)


def placeholder_pin() -> Pin:
    return Pin(repo=UPSTREAM, sha=PLACEHOLDER_SHA, comment=PLACEHOLDER_TAG, resolved=False)


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only within the API host, so an Authorization header never leaves it."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        target = urllib.parse.urlparse(newurl)
        if target.scheme != "https" or target.netloc != API_HOST:
            raise urllib.error.HTTPError(newurl, code, "redirect off the API host", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def make_http_get(env: Mapping[str, str]) -> HttpGet:
    """GET a JSON document from the GitHub API. A token is sent only when GITHUB_TOKEN or GH_TOKEN is set."""
    token = (env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or "").strip()
    opener = urllib.request.build_opener(_SameHostRedirect())

    def get(path: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "simpli-review-init",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(API_BASE + path, headers=headers)
        try:
            with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
                body = resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            raise PinError(f"GitHub API answered HTTP {e.code} for {path}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise PinError(f"could not reach the GitHub API ({type(e).__name__})") from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise PinError("GitHub API response too large")
        try:
            return json.loads(body)
        except ValueError:
            raise PinError("GitHub API returned invalid JSON") from None

    return get


def _obj(doc: Any, what: str) -> tuple[str, str]:
    try:
        obj = doc["object"]
        kind, sha = str(obj["type"]), str(obj["sha"]).lower()
    except (KeyError, TypeError):
        raise PinError(f"unexpected response resolving {what}") from None
    if not _SHA_RE.fullmatch(sha):
        raise PinError(f"unexpected sha resolving {what}")
    return kind, sha


def resolve_latest(http_get: HttpGet) -> Pin:
    """latest release -> tag -> commit sha (annotated tags are dereferenced)."""
    release = http_get(f"/repos/{UPSTREAM}/releases/latest")
    tag = release.get("tag_name") if isinstance(release, dict) else None
    if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
        raise PinError("the latest release has no usable tag")
    kind, sha = _obj(http_get(f"/repos/{UPSTREAM}/git/ref/tags/{urllib.parse.quote(tag, safe='')}"), tag)
    for _ in range(5):  # a tag can point at another tag; five levels is more than enough
        if kind == "commit":
            return Pin(repo=UPSTREAM, sha=sha, comment=tag, resolved=True)
        if kind != "tag":
            break
        kind, sha = _obj(http_get(f"/repos/{UPSTREAM}/git/tags/{sha}"), tag)
    raise PinError(f"tag {tag} does not point at a commit")


# ---------- rendering ----------


def yaml_scalar(text: str) -> str:
    if _PLAIN_HEADING_RE.fullmatch(text) and text.lower() not in _YAML_WORDS and not text.endswith(" "):
        return text
    return json.dumps(text, ensure_ascii=False)


def _note(pin: Pin) -> str:
    if pin.resolved:
        return ""
    return (
        f"# TODO: replace {PLACEHOLDER_SHA} with the full commit hash of a simpli-review release and keep\n"
        f"# the version in the trailing comment (`simpli-review init` prints how).\n"
    )


def render_code_review(answers: Answers, pin: Pin) -> str:
    return (
        CODE_REVIEW_TEMPLATE.replace("@@NOTE@@", _note(pin))
        .replace("@@HEADING@@", yaml_scalar(answers.heading))
        .replace("@@USES@@", pin.uses)
    )


def render_ai_scan(pin: Pin) -> str:
    return AI_SCAN_TEMPLATE.replace("@@NOTE@@", _note(pin)).replace("@@USES@@", pin.uses)


def build_config(answers: Answers, *, labels: bool) -> dict[str, Any]:
    """Only the fields that differ from the defaults."""
    cfg: dict[str, Any] = {}
    if answers.strictness != 2:
        cfg["strictness"] = answers.strictness
    if answers.block_on != "none":
        cfg["blockOn"] = answers.block_on
    checks = {
        name: mode
        for name, mode in (("pr-template", answers.pr_template), ("declared-scope", answers.declared_scope))
        if mode != "warning"
    }
    if checks:
        cfg["builtinChecks"] = checks
    if answers.ai_scan and answers.scan_budget != DEFAULT_SCAN_BUDGET:
        cfg["scan"] = {"maxInputTokens": answers.scan_budget}
    if "\n".join(answers.ignore_patterns) != DEFAULT_IGNORE:
        cfg["ignorePatterns"] = "\n".join(answers.ignore_patterns)
    if answers.instructions:
        cfg["instructions"] = answers.instructions
    if labels:
        cfg["disabledLabels"] = ["no-review"]  # makes the no-review label do something
    return cfg


def render_config(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"


def validate_workflow(text: str, *, heading: str | None, pin: Pin) -> None:
    """Parse the generated workflow back and check the parts init is responsible for."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise InitError(f"generated workflow is not valid YAML ({type(e).__name__})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
        raise InitError("generated workflow has no jobs")
    if heading is not None and doc.get("name") != heading:
        raise InitError("generated workflow heading does not round-trip")
    uses = [
        step["uses"]
        for job in doc["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and str(step.get("uses", "")).startswith(pin.repo + "@")
    ]
    if uses != [f"{pin.repo}@{pin.sha}"]:
        raise InitError("generated workflow does not pin the action as expected")


def validate_config(text: str) -> None:
    try:
        raw = json.loads(text)
        Config.model_validate(raw)
    except ValueError as e:  # json.JSONDecodeError and pydantic.ValidationError are ValueErrors
        raise InitError(f"generated config is not valid ({type(e).__name__})") from None


# ---------- writing ----------


def _short_diff(old: str, new: str, path: str) -> list[str]:
    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), f"{path} (existing)", f"{path} (generated)", lineterm="", n=1
        )
    )
    if len(lines) > DIFF_LINES:
        extra = len(lines) - DIFF_LINES
        lines = [*lines[:DIFF_LINES], f"... {extra} more diff lines"]
    return lines


def write_file(root: Path, rel: str, content: str, *, force: bool, dry_run: bool) -> FileResult:
    target = root / rel
    if not target.resolve().is_relative_to(root.resolve()):
        raise InitError(f"{rel} resolves outside the repository; not writing it")
    existing: str | None = None
    if target.exists():
        if not target.is_file():
            raise InitError(f"{rel} exists and is not a file")
        existing = target.read_text(encoding="utf-8", errors="replace")
    if existing is not None and existing.replace("\r\n", "\n") == content:
        return FileResult(rel, "unchanged")
    if existing is not None and not force:
        return FileResult(rel, "would-skip" if dry_run else "skipped", _short_diff(existing, content, rel))
    if dry_run:
        return FileResult(rel, "would-create" if existing is None else "would-overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return FileResult(rel, "created" if existing is None else "overwritten")


# ---------- gh: labels and secret ----------


def run_gh(args: list[str], cwd: Path) -> Proc:
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return Proc(127, "", type(e).__name__)
    return Proc(p.returncode, p.stdout, p.stderr)


def label_command(name: str, description: str, colour: str) -> str:
    return f'gh label create {name} --description "{description}" --color {colour}'


def _names(proc: Proc) -> set[str] | None:
    if proc.returncode != 0:
        return None
    try:
        rows = json.loads(proc.stdout or "[]")
        return {str(r["name"]) for r in rows}
    except (ValueError, KeyError, TypeError):
        return None


def gh_ready(run: RunCmd, root: Path, which: Callable[[str], str | None]) -> str:
    """ "ok", "missing" (gh not installed) or "unauthenticated"."""
    if not which("gh"):
        return "missing"
    return "ok" if run(["gh", "auth", "status"], root).returncode == 0 else "unauthenticated"


def ensure_labels(run: RunCmd, root: Path) -> tuple[list[str], list[str], list[str]]:
    """(created, already existed, failed) label names."""
    existing = _names(run(["gh", "label", "list", "--limit", "1000", "--json", "name"], root)) or set()
    created: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for name, description, colour in LABELS:
        if name in existing:
            skipped.append(name)
            continue
        p = run(["gh", "label", "create", name, "--description", description, "--color", colour], root)
        if p.returncode == 0:
            created.append(name)
        elif "already exists" in p.stderr.lower():
            skipped.append(name)
        else:
            failed.append(name)
    return created, skipped, failed


def check_secret(run: RunCmd, root: Path) -> str:
    """ "found", "missing" or "unknown". Lists secret names only; a value is never requested."""
    names = _names(run(["gh", "secret", "list", "--json", "name"], root))
    if names is not None and SECRET_NAME in names:
        return "found"
    owner = run(["gh", "repo", "view", "--json", "owner", "--jq", ".owner.login"], root)
    org_names = None
    if owner.returncode == 0 and re.fullmatch(r"[A-Za-z0-9-]{1,39}", owner.stdout.strip()):
        org_names = _names(
            run(["gh", "secret", "list", "--org", owner.stdout.strip(), "--json", "name"], root)
        )
    if org_names is not None and SECRET_NAME in org_names:
        return "found"
    return "missing" if names is not None else "unknown"


# ---------- the command ----------


def _p(out: TextIO, text: str = "") -> None:
    print(text, file=out)


def run_init(
    opts: InitOptions,
    *,
    input_fn: InputFn = input,
    http_get: HttpGet | None = None,
    run_cmd: RunCmd = run_gh,
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] | None = None,
    out: TextIO | None = None,
) -> int:
    out = out or sys.stdout
    env = os.environ if env is None else env
    root = opts.root
    try:
        return _run(opts, root, input_fn, http_get or make_http_get(env), run_cmd, which, out)
    except InitError as e:
        _p(out, f"error: {e}")
        return 2


def _run(
    opts: InitOptions,
    root: Path,
    input_fn: InputFn,
    http_get: HttpGet,
    run_cmd: RunCmd,
    which: Callable[[str], str | None],
    out: TextIO,
) -> int:
    if not (root / ".git").exists():
        raise InitError("this is not the root of a git repository; run it from the repository root")
    pin = parse_ref(opts.ref) if opts.ref else None  # fail on a bad --ref before any question

    answers = ask_questions(input_fn, out, root, interactive=not opts.yes)
    pin_problem = ""
    if pin is None:
        try:
            pin = resolve_latest(http_get)
        except PinError as e:
            pin = placeholder_pin()
            pin_problem = str(e)

    labels = not opts.no_labels
    files: list[tuple[str, str]] = []
    review = render_code_review(answers, pin)
    validate_workflow(review, heading=answers.heading, pin=pin)
    files.append((CODE_REVIEW_PATH, review))
    if answers.ai_scan:
        scan = render_ai_scan(pin)
        validate_workflow(scan, heading="AI scan", pin=pin)
        files.append((AI_SCAN_PATH, scan))
    config_text = render_config(build_config(answers, labels=labels))
    validate_config(config_text)
    files.append((CONFIG_PATH, config_text))

    _p(out)
    results = [write_file(root, rel, text, force=opts.force, dry_run=opts.dry_run) for rel, text in files]
    contents = dict(files)
    for r in results:
        _p(out, f"{r.status:<15}{r.path}")
        if opts.dry_run and r.status.startswith("would-") and r.status != "would-skip":
            for line in contents[r.path].splitlines():
                _p(out, f"    | {line}")
        for line in r.diff:
            _p(out, f"    {line}")
    if any(r.status in ("skipped", "would-skip") for r in results):
        _p(out, "Existing files were left alone. Use --force to overwrite them.")

    if pin.resolved:
        _p(out, f"\nPinned {pin.repo}@{pin.sha[:12]} ({pin.comment}).")
    else:
        _p(out, f"\nCould not pin the action ({pin_problem}); wrote the {PLACEHOLDER_SHA} placeholder.")
        _p(out, "Fill it in with the full commit hash of a release, keeping the version in the comment:")
        _p(out, f"  gh api repos/{UPSTREAM}/releases/latest --jq .tag_name")
        _p(out, f"  gh api repos/{UPSTREAM}/git/ref/tags/<tag> --jq '.object.sha'")
        _p(
            out,
            f"  gh api repos/{UPSTREAM}/git/tags/<sha> --jq '.object.sha'  # if that was a tag",
        )

    secret_state = _github_steps(opts, root, run_cmd, which, out, labels)
    _checklist(opts, results, answers, pin, secret_state, labels, out)
    return 0


def _github_steps(
    opts: InitOptions, root: Path, run: RunCmd, which: Callable[[str], str | None], out: TextIO, labels: bool
) -> str:
    """Labels and the secret check. Returns the secret state for the checklist."""
    if opts.dry_run:
        _p(out, "\nDry run: no labels created and no secrets checked. Commands that would be used:")
        if labels:
            for lab in LABELS:
                _p(out, f"  {label_command(*lab)}")
        _p(out, f"  gh secret list   # looks for {SECRET_NAME}; the value is never read")
        return "dry-run"

    state = gh_ready(run, root, which)
    if state != "ok":
        why = "gh is not installed" if state == "missing" else "gh is not signed in (gh auth login)"
        _p(out, f"\n{why}, so nothing was created or checked. Run these yourself:")
        if labels:
            for lab in LABELS:
                _p(out, f"  {label_command(*lab)}")
        _p(out, f"  gh secret set {SECRET_NAME}")
        return "no-gh"

    if labels:
        created, skipped, failed = ensure_labels(run, root)
        _p(out, "\nLabels: " + _join_labels(created, "created", skipped, "already there", failed, "failed"))
        if failed:
            _p(
                out,
                "  Run these yourself (does the repository have a GitHub remote, and do you have access?):",
            )
        for name, description, colour in LABELS:
            if name in failed:
                _p(out, f"  {label_command(name, description, colour)}")
    return check_secret(run, root)


def _join_labels(*groups: Any) -> str:
    parts = [
        f"{', '.join(names)} {word}" for names, word in zip(groups[0::2], groups[1::2], strict=True) if names
    ]
    return "; ".join(parts) or "none"


def _checklist(
    opts: InitOptions,
    results: list[FileResult],
    answers: Answers,
    pin: Pin,
    secret_state: str,
    labels: bool,
    out: TextIO,
) -> None:
    _p(out, "\nChecklist")
    n = 0

    def step(text: str) -> None:
        nonlocal n
        n += 1
        _p(out, f"  {n}. {text}")

    if secret_state == "found":
        _p(out, f"  [done] {SECRET_NAME} secret exists")
    elif secret_state == "missing":
        step(
            f"{SECRET_NAME} was not found. Add it yourself (gh asks for the value; this tool never sees it):"
        )
        _p(out, f"       gh secret set {SECRET_NAME}")
    elif secret_state == "unknown":
        step(f"Could not list secrets. Check that {SECRET_NAME} exists (repo or organization), or run:")
        _p(out, f"       gh secret set {SECRET_NAME}")
    else:
        step(f"Make sure the {SECRET_NAME} secret exists (repo or organization): gh secret set {SECRET_NAME}")
    if not pin.resolved:
        step(f"Replace {PLACEHOLDER_SHA} in the workflow files with a release commit hash (see above).")
    if opts.dry_run:
        step("Run again without --dry-run to write the files.")
        return
    step("Review the new files, commit them and open a pull request.")
    if answers.block_on != "none":
        step(
            "To make a block stop merges, make the `review` job a required status check (branch protection)."
        )
    if labels:
        step(
            "Only maintainers should add the `review-override` label: it lets a PR through a blocking review."
        )
    else:
        step("Create the three labels by hand; only maintainers should add review-override.")
    if answers.ai_scan:
        step(
            "Run Actions > AI scan > Run workflow once and read the spend report before keeping the schedule."
        )
    _p(out, "Open a PR to try it; add the `re-review` label for another review.")
