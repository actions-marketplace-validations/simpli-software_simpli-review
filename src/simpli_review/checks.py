"""Pre-merge checks, shown as a table in the summary comment.

Two are built in:

- `pr-template` is deterministic and uses no model. If the reviewed repository has
  `.github/pull_request_template.md`, every `##` heading in it must appear in the PR
  body with real content (not the template's own placeholder text, a comment, or TODO/TBD).
- `declared-scope` is evaluated by the model: it compares the PR description with the diff.

The team's own checks (config `checks`) are also evaluated by the model, with the same
read-only tools as the rest of the review. The model reports its verdicts in the `checks`
array of `submit_review`; a check it fails to report on is shown as inconclusive.

A check in `error` mode that fails blocks the merge (see blocking.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from simpli_review.config import TEMPLATE_PATH, Config
from simpli_review.config_source import ConfigSourceError, DirSource, FileSource
from simpli_review.models import CheckMode, CheckRow, ModelCheck
from simpli_review.scrub import scrub

PR_TEMPLATE = "pr-template"
DECLARED_SCOPE = "declared-scope"
MAX_TEMPLATE_CHARS = 20_000
MAX_REASON_CHARS = 200

DECLARED_SCOPE_INSTRUCTIONS = (
    "Compare the PR description with the diff. Fail when the diff contains changes the description does "
    "not declare: new or changed database migrations; new or changed environment variables or "
    "configuration settings; new or changed public or unauthenticated endpoints; and changes clearly "
    "outside the stated purpose of the PR. Name each undeclared change in the reason. Pass when "
    "everything in the diff is covered by the description."
)

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
# Lines that say nothing: a bare bullet, an empty checkbox, a rule.
_EMPTY_LINE = re.compile(r"^[\s>*+\-_=]*(\[\s?\])?[\s>*+\-_=]*$")
_TODO = re.compile(r"^[\W_]*(todo|tbd)[\W_]*$", re.I)
_NONE = re.compile(r"^[\W_]*(none|n/?a)[\W_]*$", re.I)


@dataclass
class Section:
    level: int
    name: str  # as written
    key: str  # normalised, for matching
    lines: list[str]


# ---------- the pr-template check ----------


def read_pr_template(source: FileSource | Path) -> str | None:
    """The reviewed repository's PR template (from the trusted source), or None when it has none."""
    if isinstance(source, Path):
        source = DirSource(source)
    try:
        text = source.read(TEMPLATE_PATH)
    except ConfigSourceError:
        return None
    return None if text is None else text[:MAX_TEMPLATE_CHARS]


def parse_sections(text: str) -> list[Section]:
    """Headings with the lines under them. HTML comments are dropped; headings inside code fences are not."""
    sections: list[Section] = []
    in_fence = False
    for raw in _COMMENT.sub("", text).splitlines():
        is_fence = bool(_FENCE.match(raw))
        if is_fence:
            in_fence = not in_fence
        m = None if (in_fence or is_fence) else _HEADING.match(raw)
        if m:
            name = m.group(2).strip()
            sections.append(Section(len(m.group(1)), name, _key(name), []))
        elif sections:
            sections[-1].lines.append(raw)
    return sections


def _key(name: str) -> str:
    return " ".join(re.sub(r"[*_`:]", " ", name).lower().split())


def _norm(line: str) -> str:
    return " ".join(line.lower().split())


def _content(sections: list[Section], index: int) -> list[str]:
    """Lines under sections[index] with nested sub-headings, up to a heading of the same or higher level."""
    me = sections[index]
    lines = list(me.lines)
    for later in sections[index + 1 :]:
        if later.level <= me.level:
            break
        lines += [f"{'#' * later.level} {later.name}", *later.lines]
    return [ln for ln in lines if ln.strip()]


def _is_filled(pr_lines: list[str], template_lines: set[str]) -> bool:
    for line in pr_lines:
        if _NONE.match(line.strip()):
            return True  # "None" is an answer, even if the template shows it as an example
        if _EMPTY_LINE.match(line) or _TODO.match(line.strip()) or _norm(line) in template_lines:
            continue
        return True
    return False


def pr_template_row(template: str, pr_body: str, mode: CheckMode) -> CheckRow:
    template_sections = parse_sections(template)
    wanted = [i for i, s in enumerate(template_sections) if s.level == 2]
    if not wanted:
        return CheckRow(PR_TEMPLATE, "pass", mode, "The PR template has no `##` headings to fill in.")
    body = parse_sections(pr_body)
    missing: list[str] = []
    empty: list[str] = []
    for w in wanted:
        want = template_sections[w]
        placeholders = {_norm(ln) for ln in _content(template_sections, w)}
        found = [i for i, s in enumerate(body) if s.key == want.key]
        if not found:
            missing.append(want.name)
        elif not any(_is_filled(_content(body, i), placeholders) for i in found):
            empty.append(want.name)
    if not missing and not empty:
        return CheckRow(PR_TEMPLATE, "pass", mode, f"All {len(wanted)} template sections are filled in.")
    parts = []
    if missing:
        parts.append("Missing: " + ", ".join(missing))
    if empty:
        parts.append("Empty or still the template text: " + ", ".join(empty))
    return CheckRow(PR_TEMPLATE, "fail", mode, scrub(". ".join(parts) + ".", MAX_REASON_CHARS))


# ---------- the model-evaluated checks ----------


def model_checks(config: Config) -> list[tuple[str, str, CheckMode]]:
    """(name, instructions, mode) of every model-evaluated check that is on, in table order."""
    out: list[tuple[str, str, CheckMode]] = []
    scope_mode = config.builtin_checks.declared_scope
    if scope_mode != "off":
        out.append((DECLARED_SCOPE, DECLARED_SCOPE_INSTRUCTIONS, scope_mode))
    out += [(c.name, c.instructions, c.mode) for c in config.checks if c.mode != "off"]
    return out


def prompt_section(config: Config) -> str:
    wanted = model_checks(config)
    if not wanted:
        return ""
    lines = [
        "## Pre-merge checks to evaluate",
        "In `checks`, report exactly one entry per check below, using its name exactly. `result` is "
        "pass, fail or inconclusive; use inconclusive when the diff and your tools cannot settle it, "
        "never guess. `reason` is one line (under 200 characters) naming the concrete thing you found. "
        "Checks are separate from `findings`: add a finding only if the problem is also a real defect.",
        "",
    ]
    lines += [f"- `{name}`: {' '.join(instructions.split())}" for name, instructions, _ in wanted]
    return "\n".join(lines)


def build_rows(
    config: Config, reported: list[ModelCheck], template: str | None, pr_body: str | None
) -> list[CheckRow]:
    """The table: pr-template, declared-scope, then the team's checks. Checks that are off are left out.

    `pr_body=None` means there is no PR description (a local review), so pr-template does not apply.
    """
    rows: list[CheckRow] = []
    template_mode = config.builtin_checks.pr_template
    if template is not None and pr_body is not None and template_mode != "off":
        rows.append(pr_template_row(template, pr_body, template_mode))
    verdicts: dict[str, ModelCheck] = {}
    for c in reported:
        verdicts.setdefault(c.name.strip().lower(), c)  # the first report for a name wins
    for name, _instructions, mode in model_checks(config):
        got = verdicts.get(name.strip().lower())
        if got is None:
            rows.append(CheckRow(name, "inconclusive", mode, "The reviewer did not report a result."))
        else:
            rows.append(CheckRow(name, got.result, mode, scrub(got.reason, MAX_REASON_CHARS)))
    return rows


def failed_error_checks(rows: list[CheckRow]) -> list[CheckRow]:
    return [r for r in rows if r.mode == "error" and r.result == "fail"]
