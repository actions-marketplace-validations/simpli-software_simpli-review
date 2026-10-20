"""A prioritised map of the repository, built from the code index, to seed a full-repository scan.

A scan has no diff to start from. Without a map the model spends its first turns listing directories, and
a budgeted scan can end before it reaches the code that matters. The map puts the risky code first:
authentication and sessions, tenant isolation, webhooks, payments, migrations, HTTP routes and workers.
It is plain heuristics over paths and source lines (no model call), so it is cheap and the same on every run.
The model still has the tools to read anything the map leaves out.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from simpli_review.index import SOURCE_SUFFIXES, RepoIndex
from simpli_review.scrub import redact

MAX_PRIORITY_FILES = 60
MAX_ENTRY_POINTS = 200
MAX_MIGRATIONS = 60
MAX_FILE_LIST_CHARS = 60_000
MAX_LINE_CHARS = 110

# (tag, weight, path pattern). A file's score is its highest weight; every matching tag is shown.
_PATH_TAGS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    (
        "auth",
        100,
        re.compile(
            r"(^|/)(auth\w*|login|logout|sessions?|jwt|oauth\w*|sso|saml|mfa|totp|rbac|permissions?|acl|"
            r"security|passwords?|api_?keys?|tokens?|csrf|cors)(/|[._-]|$)"
        ),
    ),
    (
        "tenancy",
        95,
        re.compile(
            r"(^|/)(tenants?|tenancy|rls|multi_?tenant\w*|workspaces?|organi[sz]ations?|compan(y|ies))(/|[._-]|$)"
        ),
    ),
    ("webhook", 90, re.compile(r"webhooks?|callbacks?|ipn(/|[._-]|$)")),
    (
        "payments",
        85,
        re.compile(
            r"stripe|payments?|billing|invoices?|checkout|payouts?|refunds?|subscriptions?|ledger|wallet"
        ),
    ),
    ("migration", 80, re.compile(r"(^|/)(migrations?|alembic|versions)(/|$)|\.sql$")),
    (
        "route",
        75,
        re.compile(
            r"(^|/)(routes?|routers?|api|views|handlers?|endpoints?|controllers?|resolvers?)(/|[._-]|$)"
        ),
    ),
    (
        "worker",
        70,
        re.compile(r"(^|/)(workers?|tasks?|jobs?|celery|cron|queues?|consumers?|scheduler)(/|[._-]|$)"),
    ),
    ("config", 60, re.compile(r"(^|/)(settings|config|configs|middleware|deps|dependencies)(/|[._-]|$)")),
    (
        "infra",
        40,
        re.compile(
            r"(^|/)(dockerfile|docker-compose[\w.-]*|nginx[\w./-]*|\.github/workflows/[\w.-]+|[\w-]+\.tf)$"
        ),
    ),
)

_PY_ENTRY = re.compile(
    r"^\s*@\s*(?:[\w.]+\.)?(?:get|post|put|patch|delete|head|options|route|websocket|api_route|task|"
    r"shared_task|periodic_task|on_event|webhook)\b\s*(?:\(|$)|\b(?:APIRouter|Blueprint|include_router)\("
)
_JS_ENTRY = re.compile(
    r"\b(?:app|router|server|api|fastify|route)\.(?:get|post|put|patch|delete|all|use|route|head)\s*\("
    r"|^\s*export\s+(?:async\s+)?function\s+(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"
    r"|export\s+default\s+(?:async\s+)?function\s+handler\b|\bexports?\.handler\b"
    r"|\bcron\.schedule\(|\bnew\s+(?:Worker|Queue)\(|^\s*@(?:Get|Post|Put|Patch|Delete|Controller|Cron)\("
)


@dataclass
class MapEntry:
    path: str
    lines: int
    tags: list[str]
    score: int
    entry_points: list[tuple[int, str]] = field(default_factory=list)


def classify(path: str) -> tuple[list[str], int]:
    low = path.lower()
    tags = [tag for tag, _w, rx in _PATH_TAGS if rx.search(low)]
    score = max((w for tag, w, _rx in _PATH_TAGS if tag in tags), default=0)
    return tags, score


def build_entries(index: RepoIndex) -> list[MapEntry]:
    entries: list[MapEntry] = []
    for path, info in index.files.items():
        tags, score = classify(path)
        suffix = path[path.rfind(".") :].lower() if "." in path else ""
        points: list[tuple[int, str]] = []
        if suffix in SOURCE_SUFFIXES:
            rx = _PY_ENTRY if suffix == ".py" else _JS_ENTRY
            for n, line in enumerate(info.text.splitlines(), start=1):
                if rx.search(line):
                    points.append((n, redact(line.strip())[:MAX_LINE_CHARS]))
        if points:
            tags = [*tags, "entry"] if "entry" not in tags else tags
            score += min(len(points), 10) * 2 + (30 if score == 0 else 0)  # a route file outranks plain code
        entries.append(MapEntry(path, info.text.count("\n") + 1, tags, score, points))
    entries.sort(key=lambda e: (-e.score, e.path))
    return entries


def repo_map(index: RepoIndex) -> str:
    entries = build_entries(index)
    if not entries:
        return "(no indexed files - use list_directory to explore)"
    out: list[str] = []

    top = [e for e in entries if e.score > 0][:MAX_PRIORITY_FILES]
    out.append(
        "## Read these first (highest risk first)\n"
        + (
            "\n".join(f"- {e.path} ({e.lines} lines) [{', '.join(e.tags)}]" for e in top)
            or "(no file matched a risky-path pattern - start from the entry points and the file list)"
        )
    )

    points = [(e.path, n, text) for e in entries for n, text in e.entry_points]
    shown = points[:MAX_ENTRY_POINTS]
    out.append(
        "## Entry points: routes, handlers, workers, webhooks (file:line  source line)\n"
        + (
            "\n".join(f"{p}:{n}  {t}" for p, n, t in shown)
            or "(none detected - search for the framework's routing calls)"
        )
        + (f"\n... {len(points) - len(shown)} more; use search_code" if len(points) > len(shown) else "")
    )

    migrations = sorted(e.path for e in entries if "migration" in e.tags)
    if migrations:
        out.append(
            "## Migrations (check tenant columns, RLS, grants, destructive changes)\n"
            + "\n".join(f"- {p}" for p in migrations[:MAX_MIGRATIONS])
            + (f"\n... {len(migrations) - MAX_MIGRATIONS} more" if len(migrations) > MAX_MIGRATIONS else "")
        )

    out.append(_file_list(entries))
    return "\n\n".join(out)


def _file_list(entries: list[MapEntry]) -> str:
    """Every indexed file, grouped by directory (name and line count), capped in size."""
    by_dir: dict[str, list[MapEntry]] = defaultdict(list)
    for e in entries:
        folder, _, _name = e.path.rpartition("/")
        by_dir[folder].append(e)
    lines: list[str] = []
    used = 0
    omitted = 0
    for folder in sorted(by_dir):
        files = sorted(by_dir[folder], key=lambda e: e.path)
        row = f"{folder or '.'}/: " + ", ".join(f"{e.path.rpartition('/')[2]} ({e.lines})" for e in files)
        if used + len(row) > MAX_FILE_LIST_CHARS:
            omitted += len(files)
            continue
        used += len(row)
        lines.append(row)
    tail = f"\n... {omitted} more file(s) in directories not listed; use list_directory." if omitted else ""
    return (
        f"## All indexed files ({len(entries)}; name and line count, by directory)\n"
        + "\n".join(lines)
        + tail
    )
