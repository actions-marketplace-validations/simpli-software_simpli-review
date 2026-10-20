from __future__ import annotations

from pathlib import Path

import pytest

from simpli_review.index import RepoIndex
from simpli_review.tools import Toolbox, ToolError


def test_python_graph_definitions_callers_callees(repo: Path):
    idx = RepoIndex(repo)
    (charge,) = idx.definitions("charge")
    assert (charge.path, charge.start, charge.end, charge.kind) == ("app/billing.py", 4, 6, "function")
    callers = {s.qualname for s in idx.callers_of("charge")}
    assert callers == {"Checkout.pay"}
    callees = {s.qualname for s in idx.callees_of(charge)}
    assert callees == {"to_cents", "send"}
    assert {s.qualname for s in idx.callers_of("to_cents")} == {"charge", "refund"}


def test_symbols_touching_changed_lines_innermost_first(repo: Path):
    idx = RepoIndex(repo)
    assert [s.qualname for s in idx.symbols_touching("app/api.py", {6})] == ["Checkout.pay"]
    assert idx.symbols_touching("app/billing.py", {2}) == []


def test_typescript_graph(repo: Path):
    idx = RepoIndex(repo)
    (total,) = idx.definitions("cartTotal")
    assert (total.path, total.start, total.end) == ("web/src/cart.ts", 3, 5)
    assert {s.qualname for s in idx.callers_of("cartTotal")} == {"showTotal"}
    assert {s.qualname for s in idx.callers_of("formatPrice")} == {"showTotal"}


def test_similar_functions_and_importers(repo: Path):
    idx = RepoIndex(repo)
    (refund,) = idx.definitions("refund")
    assert idx.similar_to(refund) == []  # no shared name token
    assert idx.importers_of("app/billing.py") == ["app/api.py"]
    assert idx.importers_of("web/src/format.ts") == ["web/src/cart.ts"]


def test_search_and_ignore(repo: Path):
    idx = RepoIndex(repo, ignored=lambda p: p.startswith("web/"))
    assert all(not p.startswith("web/") for p in idx.files)
    hits = idx.search(r"to_cents\(")
    assert {(p, ln) for p, ln, _ in hits} == {
        ("app/billing.py", 5),
        ("app/billing.py", 10),
        ("app/money.py", 1),
    }


def test_read_file_numbers_lines_and_pages(repo: Path):
    tb = Toolbox(repo, RepoIndex(repo))
    out = tb.run("read_file", {"path": "app/billing.py", "start_line": 4, "end_line": 5})
    assert "    4  def charge(amount):" in out and "    5      cents = to_cents(amount)" in out
    assert "more lines" in out
    assert "app/billing.py" in tb.files_read


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", ".git/config", ".env", "keys/server.pem"])
def test_paths_outside_repo_git_and_secrets_are_refused(repo: Path, path: str):
    tb = Toolbox(repo, RepoIndex(repo))
    with pytest.raises(ToolError):
        tb.run("read_file", {"path": path})


def test_env_file_is_not_indexed_or_searchable(repo: Path):
    idx = RepoIndex(repo)
    assert ".env" not in idx.files
    assert idx.search("do-not-read") == []


def test_find_tools_and_listing(repo: Path):
    tb = Toolbox(repo, RepoIndex(repo))
    assert "app/billing.py:4-6  function charge" in tb.run("find_definition", {"name": "charge"})
    assert "Checkout.pay" in tb.run("find_callers", {"name": "charge"})
    assert "No indexed function calls" in tb.run("find_callers", {"name": "nothing"})
    listing = tb.run("list_directory", {"path": ""}).splitlines()
    assert listing[:2] == ["app/", "web/"]
    with pytest.raises(ToolError):
        tb.run("search_code", {"pattern": "("})


def test_git_history(git_repo: Path):
    tb = Toolbox(git_repo, RepoIndex(git_repo))
    assert "initial" in tb.run("git_history", {"path": "app/money.py"})
