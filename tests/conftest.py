from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repository with Python and TypeScript code that call each other."""
    files = {
        "app/billing.py": (
            "from app.money import to_cents\n"
            "\n"
            "\n"
            "def charge(amount):\n"
            "    cents = to_cents(amount)\n"
            "    return send(cents)\n"
            "\n"
            "\n"
            "def refund(amount):\n"
            "    return send(-to_cents(amount))\n"
            "\n"
            "\n"
            "def send(cents):\n"
            "    return cents\n"
        ),
        "app/money.py": "def to_cents(dollars):\n    return round(dollars * 100)\n",
        "app/api.py": (
            "from app.billing import charge\n\n\nclass Checkout:\n    def pay(self, total):\n"
            "        return charge(total)\n"
        ),
        "web/src/cart.ts": (
            "import { formatPrice } from './format';\n"
            "\n"
            "export function cartTotal(items: number[]): number {\n"
            "  return items.reduce((a, b) => a + b, 0);\n"
            "}\n"
            "\n"
            "export const showTotal = (items: number[]) => formatPrice(cartTotal(items));\n"
        ),
        "web/src/format.ts": "export function formatPrice(n: number): string {\n  return `$${n}`;\n}\n",
        ".env": "SECRET=do-not-read\n",
        "CLAUDE.md": "Money is integer cents.\n",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def git_repo(repo: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("-c", "user.email=t@example.com", "-c", "user.name=T", "add", "-A")
    git("-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-q", "-m", "initial")
    return repo


# ---------- a scripted stand-in for the Anthropic client ----------


def tool_use(id_: str, name: str, input_: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def message(
    *blocks: SimpleNamespace, stop: str = "tool_use", tokens: tuple[int, int, int, int] | None = None
) -> SimpleNamespace:
    """`tokens` = (input, output, cache read, cache write) for this turn; default 100 / 10 / 50 / 0."""
    i, o, r, w = tokens or (100, 10, 50, 0)
    usage = SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=r,
                            cache_creation_input_tokens=w)  # fmt: skip
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


class _Stream:
    def __init__(self, msg: SimpleNamespace) -> None:
        self.msg = msg

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self.msg


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw: Any) -> _Stream:
        # Snapshot: the loop keeps appending to the same list.
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return _Stream(self.responses.pop(0))
