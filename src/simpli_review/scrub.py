"""Remove secrets and markup tricks from text before it is shown or re-used.

Used for human replies fed back into the prompt (memory.py) and for the
"Prompt for AI agents" blocks (agent_prompt.py), so both strip the same things:
HTML comments (they could carry a fake reviewer marker), email addresses, and
token-like strings.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TOKEN = re.compile(r"\b(?:gh[pousr]_|github_pat_|sk-|xox[bapr]-|AKIA)[\w-]{8,}|\b[A-Za-z0-9_+/=-]{32,}\b")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def redact(text: str) -> str:
    """Same removals as `scrub`, but whitespace and line breaks are kept (for code and multi-line text)."""
    text = _HTML_COMMENT.sub(" ", text)
    text = _EMAIL.sub("[email]", text)
    return _TOKEN.sub("[redacted]", text)


def scrub(text: str, limit: int) -> str:
    """One short line with emails, token-like strings and HTML comments removed."""
    text = " ".join(redact(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S
)
_SECRET_ASSIGN = re.compile(
    r"""(?ix)
    \b([\w.-]*(?:secret|passw(?:or)?d|pwd|token|api[_-]?key|private[_-]?key|access[_-]?key|credential)[\w.-]*)
    (\s*[:=]\s*)
    (['"]?)(?!\[redacted\])[^\s'",;)\]}]{4,}\3
    """
)


def redact_secrets(text: str) -> str:
    """`redact`, plus values assigned to secret-looking names and private key blocks.

    For text that quotes code (scan findings and issues), where a hard-coded secret can show up as
    `password = "hunter2"`, which the token-shape rules in `redact` do not see. It over-redacts
    (`token = get_token()` loses its right-hand side) because a missed secret in an issue is worse.
    """
    text = _PRIVATE_KEY.sub("[private key redacted]", text)
    text = _SECRET_ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
    return redact(text)


def api_error_text(e: Exception) -> str:
    """The API's own error type and message (never the request), scrubbed and capped."""
    detail = ""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            detail = f"{err.get('type', '')}: {err.get('message', '')}".strip(": ")
    status = getattr(e, "status_code", None)
    head = f"Anthropic API error ({type(e).__name__}{f', HTTP {status}' if status else ''})"
    return head + (f": {redact_secrets(detail)[:300]}" if detail else "")
