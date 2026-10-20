"""Price table for the cost estimate in the scan report.

Prices are USD per million tokens, kept as integer cents per million tokens so the arithmetic is exact
(money is never a float here). The result is an ESTIMATE: it prices the tokens the API reported at the list
prices below. It ignores discounts, taxes, a different model chosen by the server-side refusal fallback, and
price changes after PRICES_AS_OF.

Source: the `claude-api` skill's model table (cached 2026-10-06) and its prompt-caching notes:
  - claude-opus-5-5:   $4.00 input, $20.00 output, cache read $0.20 (0.05x input)
  - claude-sonnet-5-5: $2.00 input, $10.00 output, cache read $0.20 (0.10x input)
  - cache write: 1.25x input for the 5-minute TTL (what `cache_control: {"type": "ephemeral"}` uses here).
Re-check https://platform.claude.com/docs/en/about-claude/pricing when you change a model or this table.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_AS_OF = "2026-10-06"
_MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    """Cents per million tokens."""

    input: int
    output: int
    cache_read: int
    cache_write: int  # 5-minute TTL (1.25x input)


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=400, output=2000, cache_read=20, cache_write=500),
    "claude-sonnet-5-5": Price(input=200, output=1000, cache_read=20, cache_write=250),
}


def estimate_micros(model: str, usage: dict[str, int]) -> int | None:
    """Estimated cost in millionths of a US dollar (integer), or None for a model not in the table.

    cents-per-million-tokens x tokens / 1e6 = cents; x 10_000 = millionths of a dollar. Rounded half up.
    """
    price = PRICES.get(model)
    if price is None:
        return None
    cents_x_million = (
        usage.get("input", 0) * price.input
        + usage.get("output", 0) * price.output
        + usage.get("cache_read", 0) * price.cache_read
        + usage.get("cache_write", 0) * price.cache_write
    )
    micros_x_million = cents_x_million * 10_000
    return (micros_x_million + _MILLION // 2) // _MILLION


def format_usd(micros: int | None) -> str:
    """'$1.23' for a whole number of cents rounded half up; 'unknown' when there is no estimate."""
    if micros is None:
        return "unknown (model not in the price table)"
    cents = (micros + 5_000) // 10_000
    return f"${cents // 100}.{cents % 100:02d}"
