"""STREAM 10 v-next — data-provenance guardrails (reusable, tested).

S0 surfaced two silent traps that no green test catches on its own — a backtest can pass while
training on the wrong data. These helpers make the traps loud:

  P1  ``check_funding_provenance`` / ``assert_deep_funding`` — flag any SQL that reads the SHALLOW
      ``binance_funding`` stub (exactly ~200 rows/symbol, forward-only from 2026-04 — the
      ``/fapi/v1/fundingRate`` startTime=0 clamp) instead of the DEEP ``binance_funding_native``
      view (3 459-7 453 rows/symbol back to inception). [VÉRIFIÉ S0/COVERAGE.md, root cause
      FUNDING_DEEP_START_MS=1 in binance_extended.py].

  P2  ``detect_symbol_format`` / ``validate_symbol_join`` / ``normalize_symbol`` /
      ``assert_joinable`` — the lake mixes three symbol spellings: 'BTC/USDT'
      (binance_klines_4h, binance_funding_native), bare 'BTC' (coinglass_multi_*), and 'BTCUSDT'
      (coinglass_funding_rate, coinglass_open_interest, binance_funding_info). A cross-format join
      silently yields zero rows; validate (and normalize) before joining.

Pure functions, no side effects at import.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

__all__ = [
    "DEEP_FUNDING_VIEWS",
    "SHALLOW_FUNDING_VIEWS",
    "ProvenanceError",
    "assert_deep_funding",
    "assert_joinable",
    "check_funding_provenance",
    "detect_symbol_format",
    "normalize_symbol",
    "validate_symbol_join",
]


class ProvenanceError(ValueError):
    """Raised when a query/join would silently use the wrong data provenance."""


# =========================================================================== #
# P1 — shallow vs deep funding view
# =========================================================================== #
# The forward-only stub (~200 rows/sym) vs the deep, inception-to-now views.
SHALLOW_FUNDING_VIEWS: tuple[str, ...] = ("binance_funding",)
DEEP_FUNDING_VIEWS: tuple[str, ...] = (
    "binance_funding_native",
    "binance_funding_on_4h",
    "coinglass_funding_rate",
    "coinglass_multi_funding_rate",
)

# Match the BARE token 'binance_funding' but NOT the deep views that merely contain it as a prefix
# (binance_funding_native / _on_4h / _info). A word boundary already excludes a following '_', the
# explicit negative lookahead documents the intent.
_SHALLOW_FUNDING_RE = re.compile(r"\bbinance_funding\b(?!_)")


def check_funding_provenance(sql: str) -> dict:
    """Inspect a SQL string for use of the shallow ``binance_funding`` stub.

    Returns ``{flagged, shallow_views_found, recommendation}``. ``flagged`` is True iff the query
    reads the forward-only stub (a backtest on it sees only ~2 months of funding history).
    """
    found = sorted({m.group(0) for m in _SHALLOW_FUNDING_RE.finditer(sql)})
    flagged = bool(found)
    if flagged:
        recommendation = (
            "Reads the SHALLOW binance_funding stub (~200 rows/sym, forward-only since 2026-04 — the "
            "/fapi/v1/fundingRate startTime=0 clamp). Use binance_funding_native (deep, since inception) "
            "or binance_funding_on_4h (causal 4h join); for cross-venue use coinglass_multi_funding_rate."
        )
    else:
        recommendation = "OK — no shallow binance_funding stub referenced."
    return {"flagged": flagged, "shallow_views_found": found, "recommendation": recommendation}


def assert_deep_funding(sql: str) -> None:
    """Raise :class:`ProvenanceError` if ``sql`` reads the shallow ``binance_funding`` stub."""
    res = check_funding_provenance(sql)
    if res["flagged"]:
        raise ProvenanceError(res["recommendation"])


# =========================================================================== #
# P2 — symbol-format join validation
# =========================================================================== #
_QUOTES: tuple[str, ...] = ("USDT", "USDC", "BUSD", "USD")


def detect_symbol_format(symbol: str) -> str:
    """Classify a crypto-lake symbol spelling: ``'slash'`` (BTC/USDT), ``'concat'`` (BTCUSDT) or
    ``'bare'`` (BTC).

    SCOPE: assumes USD-family-quoted CRYPTO spellings (the lake's BTC/USDT · BTC · BTCUSDT trio).
    It is NOT valid for FX cross / USD-BASE pairs — 'USDJPY' classifies as 'bare' and 'EURUSD' as
    'concat' with base 'EUR'; do not reuse on the FX/metals book without a quote map.
    """
    s = symbol.strip()
    if "/" in s:
        return "slash"
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return "concat"
    return "bare"


def _dominant_format(symbols: Sequence[str]) -> str:
    formats = [detect_symbol_format(s) for s in symbols]
    if not formats:
        return "bare"
    # most common format; Counter.most_common breaks ties by first insertion (deterministic)
    return Counter(formats).most_common(1)[0][0]


def normalize_symbol(symbol: str, to: str, quote: str = "USDT") -> str:
    """Convert a symbol between ``'slash'`` / ``'concat'`` / ``'bare'`` formats.

    The base is extracted from any input format; the chosen quote (default USDT) is used when a
    quote is needed and the input did not carry one (bare → slash/concat).
    """
    s = symbol.strip()
    fmt = detect_symbol_format(s)
    if fmt == "slash":
        base, _, q = s.partition("/")
        quote = q or quote
    elif fmt == "concat":
        for cand in _QUOTES:
            if s.endswith(cand) and len(s) > len(cand):
                base, quote = s[: -len(cand)], cand
                break
        else:  # pragma: no cover - detect_symbol_format guarantees a match
            base = s
    else:  # bare
        base = s
    if to == "bare":
        return base
    if to == "slash":
        return f"{base}/{quote}"
    if to == "concat":
        return f"{base}{quote}"
    raise ValueError(f"unknown target format {to!r}; use 'slash', 'concat' or 'bare'.")


def validate_symbol_join(left_symbols: Sequence[str], right_symbols: Sequence[str]) -> dict:
    """Validate that two symbol lists can be joined directly.

    Returns ``{left_format, right_format, compatible, n_overlap, recommendation}``. ``compatible``
    is False when the two sides use different spellings — a direct join would silently yield ~0
    rows; the caller must ``normalize_symbol`` one side first.
    """
    lf = _dominant_format(left_symbols)
    rf = _dominant_format(right_symbols)
    overlap = set(left_symbols) & set(right_symbols)
    compatible = lf == rf
    if compatible:
        recommendation = f"OK — both sides use the '{lf}' format; join directly."
    else:
        recommendation = (
            f"INCOMPATIBLE symbol formats: left='{lf}', right='{rf}'. A direct join yields ~0 rows. "
            f"Normalize one side first, e.g. normalize_symbol(sym, '{rf}')."
        )
    return {
        "left_format": lf,
        "right_format": rf,
        "compatible": compatible,
        "n_overlap": len(overlap),
        "recommendation": recommendation,
    }


def assert_joinable(left_symbols: Sequence[str], right_symbols: Sequence[str]) -> None:
    """Raise :class:`ProvenanceError` if the two symbol lists use incompatible formats."""
    res = validate_symbol_join(left_symbols, right_symbols)
    if not res["compatible"]:
        raise ProvenanceError(res["recommendation"])
