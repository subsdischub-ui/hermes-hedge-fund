"""Long-history OHLCV foundation: paginated fetch, gap detection, CSV IO.

Thesis: robust validation (CPCV/DSR) needs 2-4 years of bars covering a full
bull/bear/range cycle. This module assembles that history from a paginated
exchange endpoint, then audits it for holes before any backtest trusts it.

No look-ahead by construction: this layer only *collects and orders* raw bars.
Every row is the exchange's record for a closed candle; dedup keeps the row as
returned and sorting is purely chronological — no future bar ever influences a
past one. The client is INJECTED so the whole pipeline is testable offline.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

_COLS = ["ts", "open", "high", "low", "close", "volume"]
_GAP_FACTOR = 1.5  # an inter-bar delta above 1.5x the timeframe counts as a gap


@runtime_checkable
class OHLCVClient(Protocol):
    """Minimal exchange surface we depend on. CCXT (sync or async) satisfies it."""

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> Any:  # pragma: no cover - structural type
        ...


async def _call_fetch(
    client: OHLCVClient, symbol: str, timeframe: str, since: int, limit: int
) -> list[list[float]]:
    """Invoke the injected client, awaiting it iff it returned an awaitable.

    Supports both sync clients (plain list) and async CCXT clients (coroutine),
    so the same fetch loop works in tests and in production.
    """
    result = client.fetch_ohlcv(symbol, timeframe, since, limit)
    if inspect.isawaitable(result):
        return await result
    return result


async def fetch_history(
    symbol: str,
    timeframe: str,
    since_ms: int,
    client: OHLCVClient,
    max_batches: int = 50,
    limit: int = 1000,
) -> pd.DataFrame:
    """Paginate `fetch_ohlcv` from `since_ms` forward into one ordered DataFrame.

    Each batch advances `since` to the last timestamp seen (CCXT-style), so the
    boundary bar reappears in the next batch; we dedup on `ts` to drop it. Stops
    on an empty batch, when no forward progress is made, or at `max_batches`.
    """
    frames: list[pd.DataFrame] = []
    since = int(since_ms)
    last_seen: int | None = None

    for _ in range(max_batches):
        rows = await _call_fetch(client, symbol, timeframe, since, limit)
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=_COLS))
        batch_last = int(rows[-1][0])
        if last_seen is not None and batch_last <= last_seen:
            break  # no forward progress -> exchange has nothing newer
        last_seen = batch_last
        since = batch_last  # boundary bar re-served next call; dedup removes it

    if not frames:
        return pd.DataFrame(columns=_COLS)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def detect_gaps(df: pd.DataFrame, timeframe_ms: int) -> list[tuple[int, int]]:
    """Return (start_ts, end_ts) pairs where consecutive bars are >1.5x apart.

    `start_ts` is the bar before the hole, `end_ts` the bar after it. Operates on
    the already-sorted timestamps; never inspects future bars beyond the immediate
    successor, so it introduces no look-ahead.
    """
    if len(df) < 2:
        return []
    ts = df["ts"].to_numpy()
    threshold = _GAP_FACTOR * timeframe_ms
    gaps: list[tuple[int, int]] = []
    for i in range(1, len(ts)):
        if (ts[i] - ts[i - 1]) > threshold:
            gaps.append((int(ts[i - 1]), int(ts[i])))
    return gaps


# --------------------------------------------------------------------------- #
# Hugging Face deep-history source (linxy/CryptoCoin — Binance klines, 2017→)   #
# --------------------------------------------------------------------------- #
_HF_BASE = "https://huggingface.co/datasets/linxy/CryptoCoin/resolve/main"


def hf_symbol(pair: str) -> str:
    """Map a Hermès pair (e.g. ``BTC/USDC``) to the dataset's Binance symbol
    (``BTCUSDT``). USDT vs USDC is immaterial for OHLCV backtesting."""
    base = pair.split("/", 1)[0].upper()
    return f"{base}USDT"


def normalize_hf_klines(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise linxy/CryptoCoin Binance-kline rows to the canonical OHLCV frame.

    The dataset's ``Open time`` is a UTC datetime string; convert it to epoch-ms
    (matching the rest of this module), keep only the canonical columns, then
    dedup on timestamp (keep first) and sort ascending. No look-ahead: this only
    reshapes already-closed candles.
    """
    df = raw.copy()
    ts = pd.to_datetime(df["Open time"], utc=True)
    df["ts"] = (ts.astype("int64") // 1_000_000).astype("int64")
    df = df[_COLS]
    return df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)


def fetch_hf_history(
    pair: str,
    timeframe: str,
    dest_csv: str | os.PathLike[str] | None = None,
    base_url: str = _HF_BASE,
) -> pd.DataFrame:
    """Download one symbol/timeframe CSV from linxy/CryptoCoin and normalise it.

    Network IO is isolated here (the parsing in ``normalize_hf_klines`` is pure
    and unit-tested). Optionally persists the canonical frame to ``dest_csv``.
    """
    import urllib.request

    url = f"{base_url}/{hf_symbol(pair)}_{timeframe}.csv"
    with urllib.request.urlopen(url, timeout=180) as resp:
        raw = pd.read_csv(resp)
    df = normalize_hf_klines(raw)
    if dest_csv is not None:
        save_history(df, dest_csv)
    return df


def save_history(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    """Persist OHLCV history as CSV (no pyarrow dependency)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def load_history(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load OHLCV history from CSV, preserving the canonical column order."""
    df = pd.read_csv(path)
    return df[_COLS]
