"""Live data sources for the SHADOW new-data sleeves (best-effort, cached, safe).

Surfaces two causally-computed scalars that the shadow skills read off the OHLCV
frame as a column:

  * ``dvol_z_90d``          — Deribit DVOL 90-day z-score (BTC/ETH only). The DVOL
    bar is open-timestamped, so it is lagged by one bar before the z-score (the same
    causality fix applied in research_options/deribit_iv.py).
  * ``stbl_supply_growth_z`` — USDT+USDC 60-day circulating-supply-growth z-score
    (market-wide). Uses ONLY supply data >= ``_STBL_PUB_LAG_D`` days old: the
    point-in-time audit showed DefiLlama serves an as-revised series, and acting on
    settled (>=3-day-old) supply keeps the edge strong (BTC OOS ~1.96 / ETH ~1.47)
    while immunising it against late backfills.

Every fetch is wrapped and returns ``None`` on ANY failure (network, parse, thin
history) and is process-cached with a TTL, so a Deribit/DefiLlama outage can never
wedge or slow the live tick — the shadow skills simply HOLD when the column is
absent. Nothing here ever places an order.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from hermes.logging_setup import get_logger

log = get_logger(__name__)

_DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_DEFILLAMA = "https://stablecoins.llama.fi/stablecoin"
_HTTP_TIMEOUT = 10.0

# DVOL: 90-day z on hourly bars; fetch a little more than 90d of history.
_DVOL_WIN_H = 90 * 24
_DVOL_FETCH_DAYS = 100
_DVOL_TTL_S = 3600.0  # refresh hourly (z-score is slow; this is plenty)

# Stablecoin: 60-day growth, z over a 60-day window, acted on >=3-day-old data.
_STBL_GROWTH_L = 60
_STBL_Z_WIN = 60
_STBL_PUB_LAG_D = 3
_STBL_TTL_S = 6 * 3600.0  # daily series; refresh a few times a day

_cache: dict[str, tuple[float, float | None]] = {}


def _zscore_last(g: np.ndarray, win: int) -> float | None:
    """Last value of the rolling-``win`` z-score of ``g`` (ddof=0), or None."""
    s = pd.Series(g, dtype=float)
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0).replace(0, np.nan)
    z = (s - mu) / sd
    v = z.to_numpy()
    last = v[-1] if len(v) else np.nan
    return float(last) if np.isfinite(last) else None


async def dvol_z_now(base: str) -> float | None:
    """Current causal 90-day DVOL z-score for BTC/ETH (None otherwise/on failure)."""
    ccy = base.upper()
    if ccy not in ("BTC", "ETH"):
        return None
    key = f"dvol_z::{ccy}"
    hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    val: float | None = None
    try:
        import httpx

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - _DVOL_FETCH_DAYS * 86400 * 1000
        rows: list[list] = []
        end = now_ms
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as cx:
            for _ in range(6):  # paginate backward; ~3 calls cover 100 days hourly
                r = await cx.get(_DERIBIT, params={
                    "currency": ccy, "start_timestamp": start_ms,
                    "end_timestamp": end, "resolution": 3600})
                r.raise_for_status()
                res = r.json().get("result", {})
                data = res.get("data", [])
                if not data:
                    break
                rows.extend(data)
                earliest = data[0][0]
                if res.get("continuation") is None or earliest <= start_ms:
                    break
                end = earliest - 1
        if rows:
            df = (pd.DataFrame(rows, columns=["ts", "o", "h", "l", "dvol"])
                  .drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True))
            # Causality: the DVOL bar labelled ts closes at ts+1h, so lag by one bar
            # before the z-score (mirrors research_options/deribit_iv.py).
            dvol = df["dvol"].shift(1).to_numpy(float)
            if len(dvol) > _DVOL_WIN_H:
                val = _zscore_last(dvol, _DVOL_WIN_H)
    except Exception as e:
        log.warning("dvol_z_fetch_failed", base=ccy, error=str(e))
        val = None
    _cache[key] = (time.time() + _DVOL_TTL_S, val)
    return val


async def stbl_supply_growth_z_now() -> float | None:
    """Current causal 60d stablecoin-supply-growth z (USDT+USDC), market-wide.

    Uses only supply data >= ``_STBL_PUB_LAG_D`` days old (point-in-time safety)."""
    key = "stbl_z"
    hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    val: float | None = None
    try:
        import httpx

        async def _supply(sid: int) -> pd.DataFrame:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as cx:
                r = await cx.get(f"{_DEFILLAMA}/{sid}")
                r.raise_for_status()
                tk = r.json().get("tokens", [])
            rows = [{"date": pd.to_datetime(int(x["date"]), unit="s").normalize(),
                     "circ": (x.get("circulating") or {}).get("peggedUSD")} for x in tk]
            return pd.DataFrame(rows)

        usdt = (await _supply(1)).rename(columns={"circ": "usdt"})
        usdc = (await _supply(2)).rename(columns={"circ": "usdc"})
        sup = (usdt.merge(usdc, on="date", how="outer").sort_values("date")
               .reset_index(drop=True))
        sup["tot"] = pd.to_numeric(sup["usdt"], errors="coerce").fillna(0.0) + \
            pd.to_numeric(sup["usdc"], errors="coerce").fillna(0.0)
        # Publication lag: drop the most-recent (possibly-unsettled) rows so we only
        # act on supply that has had time to settle past any backfill.
        if _STBL_PUB_LAG_D > 0 and len(sup) > _STBL_PUB_LAG_D:
            sup = sup.iloc[:-_STBL_PUB_LAG_D]
        x = sup["tot"].to_numpy(float)
        x = np.where(x > 0, x, np.nan)
        if len(x) > _STBL_GROWTH_L + _STBL_Z_WIN:
            g = np.full(len(x), np.nan)
            g[_STBL_GROWTH_L:] = x[_STBL_GROWTH_L:] / x[:-_STBL_GROWTH_L] - 1.0
            val = _zscore_last(g, _STBL_Z_WIN)
    except Exception as e:
        log.warning("stbl_z_fetch_failed", error=str(e))
        val = None
    _cache[key] = (time.time() + _STBL_TTL_S, val)
    return val


def reset_cache() -> None:
    """Clear the process cache (tests / forced refresh)."""
    _cache.clear()
