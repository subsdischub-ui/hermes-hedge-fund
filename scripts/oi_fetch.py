"""Fetch Binance futures/data OI + crowd-positioning series (1h) for 7 pairs.

Endpoints (base https://fapi.binance.com/futures/data), period=1h, max retention
~30d but limit=500 returns only the most recent ~21 days. We paginate backwards
with endTime to grab everything available, then merge per symbol.

Series per symbol:
  - openInterestHist:           sumOpenInterest, sumOpenInterestValue
  - globalLongShortAccountRatio: longShortRatio (all accounts)
  - topLongShortPositionRatio:  longShortRatio (top traders, by position)
  - takerlongshortRatio:        buySellRatio (aggressive taker buy/sell vol)

We also (re)fetch aligned 1h OHLCV for the same window so price and positioning
share a timestamp grid. Output: data/oi/{COIN}_oi.csv with a unified hourly index.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import ccxt
import pandas as pd
import requests

BASE = "https://fapi.binance.com/futures/data"
OUT = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/oi")
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]
PERIOD = "1h"
HOUR = 3600_000


def _get(ep: str, sym: str, end: int | None = None) -> list[dict]:
    params = {"symbol": sym, "period": PERIOD, "limit": 500}
    if end is not None:
        params["endTime"] = end
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}/{ep}", params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.0)
        except Exception:
            time.sleep(1.5)
    return []


def paginate(ep: str, sym: str) -> pd.DataFrame:
    """Walk backwards via endTime until no new data (retention boundary)."""
    rows: dict[int, dict] = {}
    end = None
    for _ in range(6):  # 6*500 way more than retention; will stop early
        chunk = _get(ep, sym, end)
        if not isinstance(chunk, list) or not chunk:
            break
        new = 0
        for d in chunk:
            t = int(d["timestamp"])
            if t not in rows:
                rows[t] = d
                new += 1
        ts_min = min(int(d["timestamp"]) for d in chunk)
        if new == 0:
            break
        end = ts_min - 1
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(sorted(rows.values(), key=lambda d: int(d["timestamp"])))
    df["ts"] = df["timestamp"].astype("int64")
    return df


def fetch_ohlcv_window(coin: str, since_ms: int) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    sym = f"{coin}/USDT"
    out = []
    since = since_ms
    now = ex.milliseconds()
    while since < now:
        try:
            o = ex.fetch_ohlcv(sym, "1h", since=since, limit=1000)
        except Exception:
            time.sleep(1.0)
            continue
        if not o:
            break
        out.extend(o)
        since = o[-1][0] + HOUR
        if len(o) < 2:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def build(coin: str) -> pd.DataFrame | None:
    sym = f"{coin}USDT"
    oi = paginate("openInterestHist", sym)
    gl = paginate("globalLongShortAccountRatio", sym)
    tp = paginate("topLongShortPositionRatio", sym)
    tk = paginate("takerlongshortRatio", sym)
    if oi.empty:
        print(f"  {coin}: no OI data", file=sys.stderr)
        return None

    base = oi[["ts"]].copy()
    base["oi"] = oi["sumOpenInterest"].astype(float).values
    base["oi_val"] = oi["sumOpenInterestValue"].astype(float).values

    def merge(df, col_src, col_dst):
        nonlocal base
        if df.empty:
            base[col_dst] = float("nan")
            return
        m = df[["ts", col_src]].copy()
        m[col_dst] = m[col_src].astype(float)
        base = base.merge(m[["ts", col_dst]], on="ts", how="left")

    merge(gl, "longShortRatio", "gl_lsr")
    merge(tp, "longShortRatio", "top_lsr")
    merge(tk, "buySellRatio", "taker_bsr")

    span0 = int(base["ts"].min())
    px = fetch_ohlcv_window(coin, span0 - 5 * HOUR)
    base = base.merge(px[["ts", "open", "high", "low", "close", "volume"]], on="ts", how="left")
    # forward-fill positioning across any gaps, then drop rows missing price
    base = base.sort_values("ts").reset_index(drop=True)
    base[["oi", "oi_val", "gl_lsr", "top_lsr", "taker_bsr"]] = base[
        ["oi", "oi_val", "gl_lsr", "top_lsr", "taker_bsr"]
    ].ffill()
    base = base.dropna(subset=["close"]).reset_index(drop=True)
    return base


def main():
    for coin in PAIRS:
        df = build(coin)
        if df is None or df.empty:
            print(f"{coin}: SKIP", file=sys.stderr)
            continue
        p = OUT / f"{coin}_oi.csv"
        df.to_csv(p, index=False)
        t0 = pd.to_datetime(df["ts"].iloc[0], unit="ms")
        t1 = pd.to_datetime(df["ts"].iloc[-1], unit="ms")
        days = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86400000
        print(f"{coin}: {len(df)} rows  {t0} -> {t1}  ({days:.1f}d)  "
              f"nan_gl={df['gl_lsr'].isna().sum()} nan_top={df['top_lsr'].isna().sum()} "
              f"nan_tk={df['taker_bsr'].isna().sum()}", file=sys.stderr)


if __name__ == "__main__":
    main()
