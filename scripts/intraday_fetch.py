"""Fetch full available 1h/2h/15m OHLCV history for the 7 pairs via ccxt (Binance spot),
plus funding from Binance USDT-M fapi. Cached to data/intraday/*.csv.

Pagination: ccxt fetch_ohlcv limit=1000, advance `since` by last_ts+tf_ms, loop until now.
Do NOT break on a short batch — Binance caps ~1000/call.
"""
from __future__ import annotations

import sys
import time
import json
import urllib.request
from pathlib import Path

import ccxt
import pandas as pd

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "SUI/USDT", "ARB/USDT", "WIF/USDT"]
TF_MS = {"15m": 900_000, "1h": 3_600_000, "2h": 7_200_000}
OUT = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/intraday")
OUT.mkdir(parents=True, exist_ok=True)

# Start dates per timeframe — go as far back as Binance has. 15m is huge so cap ~2.5y.
SINCE = {
    "15m": "2023-01-01T00:00:00Z",
    "1h": "2020-01-01T00:00:00Z",
    "2h": "2020-01-01T00:00:00Z",
}


def fetch_ohlcv_full(ex, symbol, timeframe, since_ms):
    tf_ms = TF_MS[timeframe]
    now = ex.milliseconds()
    out = []
    since = since_ms
    last = None
    while since < now:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"   retry {symbol} {timeframe} @ {since}: {e}", file=sys.stderr)
            time.sleep(2.0)
            continue
        if not batch:
            # advance a window in case of a hole; stop if no progress for long
            since += tf_ms * 1000
            if since > now:
                break
            continue
        out.extend(batch)
        batch_last = batch[-1][0]
        if last is not None and batch_last <= last:
            break
        last = batch_last
        since = batch_last + tf_ms
        time.sleep(ex.rateLimit / 1000.0)
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def fetch_funding(coin):
    """Binance USDT-M funding history (paginated, 1000/call)."""
    sym = f"{coin}USDT"
    base = "https://fapi.binance.com/fapi/v1/fundingRate"
    out = []
    start = 1514764800000  # 2018-01-01
    while True:
        url = f"{base}?symbol={sym}&startTime={start}&limit=1000"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                rows = json.loads(r.read())
        except Exception as e:
            print(f"   funding {sym} err: {e}", file=sys.stderr)
            break
        if not rows:
            break
        out.extend(rows)
        last = rows[-1]["fundingTime"]
        if len(rows) < 1000:
            break
        start = last + 1
        time.sleep(0.25)
    if not out:
        return pd.DataFrame(columns=["ts", "funding_rate"])
    df = pd.DataFrame(out)
    df["ts"] = df["fundingTime"].astype("int64")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main():
    tfs = sys.argv[1:] if len(sys.argv) > 1 else ["1h", "2h", "15m"]
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    for tf in tfs:
        since_ms = ex.parse8601(SINCE[tf])
        for pair in PAIRS:
            coin = pair.split("/")[0]
            dest = OUT / f"{coin}_{tf}.csv"
            if dest.exists():
                print(f"skip {dest.name} (exists)")
                continue
            t0 = time.time()
            df = fetch_ohlcv_full(ex, pair, tf, since_ms)
            df.to_csv(dest, index=False)
            print(f"{pair} {tf}: {len(df)} bars [{pd.to_datetime(df.ts.iloc[0],unit='ms')} -> "
                  f"{pd.to_datetime(df.ts.iloc[-1],unit='ms')}] {time.time()-t0:.0f}s")
    # funding once
    for pair in PAIRS:
        coin = pair.split("/")[0]
        dest = OUT / f"{coin}_funding.csv"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        fdf = fetch_funding(coin)
        fdf.to_csv(dest, index=False)
        print(f"{coin} funding: {len(fdf)} rows")


if __name__ == "__main__":
    main()
