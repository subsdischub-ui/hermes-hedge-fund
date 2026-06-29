"""Adversarial scrutiny of the surviving OI/positioning candidates.

The per-asset gate is gamed by tiny-sample Sharpe (4-9 OOS trades). Here we:
  1. POOL all 7 assets' trades into one chronological sample (n in the 80-160
     range) so OOS-40% and folds have real statistical content.
  2. Compute a proper Newey-West-free per-trade t-stat on the pooled sample.
  3. Run a TIME-SPLIT OOS: trades whose ENTRY ts is in the last 40% of the
     calendar window (not last 40% of trades) — honest chronological OOS.
  4. Block-bootstrap the pooled per-trade net to get a 5th percentile Sharpe.
  5. Report last-3-days pooled behaviour.

Candidates examined are the 3 configs with full cross-asset sign agreement plus
the per-family gate winners.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, "/Users/lacarbonnerie/Developer/hermes-hedge-fund/scripts")
from oi_harness import (  # noqa: E402
    load, PAIRS, TAKER, MAKER, BPY,
    sig_oi_trend, sig_lsr_contrarian, sig_taker_flow, _forward_logret,
)

rng = np.random.default_rng(7)


def pooled_trades(builder, params, horizon):
    """Return DataFrame of all no-overlap trades across assets: entry_ts, net_t, net_m, coin."""
    recs = []
    for coin in PAIRS:
        df = load(coin)
        sig = builder(df, **params)
        close = df["close"].to_numpy(float)
        ts = df["ts"].to_numpy("int64")
        fwd = _forward_logret(close, horizon)
        n = len(close)
        i = 0
        while i < n:
            s = sig[i]
            if s != 0 and np.isfinite(fwd[i]):
                g = s * fwd[i]
                recs.append((int(ts[i]), g - TAKER, g - MAKER, coin))
                i += horizon
            else:
                i += 1
    t = pd.DataFrame(recs, columns=["ts", "net_t", "net_m", "coin"]).sort_values("ts")
    return t.reset_index(drop=True)


def ann_sharpe(x, tpy):
    x = np.asarray(x)
    if len(x) < 5 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(tpy))


def analyze(name, builder, params, horizon, now_ms):
    t = pooled_trades(builder, params, horizon)
    n = len(t)
    if n < 10:
        print(f"\n## {name}: only {n} pooled trades — skip"); return
    net = t["net_t"].to_numpy()
    net_m = t["net_m"].to_numpy()
    # trade rate: total trades / (calendar span across pool, in years) — but pooled
    # across 7 assets, so per-asset tpy ~ pooled_tpy/7. Report per-asset.
    span_years = (t["ts"].iloc[-1] - t["ts"].iloc[0]) / (365 * 24 * 3600 * 1000)
    tpy_per_asset = (n / 7) / max(span_years, 1e-9)
    # full-sample per-trade t-stat
    tstat = net.mean() / (net.std(ddof=1) / np.sqrt(n)) if net.std(ddof=1) > 0 else 0
    # honest CHRONOLOGICAL time-split OOS: last 40% of calendar
    t0, t1 = t["ts"].iloc[0], t["ts"].iloc[-1]
    cut = t0 + 0.6 * (t1 - t0)
    oos = net[t["ts"].to_numpy() >= cut]
    ins = net[t["ts"].to_numpy() < cut]
    # 3 expanding folds on pooled, calendar-based
    folds = []
    for frac in (1 / 3, 2 / 3, 1.0):
        cf = t0 + frac * (t1 - t0)
        seg = net[t["ts"].to_numpy() <= cf]
        folds.append(1.0 if (len(seg) >= 3 and seg.mean() > 0) else 0.0)
    foldpos = np.mean(folds)
    # stationary block bootstrap (block ~ avg trades per 2 days) for Sharpe 5th pct
    blk = max(3, int(round(n / span_years / 365 * 2)))
    sims = []
    for _ in range(2000):
        idx = []
        while len(idx) < n:
            st = rng.integers(0, n)
            idx.extend(range(st, min(st + blk, n)))
        idx = np.array(idx[:n])
        sims.append(ann_sharpe(net[idx], tpy_per_asset))
    p5 = np.percentile(sims, 5)
    # last 3 days
    c3 = now_ms - 3 * 24 * 3600 * 1000
    m3 = t["ts"].to_numpy() >= c3
    l3n = int(m3.sum())
    l3pnl = float(net[m3].sum()) if l3n else 0.0

    print(f"\n## {name}")
    print(f"   params={params} horizon={horizon}h")
    print(f"   pooled n={n}  per-asset tpy~{tpy_per_asset:.0f}  span={span_years*365:.1f}d")
    print(f"   net/trade taker={net.mean()*1e4:.1f}bps  maker={net_m.mean()*1e4:.1f}bps  t-stat={tstat:.2f}")
    print(f"   full Sharpe(taker, per-asset-ann)={ann_sharpe(net,tpy_per_asset):.2f}")
    print(f"   IN-sample net={ins.mean()*1e4:.1f}bps (n={len(ins)})  "
          f"OOS(last40%cal) net={oos.mean()*1e4:.1f}bps (n={len(oos)})  "
          f"OOS Sharpe={ann_sharpe(oos,tpy_per_asset):.2f}")
    print(f"   expanding-fold positive frac={foldpos:.2f}")
    print(f"   bootstrap Sharpe 5th-pct={p5:.2f}  (>0 => robustly positive)")
    print(f"   last-3-days pooled: {l3n} trades, net pnl={l3pnl*1e4:.1f}bps")


def main():
    now_ms = max(int(load(c)["ts"].iloc[-1]) for c in PAIRS)
    # full cross-asset-agree configs + family winners
    analyze("lsr_contrarian top_lsr win48 z1.0 h48", sig_lsr_contrarian,
            {"col": "top_lsr", "win": 48, "z_thr": 1.0}, 48, now_ms)
    analyze("oi_price WITH win48 dz1.5 h24", sig_oi_trend,
            {"win": 48, "dz_thr": 1.5, "mode": "with"}, 24, now_ms)
    analyze("oi_price WITH win6 dz1.0 h24", sig_oi_trend,
            {"win": 6, "dz_thr": 1.0, "mode": "with"}, 24, now_ms)
    analyze("taker_flow fade win24 z1.5 h24", sig_taker_flow,
            {"win": 24, "z_thr": 1.5, "mode": "fade"}, 24, now_ms)
    analyze("lsr_contrarian gl_lsr win48 z1.0 h24", sig_lsr_contrarian,
            {"col": "gl_lsr", "win": 48, "z_thr": 1.0}, 24, now_ms)
    # a shorter-horizon (higher cadence) lsr to chase the >150 tpy goal harder
    analyze("lsr_contrarian top_lsr win24 z1.0 h12", sig_lsr_contrarian,
            {"col": "top_lsr", "win": 24, "z_thr": 1.0}, 12, now_ms)


if __name__ == "__main__":
    main()
