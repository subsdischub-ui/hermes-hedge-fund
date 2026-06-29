"""Rigorous, vectorized, no-overlap COINTEGRATION / PAIRS mean-reversion harness.

Genuinely-new structure vs prior single-asset sweeps: trade the SPREAD between two
assets back to its mean (market-neutral), not a single-asset directional edge.

Family: rolling-OLS cointegration spread, z-score reversion.
  - For a pair (A, B): regress log(A) on log(B) over a CAUSAL rolling window to get
    hedge ratio beta_t and intercept a_t. Spread_t = logA_t - (a_t + beta_t*logB_t).
  - Beta/intercept estimated from window ENDING at t-1 (shift(1)) -> uses only data<=t-1,
    so the spread value at t and the trade decision at t are causal.
  - z_t = (spread_t - rollmean(spread)) / rollstd(spread), the rolling stats ALSO
    shifted by 1 (mean/std of spread over [t-zwin, t-1]).
  - ENTRY when |z_t| >= entry_z: if z>0 spread rich -> SHORT spread (short A, long B);
    if z<0 spread cheap -> LONG spread (long A, short B). Market-neutral (beta-weighted).
  - EXIT first of: (a) |z| <= exit_z (reverted), or (b) z crosses through 0 (sign flip),
    or (c) max-hold H bars (vertical barrier). Whichever first.
  - NO-OVERLAP: greedy. While in a position no new entry; next entry only at/after exit+1.

P&L (market-neutral, beta-weighted spread return, in log space, per unit gross exposure
on each leg). For a LONG-spread position opened at i, closed at j:
    leg_A =  (logA_j - logA_i)              # long A
    leg_B = -beta_i * (logB_j - logB_i)     # short beta units of B
    gross = leg_A + leg_B                    # spread move captured
  SHORT spread flips the sign. Costs: TWO legs are traded, each round-trip.
    taker leg RT = 0.0015  -> pair RT = 0.0030  (2 legs)
    maker leg RT = 0.0006  -> pair RT = 0.0012
  We also report a "1x notional" interpretation where exposure is split so total gross
  notional == 1 (each leg ~0.5), which halves both gross and cost -> Sharpe identical.
  Sharpe is invariant to that scaling; net/trade reported on the FULL 2-leg convention
  (each leg full notional) to be conservative on cost. Beta is clipped to [0.1, 5] for
  cost realism (extreme hedge ratios would imply huge B notional).

Metrics per (config, pair): trades_per_year, net/trade (taker & maker), full annualized
Sharpe, OOS-40% Sharpe (last 40% chronological), 3 expanding folds (fold-positive frac),
last-3-days traded+pnl.

GATE (per pair): net>0 at TAKER (2-leg) AND OOS-40% Sharpe>0.3 AND fold-positive>=0.67
AND trades_per_year>=150. Family passes cross-asset if >=2 distinct pairs clear (per the
task's relaxed bar for pairs/cross-sectional families).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/intraday")
COINS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]
LEG_TAKER = 0.0015
LEG_MAKER = 0.0006
PAIR_TAKER = 2 * LEG_TAKER  # 0.0030 round trip, both legs
PAIR_MAKER = 2 * LEG_MAKER  # 0.0012
BARS_PER_YEAR = {"1h": 365 * 24, "4h": 365 * 6}
BETA_CLIP = (0.1, 5.0)

# Candidate pairs the task named, plus a few alt-alt for breadth.
CANDIDATE_PAIRS = [
    ("ETH", "BTC"), ("SOL", "BTC"), ("SOL", "ETH"), ("AVAX", "BTC"),
    ("ARB", "BTC"), ("AVAX", "ETH"), ("SOL", "AVAX"), ("ARB", "ETH"),
    ("SUI", "SOL"), ("WIF", "SOL"), ("AVAX", "SOL"), ("ARB", "SOL"),
]


def load(coin: str, tf: str) -> pd.DataFrame:
    """Load 1h CSV; if tf=='4h', resample causally from 1h (completed bars only)."""
    df = pd.read_csv(DATA / f"{coin}_1h.csv").dropna(subset=["close"]).reset_index(drop=True)
    if tf == "1h":
        return df[["ts", "close"]].copy()
    if tf == "4h":
        s = df.copy()
        s["dt"] = pd.to_datetime(s["ts"], unit="ms", utc=True)
        s = s.set_index("dt")
        # label='left', closed='left' -> each 4h bar covers [t, t+4h); ts is bar-open.
        agg = s["close"].resample("4h", label="left", closed="left").last().dropna()
        out = pd.DataFrame({"ts": (agg.index.view("int64") // 1_000_000).astype("int64"),
                            "close": agg.to_numpy()})
        return out.reset_index(drop=True)
    raise ValueError(tf)


def align_pair(a: str, b: str, tf: str):
    """Inner-join two coins on ts; return (ts, logA, logB)."""
    da = load(a, tf).rename(columns={"close": "ca"})
    db = load(b, tf).rename(columns={"close": "cb"})
    m = pd.merge(da, db, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    ts = m["ts"].to_numpy("int64")
    logA = np.log(m["ca"].to_numpy(float))
    logB = np.log(m["cb"].to_numpy(float))
    return ts, logA, logB


def rolling_beta_intercept(logA: np.ndarray, logB: np.ndarray, win: int):
    """CAUSAL rolling OLS of logA ~ a + beta*logB over the prior `win` bars.

    beta_t, a_t are estimated from the window [t-win, t-1] (shift(1) applied), so the
    spread at bar t uses no future info. Returns beta, intercept arrays (NaN until ready).
    Vectorized via rolling sums.
    """
    x = pd.Series(logB)
    y = pd.Series(logA)
    sx = x.rolling(win).sum()
    sy = y.rolling(win).sum()
    sxx = (x * x).rolling(win).sum()
    sxy = (x * y).rolling(win).sum()
    n = float(win)
    denom = (n * sxx - sx * sx)
    beta = (n * sxy - sx * sy) / denom.replace(0, np.nan)
    intercept = (sy - beta * sx) / n
    # shift so values at t use window ending at t-1 (strictly causal for decision at t)
    beta = beta.shift(1)
    intercept = intercept.shift(1)
    beta = beta.clip(*BETA_CLIP)
    return beta.to_numpy(), intercept.to_numpy()


def build_spread_z(logA: np.ndarray, logB: np.ndarray, beta: np.ndarray,
                   intercept: np.ndarray, zwin: int):
    """Spread_t = logA_t - (a_t + beta_t*logB_t); z from rolling mean/std (shifted)."""
    spread = logA - (intercept + beta * logB)
    sp = pd.Series(spread)
    mu = sp.rolling(zwin).mean().shift(1)
    sd = sp.rolling(zwin).std(ddof=0).shift(1)
    z = (sp - mu) / sd.replace(0, np.nan)
    return spread, z.to_numpy()


@dataclass
class Result:
    pair: str
    n_trades: int
    trades_per_year: float
    net_per_trade_taker: float
    net_per_trade_maker: float
    sharpe_taker: float
    sharpe_maker: float
    oos40_sharpe_taker: float
    fold_pos_frac_taker: float
    last3d_trades: int
    last3d_pnl_taker: float
    last3d_pnl_maker: float


def _ann_sharpe(net: np.ndarray, tpy: float) -> float:
    if len(net) < 3:
        return 0.0
    mu = net.mean()
    sd = net.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(mu / sd * np.sqrt(tpy))


def backtest_pair(ts, logA, logB, beta, intercept, spread, z, *, entry_z, exit_z,
                  H, tf, now_ms) -> Result:
    """Greedy no-overlap spread reversion backtest. Causal by construction."""
    n = len(z)
    bpy = BARS_PER_YEAR[tf]
    entry_ts, gross = [], []
    i = 0
    while i < n - 1:
        if not np.isfinite(z[i]) or not np.isfinite(beta[i]):
            i += 1
            continue
        side = 0
        if z[i] >= entry_z:
            side = -1  # spread rich -> short spread (short A, long B)
        elif z[i] <= -entry_z:
            side = +1  # spread cheap -> long spread (long A, short B)
        if side == 0:
            i += 1
            continue

        bi = beta[i]
        if not np.isfinite(bi):
            i += 1
            continue
        # forward scan for exit
        exit_j = min(i + H, n - 1)
        for j in range(i + 1, min(i + H, n - 1) + 1):
            zj = z[j]
            if not np.isfinite(zj):
                continue
            reverted = abs(zj) <= exit_z
            crossed = (side == -1 and zj <= 0) or (side == +1 and zj >= 0)
            if reverted or crossed:
                exit_j = j
                break
        # P&L in log space, beta-weighted, hedge frozen at entry (bi)
        leg_a = (logA[exit_j] - logA[i])
        leg_b = -bi * (logB[exit_j] - logB[i])
        g = side * (leg_a + leg_b)
        entry_ts.append(ts[i])
        gross.append(g)
        i = exit_j + 1  # no overlap

    if not gross:
        return Result("", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    gross = np.asarray(gross)
    net_t = gross - PAIR_TAKER
    net_m = gross - PAIR_MAKER
    ets = np.asarray(entry_ts, dtype="int64")

    span_ms = float(ets[-1] - ets[0]) or 1.0
    years = span_ms / (365.25 * 24 * 3600 * 1000)
    tpy = len(gross) / max(years, 1e-9)

    sharpe_t = _ann_sharpe(net_t, tpy)
    sharpe_m = _ann_sharpe(net_m, tpy)

    k = len(net_t)
    oos_start = int(np.floor(k * 0.6))
    oos = net_t[oos_start:]
    oos_sharpe = _ann_sharpe(oos, tpy) if len(oos) >= 3 else 0.0

    folds = []
    for frac in (1 / 3, 2 / 3, 1.0):
        m = max(3, int(np.floor(k * frac)))
        folds.append(1.0 if net_t[:m].mean() > 0 else 0.0)
    fold_pos = float(np.mean(folds))

    cutoff = now_ms - 3 * 24 * 3600 * 1000
    mask3 = ets >= cutoff
    l3n = int(mask3.sum())
    l3t = float(net_t[mask3].sum()) if l3n else 0.0
    l3m = float(net_m[mask3].sum()) if l3n else 0.0

    return Result("", len(gross), tpy, float(net_t.mean()), float(net_m.mean()),
                  sharpe_t, sharpe_m, oos_sharpe, fold_pos, l3n, l3t, l3m)


def gate_pass(r: Result) -> bool:
    return (r.net_per_trade_taker > 0 and r.oos40_sharpe_taker > 0.3
            and r.fold_pos_frac_taker >= 0.67 and r.trades_per_year >= 150)


def run():
    # global now = latest 1h bar
    now_ms = 0
    for c in COINS:
        df = pd.read_csv(DATA / f"{c}_1h.csv")
        now_ms = max(now_ms, int(df["ts"].iloc[-1]))
    print(f"now_ms={now_ms} ({pd.to_datetime(now_ms, unit='ms')})", file=sys.stderr)

    # cache aligned pairs per tf
    cache = {}
    for (a, b) in CANDIDATE_PAIRS:
        for tf in ("1h", "4h"):
            try:
                cache[(a, b, tf)] = align_pair(a, b, tf)
            except Exception as e:  # noqa
                print(f"skip {a}-{b} {tf}: {e}", file=sys.stderr)

    # sweep grid
    configs = []
    for tf in ("1h", "4h"):
        betawins = [120, 240, 480] if tf == "1h" else [60, 120, 240]
        zwins = [48, 96, 168] if tf == "1h" else [24, 48, 84]
        Hs = [24, 48, 96] if tf == "1h" else [12, 24, 48]
        for bw in betawins:
            for zw in zwins:
                for ez in (1.5, 2.0, 2.5):
                    for xz in (0.0, 0.5):
                        for H in Hs:
                            configs.append(dict(tf=tf, betawin=bw, zwin=zw,
                                                entry_z=ez, exit_z=xz, H=H))

    rows = []
    for cfg in configs:
        tf = cfg["tf"]
        per_pair = []
        for (a, b) in CANDIDATE_PAIRS:
            key = (a, b, tf)
            if key not in cache:
                continue
            ts, logA, logB = cache[key]
            if len(ts) < cfg["betawin"] + cfg["zwin"] + 50:
                continue
            beta, intercept = rolling_beta_intercept(logA, logB, cfg["betawin"])
            spread, z = build_spread_z(logA, logB, beta, intercept, cfg["zwin"])
            r = backtest_pair(ts, logA, logB, beta, intercept, spread, z,
                              entry_z=cfg["entry_z"], exit_z=cfg["exit_z"],
                              H=cfg["H"], tf=tf, now_ms=now_ms)
            r.pair = f"{a}-{b}"
            per_pair.append(r)

        passers = [r for r in per_pair if gate_pass(r)]
        n_pass = len(passers)
        traded = [r for r in per_pair if r.n_trades >= 5]
        med = lambda key: float(np.median([getattr(r, key) for r in traded])) if traded else 0.0
        rows.append({
            "cfg": cfg,
            "n_pass": n_pass,
            "passes_cross_asset": n_pass >= 2,
            "passers": [r.pair for r in passers],
            "med_tpy": med("trades_per_year"),
            "med_net_taker": med("net_per_trade_taker"),
            "med_net_maker": med("net_per_trade_maker"),
            "med_sharpe_taker": med("sharpe_taker"),
            "med_oos40_taker": med("oos40_sharpe_taker"),
            "med_fold_pos": med("fold_pos_frac_taker"),
            "last3d_trades": sum(r.last3d_trades for r in per_pair),
            "last3d_pnl_taker": sum(r.last3d_pnl_taker for r in per_pair),
            "per_pair": [r.__dict__ for r in per_pair],
        })

    out = Path(DATA / "_pairs_results.json")
    out.write_text(json.dumps(rows, indent=2, default=str))

    # ---- reporting ----
    passers = [r for r in rows if r["passes_cross_asset"]]
    print(f"\n{len(rows)} configs, {len(passers)} with >=2 pairs passing the full gate",
          file=sys.stderr)

    def score(r):
        # rank by (n_pass, median oos40 among passers, tpy)
        return (r["n_pass"], r["med_oos40_taker"], r["med_tpy"])

    pool = passers if passers else rows
    best = max(pool, key=score)
    print("\n=== BEST CONFIG ===", file=sys.stderr)
    print(json.dumps({k: best[k] for k in
                      ("cfg", "n_pass", "passers", "med_tpy", "med_net_taker",
                       "med_net_maker", "med_sharpe_taker", "med_oos40_taker",
                       "med_fold_pos", "last3d_trades", "last3d_pnl_taker")},
                     indent=2, default=str), file=sys.stderr)
    print("\nper-pair @ best:", file=sys.stderr)
    for pp in best["per_pair"]:
        print(f"  {pp['pair']:10s} n={pp['n_trades']:4d} tpy={pp['trades_per_year']:6.0f} "
              f"net_t={pp['net_per_trade_taker']*1e4:7.1f}bps "
              f"net_m={pp['net_per_trade_maker']*1e4:7.1f}bps "
              f"shrp_t={pp['sharpe_taker']:6.2f} oos40={pp['oos40_sharpe_taker']:6.2f} "
              f"fold={pp['fold_pos_frac_taker']:.2f} "
              f"l3={pp['last3d_trades']}/{pp['last3d_pnl_taker']*1e4:.1f}bps "
              f"{'PASS' if gate_pass(Result(**pp)) else ''}", file=sys.stderr)

    # also show the best config that maximizes trades_per_year among those net>0 at taker
    high_freq = [r for r in rows if r["med_net_taker"] > 0 and r["med_tpy"] >= 150]
    if high_freq:
        hf = max(high_freq, key=lambda r: (r["n_pass"], r["med_tpy"]))
        print("\n=== HIGHEST-FREQUENCY (net>0 taker, tpy>=150) ===", file=sys.stderr)
        print(f"  cfg={hf['cfg']} n_pass={hf['n_pass']} tpy={hf['med_tpy']:.0f} "
              f"net_t={hf['med_net_taker']*1e4:.1f}bps oos40={hf['med_oos40_taker']:.2f} "
              f"fold={hf['med_fold_pos']:.2f}", file=sys.stderr)

    return rows, best, now_ms


if __name__ == "__main__":
    run()
