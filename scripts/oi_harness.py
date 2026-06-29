"""Rigorous, vectorized, no-overlap backtest for OI + crowd-positioning edges.

Data: data/oi/{COIN}_oi.csv (1h grid, ~31 days, 744 rows/coin). Columns:
  ts, oi, oi_val, gl_lsr, top_lsr, taker_bsr, open, high, low, close, volume

*** OOS WARNING ***  Only ~31 days of history. "OOS-40%" is ~12.5 days and the
3 expanding folds are ~10/21/31 days. This is far too short to trust; treat any
pass as a HYPOTHESIS, not a validated edge. Reported honestly throughout.

Signal families (all CAUSAL — feature at bar i uses only data <= i):
  A1  oi_price_trend:   sign(d_oi) confirms sign(d_price) -> go WITH the move
                        (OI up + price up = longs building -> long;
                         OI up + price down = shorts building -> we test BOTH a
                         "fade the build" (long, squeeze-up) and "go with").
  A2  oi_squeeze_up:    OI rising fast + price falling -> shorts crowding -> LONG
  B   lsr_contrarian:   extreme global / top-trader long-short ratio -> FADE
                        (very long crowd -> short; very short crowd -> long)
  C   taker_flow:       extreme taker buy/sell ratio -> test continuation AND fade

Lead-lag causality: a feature built from series X at bar i (e.g. z-score of d_oi
over a trailing window ending at i) uses only X[<=i]. The trade enters at bar i
close and realizes the forward return over `horizon` bars — so it only ever uses
the positioning move that already happened. No look-ahead.

Protocol mirrors scripts/intraday_harness.py exactly:
  * greedy no-overlap (min-gap == horizon), independent trades
  * net of cost: maker 0.0006 RT, taker 0.0015 RT
  * per (cfg, asset): trades_per_year, net/trade, full Sharpe, OOS-40% Sharpe,
    3 expanding folds -> fold-positive frac, last-3-days traded+pnl
  * GATE: net>0 @taker AND OOS-40% Sharpe>0.3 AND fold-pos>=0.67 AND tpy>=150,
    on >=4/7 pairs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/oi")
PAIRS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]
MAKER = 0.0006
TAKER = 0.0015
BPY = 365 * 24  # 1h bars/year


def load(coin: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{coin}_oi.csv")
    return df.dropna(subset=["close"]).reset_index(drop=True)


# ------------------------------ helpers ------------------------------------ #
def _zscore(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()


def _roll_logret(close: np.ndarray, L: int) -> np.ndarray:
    logc = np.log(close)
    out = np.full(len(close), np.nan)
    out[L:] = logc[L:] - logc[:-L]
    return out


def _forward_logret(close: np.ndarray, horizon: int) -> np.ndarray:
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = np.log(close[horizon:] / close[: n - horizon])
    return fwd


# ------------------------------ signals ------------------------------------ #
def sig_oi_trend(df, win, dz_thr, mode):
    """OI/price confirmation. dOI z-score (trailing `win`) and price change over
    same `win`. mode='with': long when both up, short when both down. mode='squeeze':
    OI up + price down -> long (short squeeze); OI down + price up -> short."""
    close = df["close"].to_numpy(float)
    oi = df["oi"].to_numpy(float)
    d_oi = np.concatenate([[np.nan], np.diff(oi)])
    z_oi = _zscore(d_oi, win)
    pret = _roll_logret(close, win)
    sig = np.zeros(len(close), dtype=int)
    oi_up = z_oi > dz_thr
    oi_dn = z_oi < -dz_thr
    p_up = pret > 0
    p_dn = pret < 0
    if mode == "with":
        sig = np.where(oi_up & p_up, 1, np.where(oi_up & p_dn, -1, 0))
    elif mode == "squeeze":
        # OI rising + price falling = shorts crowding -> LONG (squeeze up)
        sig = np.where(oi_up & p_dn, 1, np.where(oi_up & p_up, 0, 0))
    valid = np.isfinite(z_oi) & np.isfinite(pret)
    return np.where(valid, sig, 0).astype(int)


def sig_lsr_contrarian(df, col, win, z_thr):
    """Fade extreme crowd long-short ratio. col in {gl_lsr, top_lsr}.
    z>z_thr (crowd very long) -> short; z<-z_thr (crowd very short) -> long."""
    x = df[col].to_numpy(float)
    z = _zscore(x, win)
    sig = np.where(z > z_thr, -1, np.where(z < -z_thr, 1, 0))
    return np.where(np.isfinite(z), sig, 0).astype(int)


def sig_taker_flow(df, win, z_thr, mode):
    """Taker buy/sell ratio extreme. mode='cont': aggressive buyers (z>z_thr)->long
    (flow continuation). mode='fade': z>z_thr -> short (exhaustion)."""
    x = df["taker_bsr"].to_numpy(float)
    z = _zscore(x, win)
    if mode == "cont":
        sig = np.where(z > z_thr, 1, np.where(z < -z_thr, -1, 0))
    else:  # fade
        sig = np.where(z > z_thr, -1, np.where(z < -z_thr, 1, 0))
    return np.where(np.isfinite(z), sig, 0).astype(int)


# ------------------------------ backtest core ------------------------------ #
@dataclass
class Result:
    coin: str
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


def _ann_sharpe(net, tpy):
    if len(net) < 3:
        return 0.0
    sd = net.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(net.mean() / sd * np.sqrt(tpy))


def backtest(df, sig, horizon, coin, now_ms) -> Result:
    close = df["close"].to_numpy(float)
    ts = df["ts"].to_numpy("int64")
    fwd = _forward_logret(close, horizon)
    n = len(close)
    entry_idx, dir_ret = [], []
    i = 0
    while i < n:
        s = sig[i]
        if s != 0 and np.isfinite(fwd[i]):
            entry_idx.append(i)
            dir_ret.append(s * fwd[i])
            i += horizon
        else:
            i += 1
    if not dir_ret:
        return Result(coin, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    gross = np.array(dir_ret)
    net_t = gross - TAKER
    net_m = gross - MAKER
    entry_ts = ts[np.array(entry_idx)]
    span_bars = entry_idx[-1] - entry_idx[0] + horizon
    span_years = max(span_bars / BPY, 1e-9)
    tpy = len(gross) / span_years
    k = len(net_t)
    oos = net_t[int(np.floor(k * 0.6)):]
    folds = []
    for frac in (1 / 3, 2 / 3, 1.0):
        m = max(3, int(np.floor(k * frac)))
        folds.append(1.0 if net_t[:m].mean() > 0 else 0.0)
    cutoff = now_ms - 3 * 24 * 3600 * 1000
    mask3 = entry_ts >= cutoff
    l3n = int(mask3.sum())
    return Result(
        coin, k, tpy, float(net_t.mean()), float(net_m.mean()),
        _ann_sharpe(net_t, tpy), _ann_sharpe(net_m, tpy),
        _ann_sharpe(oos, tpy) if len(oos) >= 3 else 0.0,
        float(np.mean(folds)),
        l3n, float(net_t[mask3].sum()) if l3n else 0.0,
        float(net_m[mask3].sum()) if l3n else 0.0,
    )


def gate_pass(r: Result) -> bool:
    return (r.net_per_trade_taker > 0 and r.oos40_sharpe_taker > 0.3
            and r.fold_pos_frac_taker >= 0.67 and r.trades_per_year >= 150)


@dataclass
class Family:
    name: str
    builder: object
    configs: list


def run_family(fam: Family, now_ms: int):
    rows = []
    for cfg in fam.configs:
        horizon = cfg["horizon"]
        params = {k: v for k, v in cfg.items() if k != "horizon"}
        per_asset = []
        for coin in PAIRS:
            df = load(coin)
            sig = fam.builder(df, **params)
            per_asset.append(backtest(df, sig, horizon, coin, now_ms))
        n_pass = sum(gate_pass(r) for r in per_asset)
        traded = [r for r in per_asset if r.n_trades >= 5]
        med = lambda key: float(np.median([getattr(r, key) for r in traded])) if traded else 0.0
        rows.append({
            "family": fam.name,
            "cfg": {**params, "horizon": horizon},
            "n_pass": n_pass,
            "passes_cross_asset": n_pass >= 4,
            "n_traded": len(traded),
            "med_tpy": med("trades_per_year"),
            "med_net_taker": med("net_per_trade_taker"),
            "med_net_maker": med("net_per_trade_maker"),
            "med_sharpe_taker": med("sharpe_taker"),
            "med_oos40_taker": med("oos40_sharpe_taker"),
            "med_fold_pos": med("fold_pos_frac_taker"),
            "last3d_trades": sum(r.last3d_trades for r in per_asset),
            "last3d_pnl_taker": sum(r.last3d_pnl_taker for r in per_asset),
            "per_asset": [r.__dict__ for r in per_asset],
        })
    return rows


def main():
    latest = max(int(load(c)["ts"].iloc[-1]) for c in PAIRS)
    now_ms = latest
    print(f"now_ms={now_ms} ({pd.to_datetime(now_ms, unit='ms')})  "
          f"*** ~31d history: OOS is ~12d, treat passes as HYPOTHESES ***",
          file=sys.stderr)

    families = []

    # A) OI + price confirmation / squeeze
    oi_cfgs = []
    for win in (6, 12, 24, 48):
        for dz in (0.5, 1.0, 1.5):
            for mode in ("with", "squeeze"):
                for h in (3, 6, 12, 24):
                    oi_cfgs.append({"win": win, "dz_thr": dz, "mode": mode, "horizon": h})
    families.append(Family("oi_price", sig_oi_trend, oi_cfgs))

    # B) crowd long-short ratio contrarian fade (global + top-trader)
    lsr_cfgs = []
    for col in ("gl_lsr", "top_lsr"):
        for win in (12, 24, 48, 96):
            for z in (1.0, 1.5, 2.0):
                for h in (3, 6, 12, 24, 48):
                    lsr_cfgs.append({"col": col, "win": win, "z_thr": z, "horizon": h})
    families.append(Family("lsr_contrarian", sig_lsr_contrarian, lsr_cfgs))

    # C) taker aggressive-flow continuation AND fade
    tk_cfgs = []
    for win in (6, 12, 24, 48):
        for z in (1.0, 1.5, 2.0):
            for mode in ("cont", "fade"):
                for h in (3, 6, 12, 24):
                    tk_cfgs.append({"win": win, "z_thr": z, "mode": mode, "horizon": h})
    families.append(Family("taker_flow", sig_taker_flow, tk_cfgs))

    all_rows = []
    for fam in families:
        rows = run_family(fam, now_ms)
        all_rows.extend(rows)
        passers = [r for r in rows if r["passes_cross_asset"]]
        pool = passers if passers else rows
        best = max(pool, key=lambda r: (r["n_pass"], r["med_oos40_taker"]))
        print(f"\n=== {fam.name} === ({len(rows)} cfgs, {len(passers)} cross-asset passers)",
              file=sys.stderr)
        print(f"  BEST cfg={best['cfg']} n_pass={best['n_pass']}/7 "
              f"tpy={best['med_tpy']:.0f} net_t={best['med_net_taker']*1e4:.1f}bps "
              f"net_m={best['med_net_maker']*1e4:.1f}bps sharpe_t={best['med_sharpe_taker']:.2f} "
              f"oos40={best['med_oos40_taker']:.2f} foldpos={best['med_fold_pos']:.2f} "
              f"l3_tr={best['last3d_trades']} l3_pnl_t={best['last3d_pnl_taker']*1e4:.1f}bps",
              file=sys.stderr)

    out = DATA / "_oi_results.json"
    out.write_text(json.dumps(all_rows, indent=2, default=str))
    print(f"\nwrote {out}  ({len(all_rows)} configs)", file=sys.stderr)


if __name__ == "__main__":
    main()
