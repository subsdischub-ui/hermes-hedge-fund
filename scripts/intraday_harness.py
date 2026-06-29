"""Rigorous, vectorized, no-overlap intraday backtest harness.

Tests fast-cadence families on 1h/2h/15m bars across 7 pairs:
  - vol-scaled momentum (the validated edge, faster cadence)  [trend]
  - intraday mean-reversion: RSI2, z-score fade, Bollinger fade  [MR control]

Protocol (no p-hacking):
  * Signals are CAUSAL: every feature at bar i uses only bars <= i.
  * Entry at NEXT bar's open is approximated by entering at bar i close, holding
    `horizon` bars, exiting at bar (i+horizon) close. We measure the realized
    forward log-return over the horizon.
  * NO OVERLAP: greedy walk with min-gap == horizon. On a fired bar we take the
    trade and jump i += horizon, so trades never overlap (independent samples).
  * NET OF COST: round-trip fee subtracted from each trade's directional return.
      maker = 0.0006, taker = 0.0015.
  * Per (config, asset): trades_per_year, net/trade, full annualized Sharpe
    (per-trade mean/std * sqrt(trades_per_year)), OOS-40% Sharpe (last 40%
    chronological), 3 expanding folds -> fold-positive fraction.
  * LAST-3-DAYS user test: trade count + summed net PnL over the final 72h.

GATE (per asset): net>0 at TAKER cost AND OOS-40% Sharpe>0.3 AND fold-positive>=0.67
                  AND trades_per_year>=150. Family passes cross-asset if >=4/7 clear.
"""
from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/intraday")
PAIRS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]
MAKER = 0.0006
TAKER = 0.0015
BARS_PER_YEAR = {"15m": 365 * 96, "1h": 365 * 24, "2h": 365 * 12}
NOW_MS = None  # set from data


def load(coin: str, tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{coin}_{tf}.csv")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ----------------------------- signal builders ----------------------------- #
def sig_volmom(df: pd.DataFrame, lookbacks: tuple[int, ...], thr: float) -> np.ndarray:
    """Vectorized vol-scaled multi-lookback momentum score -> {-1,0,+1}.
    Mirrors ts_mom_vol_scaled: per lookback L, (logret over L) / (std of L 1-bar
    rets * sqrt(L)); average across L; sign with dead-band `thr`. Causal."""
    logc = np.log(df["close"].to_numpy(dtype=float))
    r = np.diff(logc, prepend=logc[0])  # 1-bar log returns, r[0]=0
    n = len(logc)
    scores = np.zeros(n)
    counts = np.zeros(n)
    s = pd.Series(r)
    for L in lookbacks:
        ret_L = logc - np.concatenate([np.full(L, np.nan), logc[:-L]])
        vol_L = s.rolling(L).std(ddof=0).to_numpy() * np.sqrt(L)
        with np.errstate(invalid="ignore", divide="ignore"):
            leg = np.where(vol_L > 0, ret_L / vol_L, np.nan)
        ok = np.isfinite(leg)
        scores = np.where(ok, scores + np.nan_to_num(leg), scores)
        counts = counts + ok.astype(float)
    score = np.where(counts > 0, scores / np.maximum(counts, 1), 0.0)
    sig = np.where(score > thr, 1, np.where(score < -thr, -1, 0)).astype(int)
    return sig


def _rsi(close: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(close)
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    # Wilder via simple rolling mean (n small for RSI2); causal.
    ru = up.rolling(n).mean()
    rd = dn.rolling(n).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).to_numpy()


def sig_rsi2(df: pd.DataFrame, n: int, lo: float, hi: float) -> np.ndarray:
    """RSI(n) mean-reversion: long when oversold (<lo), short when overbought (>hi)."""
    rsi = _rsi(df["close"].to_numpy(dtype=float), n)
    sig = np.where(rsi < lo, 1, np.where(rsi > hi, -1, 0))
    sig = np.where(np.isfinite(rsi), sig, 0).astype(int)
    return sig


def sig_zfade(df: pd.DataFrame, win: int, z: float) -> np.ndarray:
    """Z-score fade vs rolling SMA: short when z>+z, long when z<-z. Causal."""
    c = pd.Series(df["close"].to_numpy(dtype=float))
    sma = c.rolling(win).mean()
    sd = c.rolling(win).std(ddof=0)
    zz = ((c - sma) / sd.replace(0, np.nan)).to_numpy()
    sig = np.where(zz > z, -1, np.where(zz < -z, 1, 0))
    sig = np.where(np.isfinite(zz), sig, 0).astype(int)
    return sig


def sig_bbfade(df: pd.DataFrame, win: int, k: float) -> np.ndarray:
    """Bollinger fade: short above upper band, long below lower band (==zfade with z=k)."""
    return sig_zfade(df, win, k)


# ----------------------------- backtest core ------------------------------- #
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


def _forward_logret(close: np.ndarray, horizon: int) -> np.ndarray:
    """log(close[i+h]/close[i]); NaN where i+h out of range."""
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = np.log(close[horizon:] / close[: n - horizon])
    return fwd


def _annualized_sharpe(net: np.ndarray, tpy: float) -> float:
    if len(net) < 3:
        return 0.0
    mu = net.mean()
    sd = net.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(mu / sd * np.sqrt(tpy))


def backtest(df: pd.DataFrame, sig: np.ndarray, horizon: int, tf: str,
             coin: str, now_ms: int) -> Result:
    """Greedy no-overlap: walk bars; on sig!=0 take trade, jump i+=horizon."""
    close = df["close"].to_numpy(dtype=float)
    ts = df["ts"].to_numpy(dtype="int64")
    fwd = _forward_logret(close, horizon)
    n = len(close)
    bpy = BARS_PER_YEAR[tf]

    entry_idx: list[int] = []
    dir_ret: list[float] = []
    i = 0
    while i < n:
        s = sig[i]
        if s != 0 and np.isfinite(fwd[i]):
            entry_idx.append(i)
            dir_ret.append(s * fwd[i])  # directional gross log-return over horizon
            i += horizon  # no overlap: min-gap == horizon
        else:
            i += 1

    if not dir_ret:
        return Result(coin, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    gross = np.array(dir_ret)
    net_t = gross - TAKER
    net_m = gross - MAKER
    entry_ts = ts[np.array(entry_idx)]

    # span in years from first to last ENTRY bar (consistent trade-rate denom)
    span_bars = (entry_idx[-1] - entry_idx[0] + horizon)
    span_years = max(span_bars / bpy, 1e-9)
    tpy = len(gross) / span_years

    sharpe_t = _annualized_sharpe(net_t, tpy)
    sharpe_m = _annualized_sharpe(net_m, tpy)

    # OOS-40%: last 40% of trades chronologically
    k = len(net_t)
    oos_start = int(np.floor(k * 0.6))
    oos = net_t[oos_start:]
    oos_sharpe_t = _annualized_sharpe(oos, tpy) if len(oos) >= 3 else 0.0

    # 3 expanding folds: cumulative first 1/3, 2/3, 3/3 -> mean(net)>0 fraction
    folds = []
    for frac in (1 / 3, 2 / 3, 1.0):
        m = max(3, int(np.floor(k * frac)))
        seg = net_t[:m]
        folds.append(1.0 if seg.mean() > 0 else 0.0)
    fold_pos = float(np.mean(folds))

    # last 3 days (72h) by entry timestamp
    cutoff = now_ms - 3 * 24 * 3600 * 1000
    mask3 = entry_ts >= cutoff
    l3_n = int(mask3.sum())
    l3_pnl_t = float(net_t[mask3].sum()) if l3_n else 0.0
    l3_pnl_m = float(net_m[mask3].sum()) if l3_n else 0.0

    return Result(
        coin, len(gross), tpy, float(net_t.mean()), float(net_m.mean()),
        sharpe_t, sharpe_m, oos_sharpe_t, fold_pos, l3_n, l3_pnl_t, l3_pnl_m,
    )


# ------------------------------- config sweep ------------------------------ #
def gate_pass(r: Result) -> bool:
    return (r.net_per_trade_taker > 0 and r.oos40_sharpe_taker > 0.3
            and r.fold_pos_frac_taker >= 0.67 and r.trades_per_year >= 150)


@dataclass
class Family:
    name: str
    builder: callable
    configs: list[dict]  # each: {params..., horizon, tf}


def run_family(fam: Family, now_ms: int) -> list[dict]:
    rows = []
    for cfg in fam.configs:
        tf = cfg["tf"]
        horizon = cfg["horizon"]
        params = {k: v for k, v in cfg.items() if k not in ("tf", "horizon")}
        per_asset: list[Result] = []
        for coin in PAIRS:
            try:
                df = load(coin, tf)
            except FileNotFoundError:
                continue
            sig = fam.builder(df, **params)
            r = backtest(df, sig, horizon, tf, coin, now_ms)
            per_asset.append(r)
        n_pass = sum(gate_pass(r) for r in per_asset)
        # aggregate (median over assets that traded)
        traded = [r for r in per_asset if r.n_trades >= 5]
        med = lambda key: float(np.median([getattr(r, key) for r in traded])) if traded else 0.0
        l3_trades = sum(r.last3d_trades for r in per_asset)
        l3_pnl_t = sum(r.last3d_pnl_taker for r in per_asset)
        rows.append({
            "family": fam.name,
            "cfg": {**params, "tf": tf, "horizon": horizon},
            "n_pass": n_pass,
            "passes_cross_asset": n_pass >= 4,
            "med_tpy": med("trades_per_year"),
            "med_net_taker": med("net_per_trade_taker"),
            "med_net_maker": med("net_per_trade_maker"),
            "med_sharpe_taker": med("sharpe_taker"),
            "med_oos40_taker": med("oos40_sharpe_taker"),
            "med_fold_pos": med("fold_pos_frac_taker"),
            "last3d_trades": l3_trades,
            "last3d_pnl_taker": l3_pnl_t,
            "per_asset": [r.__dict__ for r in per_asset],
        })
    return rows


def main():
    # global now from the latest bar across all 1h files
    latest = 0
    for coin in PAIRS:
        try:
            df = load(coin, "1h")
            latest = max(latest, int(df["ts"].iloc[-1]))
        except FileNotFoundError:
            pass
    now_ms = latest
    print(f"now_ms = {now_ms} ({pd.to_datetime(now_ms, unit='ms')})", file=sys.stderr)

    families: list[Family] = []

    # ---- Fast vol-scaled momentum (the validated edge, faster cadence) ---- #
    volmom_cfgs = []
    for tf in ("1h", "2h"):
        for lbs in [(12, 24, 48, 96), (6, 12, 24, 48), (24, 48, 96, 192), (8, 16, 32, 64)]:
            for thr in (0.1, 0.3, 0.5, 1.0):
                for h in ([6, 12, 24] if tf == "1h" else [3, 6, 12]):
                    volmom_cfgs.append({"lookbacks": lbs, "thr": thr, "tf": tf, "horizon": h})
    families.append(Family("volmom_fast", sig_volmom, volmom_cfgs))

    # ---- Intraday mean-reversion CONTROLS (expected to FAIL per +autocorr) - #
    rsi_cfgs = []
    for tf in ("15m", "1h"):
        for n in (2, 3):
            for (lo, hi) in [(10, 90), (5, 95), (15, 85)]:
                for h in ([4, 8, 16] if tf == "15m" else [2, 4, 8]):
                    rsi_cfgs.append({"n": n, "lo": lo, "hi": hi, "tf": tf, "horizon": h})
    families.append(Family("rsi2_mr", sig_rsi2, rsi_cfgs))

    z_cfgs = []
    for tf in ("15m", "1h"):
        for win in (20, 40, 96):
            for z in (1.0, 1.5, 2.0, 2.5):
                for h in ([4, 8, 16] if tf == "15m" else [2, 4, 8]):
                    z_cfgs.append({"win": win, "z": z, "tf": tf, "horizon": h})
    families.append(Family("zscore_fade_mr", sig_zfade, z_cfgs))

    bb_cfgs = []
    for tf in ("15m", "1h"):
        for win in (20, 30):
            for k in (2.0, 2.5):
                for h in ([4, 8] if tf == "15m" else [2, 4]):
                    bb_cfgs.append({"win": win, "k": k, "tf": tf, "horizon": h})
    families.append(Family("bollinger_fade_mr", sig_bbfade, bb_cfgs))

    all_rows = []
    for fam in families:
        # 15m files may not exist yet — skip configs that can't load
        rows = run_family(fam, now_ms)
        all_rows.extend(rows)
        # print best of family by med_oos40 among cross-asset passers, else overall
        passers = [r for r in rows if r["passes_cross_asset"]]
        pool = passers if passers else rows
        best = max(pool, key=lambda r: (r["n_pass"], r["med_oos40_taker"]))
        print(f"\n=== {fam.name} === ({len(rows)} cfgs, {len(passers)} cross-asset passers)",
              file=sys.stderr)
        print(f"  BEST cfg={best['cfg']} n_pass={best['n_pass']}/7 "
              f"tpy={best['med_tpy']:.0f} net_t={best['med_net_taker']*1e4:.1f}bps "
              f"sharpe_t={best['med_sharpe_taker']:.2f} oos40={best['med_oos40_taker']:.2f} "
              f"foldpos={best['med_fold_pos']:.2f} l3_trades={best['last3d_trades']} "
              f"l3_pnl_t={best['last3d_pnl_taker']*1e4:.1f}bps", file=sys.stderr)

    out = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/intraday/_results.json")
    out.write_text(json.dumps(all_rows, indent=2, default=str))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
