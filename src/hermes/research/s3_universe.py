"""Stream 3 — universe expansion + per-instrument calibration.

This module does NOT create a signal. It takes the ALREADY-VALIDATED Strategy #2
sleeve (``funding_squeeze_continuation``) and asks the only two honest questions a
"arm more often" mandate is allowed to ask:

  1. **Generalisation** — does the *same frozen rule* (funding_z ≥ 2 over a 180-bar
     baseline, long, ~horizon-bar hold) survive a per-asset random-entry MC-null on
     the perps BEYOND BTC/ETH/SOL, and does the *ensemble* survive a dated joint
     book-null?  A mono-instrument "win" is rejected by default (cross-asset is the
     killer filter, METAPROMPT §3.4).
  2. **Calibration** — is the single ``funding_z`` threshold already the right
     per-instrument bar, or can a per-asset threshold arm "at the right place"
     WITHOUT raising the false-trade rate?  "Better calibrated" must never collapse
     into "more trades" — that is whipsaw (the dead-end the ledger warns about).

Everything reuses the committed machinery — ``evaluate_signal``/``triple_barrier``
for labelling net of fees, ``validation.book.combine_book`` for the causal book,
``learning.stats.sharpe_ratio`` for the metric. The only new code is (a) a faithful
trade *walk* that additionally records entry context (proven byte-equivalent to
``evaluate_signal`` by a unit test), (b) the random-entry MC nulls, and (c) a
threshold-parameterised view of the frozen rule for the calibration table.

Cadence note (PROVENANCE): the frozen S2 rule was validated on 4h bars (hold
24x4h ~= 96h). The frozen lake only carries ``1d``/``6h``; ``6h`` funding starts
2025-07-01 (~1y, too short for a 180-bar baseline + an honest event count). So the
honest cadence is **1d**, where the faithful translation of "~96h" is a **4-bar**
hold. The rule (window=180, z=2.0) is UNCHANGED — only the bar size is forced by
the data. This is a translation, not a re-tuning.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from hermes.learning.stats import sharpe_ratio
from hermes.research.funding import attach_funding
from hermes.research.signals import _atr_pct, _ztail
from hermes.skills.library import SkillSignal
from hermes.validation.book import combine_book
from hermes.validation.triple_barrier import triple_barrier_label

SignalFn = Callable[[pd.DataFrame], SkillSignal]

__all__ = [
    "arm_threshold_native",
    "bear_bull_split",
    "build_pair_frame",
    "calibration_row",
    "funding_squeeze_at_z",
    "joint_dated_null",
    "load_universe",
    "normalize_symbol",
    "per_asset_null_pvalue",
    "walk_trades",
]


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
def normalize_symbol(symbol: str) -> str:
    """Unify the two lake spellings: funding uses ``BTCUSDT``, klines ``BTC/USDT``."""
    return symbol.replace("/", "").upper()


def load_universe(
    duckdb_path: str, *, interval: str = "1d", symbols: tuple[str, ...] | None = None
) -> dict[str, pd.DataFrame]:
    """Build the per-pair frames from the frozen lake (read-only DuckDB).

    Joins ``binance_klines`` (OHLCV) to ``coinglass_funding_rate`` (funding in its
    ``close`` column) per normalised symbol at ``interval``. Returns {symbol -> frame}.
    Thin I/O glue over the unit-tested ``build_pair_frame``.
    """
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        kl = con.execute(
            "SELECT ts,open,high,low,close,volume,symbol FROM binance_klines "
            "WHERE interval = ?",
            [interval],
        ).fetchdf()
        fr = con.execute(
            "SELECT ts,close,symbol FROM coinglass_funding_rate WHERE interval = ?",
            [interval],
        ).fetchdf()
    finally:
        con.close()
    kl["key"] = kl["symbol"].map(normalize_symbol)
    fr["key"] = fr["symbol"].map(normalize_symbol)
    out: dict[str, pd.DataFrame] = {}
    for key in sorted(kl["key"].unique()):
        if symbols is not None and key not in symbols:
            continue
        k = kl[kl["key"] == key]
        f = fr[fr["key"] == key]
        if f.empty:
            continue
        out[key] = build_pair_frame(
            k[["ts", "open", "high", "low", "close", "volume", "symbol"]],
            f[["ts", "close", "symbol"]],
        )
    return out


def build_pair_frame(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Causal per-pair frame: klines OHLCV + as-of funding rate.

    ``funding`` is the ``coinglass_funding_rate`` view (the funding value lives in
    its ``close`` column). Each bar inherits the most recent funding settled at or
    before the bar's open (``attach_funding`` — strict backward as-of, no leak);
    bars before the first settlement get a neutral 0.0. ``ts`` is returned as a
    UTC datetime (what the harness expects); the symbol is normalised.
    """
    kl = klines.copy()
    kl["symbol"] = kl["symbol"].map(normalize_symbol)
    fr = funding[["ts", "close"]].rename(columns={"close": "funding_rate"})
    merged = attach_funding(kl, fr)
    merged["ts"] = pd.to_datetime(merged["ts"], unit="ms", utc=True)
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the frozen rule, threshold-parameterised (calibration probe — NOT a new signal)
# --------------------------------------------------------------------------- #
def funding_squeeze_at_z(
    df: pd.DataFrame, z_thresh: float = 2.0, window: int = 180
) -> SkillSignal:
    """``funding_squeeze_continuation`` with the arm threshold exposed.

    At the frozen point ``(z_thresh=2.0, window=180)`` this is byte-identical to
    the live sleeve (proven by a unit test) — it exists ONLY so the calibration
    table can sweep the *threshold* of the *same* rule. Everything else (the
    180-bar z baseline, long-only direction, conviction = min(1, z/4), the
    atr-scaled wide barriers so the time-exit dominates) is frozen.
    """
    sid = f"funding_squeeze@z{z_thresh:g}"
    if "funding_rate" not in df.columns or len(df) < window + 1:
        return SkillSignal(sid, "HOLD", 0.0, 0.02, 0.04, "insufficient", {})
    fz = _ztail(df["funding_rate"], window)
    if fz < z_thresh:
        return SkillSignal(
            sid, "HOLD", 0.0, 0.02, 0.04,
            f"funding z={fz:.2f} < {z_thresh}", {"funding_z": float(fz)},
        )
    atrpct = float(_atr_pct(df).tail(14).mean())
    if not (np.isfinite(atrpct) and atrpct > 0):
        atrpct = 0.02
    conf = min(1.0, fz / 4.0)
    return SkillSignal(
        sid, "BUY", conf, max(0.01, 4.0 * atrpct), max(0.02, 8.0 * atrpct),
        f"funding squeeze z={fz:.2f} (>= {z_thresh})",
        {"funding_z": float(fz), "z_thresh": float(z_thresh)},
    )


# --------------------------------------------------------------------------- #
# faithful trade walk (== committed evaluate_signal, + entry context)
# --------------------------------------------------------------------------- #
def walk_trades(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    horizon: int,
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    warmup: int = 200,
    window_cap: int = 400,
    no_overlap: bool = True,
    ma_len: int = 200,
) -> list[dict]:
    """Chronological trade ledger with per-trade context.

    Mirrors ``hermes.research.harness.evaluate_signal`` EXACTLY (same loop, same
    no-overlap one-position book, same net-of-fee triple-barrier) — a unit test
    asserts the ``(event_ns, ret)`` stream is identical — and additionally records
    the entry index and whether the entry bar sits below its own MA(``ma_len``),
    so the bull/bear drift-beta diagnostic needs no second, divergent walk.
    """
    rt = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    df = df.reset_index(drop=True)
    highs, lows, closes = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
    has_ts = "ts" in df.columns
    ma = pd.Series(closes, dtype=float).rolling(ma_len).mean()
    n = len(df)
    out: list[dict] = []
    busy = -1
    for t in range(warmup, n - 1):
        lo = max(0, t + 1 - window_cap)
        sig = signal_fn(df.iloc[lo : t + 1])
        if sig.side == "HOLD":
            continue
        if no_overlap and t < busy:
            continue
        if no_overlap:
            busy = t + horizon
        _, gross = triple_barrier_label(
            highs, lows, closes, t, closes[t], sig.side,
            sig.stop_pct, sig.take_profit_pct, horizon,
        )
        ev = int(pd.Timestamp(df["ts"].iloc[t]).value) if has_ts else t
        ma_t = ma.iloc[t]
        out.append(
            {
                "event_ns": ev,
                "ret": float(gross - rt),
                "entry_idx": t,
                "below_ma": bool(np.isfinite(ma_t) and closes[t] < ma_t),
            }
        )
    out.sort(key=lambda r: r["event_ns"])
    return out


# --------------------------------------------------------------------------- #
# per-asset random-entry MC null
# --------------------------------------------------------------------------- #
def _atrpct_full(df: pd.DataFrame) -> np.ndarray:
    """The sleeve's atr% (tail-14 mean of true-range/close) at every bar.

    Equivalent to the per-window ``_atr_pct(window).tail(14).mean()`` the sleeve
    computes, for any entry ``t`` whose 14-bar window lies inside ``window_cap`` —
    precomputed once so the bootstrap is cheap.
    """
    return _atr_pct(df).rolling(14).mean().to_numpy(dtype=float)


def _long_barrier(atrpct: float) -> tuple[float, float]:
    if not (np.isfinite(atrpct) and atrpct > 0):
        atrpct = 0.02
    return max(0.01, 4.0 * atrpct), max(0.02, 8.0 * atrpct)


def per_asset_null_pvalue(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    horizon: int,
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    warmup: int = 200,
    window_cap: int = 400,
    no_overlap: bool = True,
    n_boot: int = 2000,
    rng: np.random.Generator,
) -> dict:
    """Random-entry MC null for one instrument (the METAPROMPT §3.5 floor).

    Observed = the sleeve's mean net per-trade return. Null = the same number of
    LONG entries placed at RANDOM bars, same horizon, same atr-scaled barrier
    policy — so the only thing that differs is the *funding timing*. Beating it
    means funding carries information beyond a random long entry, which ALREADY
    captures any secular drift; so this is also a drift-beta control. One-sided
    ``p = (1 + #{null_mean ≥ obs_mean}) / (n_boot + 1)``.
    """
    rt = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    recs = walk_trades(
        df, signal_fn, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps,
        warmup=warmup, window_cap=window_cap, no_overlap=no_overlap,
    )
    n_trades = len(recs)
    if n_trades == 0:
        return {"n_trades": 0, "obs_mean": 0.0, "null_mean": 0.0,
                "null_q95": 0.0, "p_value": 1.0}
    obs_mean = float(np.mean([r["ret"] for r in recs]))

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    ap = _atrpct_full(df)
    n = len(df)
    lo_idx, hi_idx = warmup, n - 2  # entries the harness could place
    if hi_idx < lo_idx:
        return {"n_trades": n_trades, "obs_mean": obs_mean, "null_mean": 0.0,
                "null_q95": 0.0, "p_value": 1.0}

    null_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        entries = rng.integers(lo_idx, hi_idx + 1, size=n_trades)
        rets = np.empty(n_trades, dtype=float)
        for i, t in enumerate(entries):
            stop, tp = _long_barrier(ap[t])
            _, gross = triple_barrier_label(
                highs, lows, closes, int(t), closes[t], "BUY", stop, tp, horizon
            )
            rets[i] = gross - rt
        null_means[b] = rets.mean()

    p = (1 + int(np.sum(null_means >= obs_mean))) / (n_boot + 1)
    return {
        "n_trades": n_trades,
        "obs_mean": obs_mean,
        "null_mean": float(null_means.mean()),
        "null_q95": float(np.quantile(null_means, 0.95)),
        "p_value": float(p),
    }


# --------------------------------------------------------------------------- #
# bull/bear drift-beta diagnostic
# --------------------------------------------------------------------------- #
def bear_bull_split(records: list[dict]) -> dict:
    """Split a trade ledger by whether the entry bar was below its MA200.

    A long-only funding edge that only earns ABOVE the MA200 (bull) and is flat /
    negative BELOW it (bear) is at risk of being secular-drift beta in disguise —
    this surfaces that, complementing the random-entry null.
    """
    bull = [r["ret"] for r in records if not r["below_ma"]]
    bear = [r["ret"] for r in records if r["below_ma"]]
    return {
        "n_bull": len(bull),
        "n_bear": len(bear),
        "bull_mean": float(np.mean(bull)) if bull else 0.0,
        "bear_mean": float(np.mean(bear)) if bear else 0.0,
    }


# --------------------------------------------------------------------------- #
# calibration (vector 3.2)
# --------------------------------------------------------------------------- #
def arm_threshold_native(df: pd.DataFrame, z: float = 2.0, window: int = 180) -> float:
    """The funding level (NATIVE units) at which ``funding_z = z`` arms, for this
    instrument: ``mean + z*sigma`` over the trailing ``window`` (population sigma,
    matching ``_ztail``). This is the transparency number for the per-instrument
    table — what "2 sigma" actually means for each pair, NOT a lowered bar."""
    tail = df["funding_rate"].tail(window).to_numpy(dtype=float)
    tail = tail[np.isfinite(tail)]
    if tail.size < 2:
        return float("nan")
    return float(tail.mean() + z * tail.std(ddof=0))


def calibration_row(
    df: pd.DataFrame,
    z_thresh: float,
    split_frac: float,
    *,
    horizon: int,
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
) -> dict:
    """One row of the calibration table for one instrument at one threshold.

    Walk the *whole* frame once (so every trade keeps its real trailing-180 funding
    context) and split trades into in-sample / out-of-sample by entry index at
    ``split_frac``. Reports n / win-rate / per-trade Sharpe on each side, so the
    table can show whether a lower threshold buys more trades at the cost of a
    degraded per-trade edge (whipsaw) — the §3.2 KILL test.
    """
    fn: SignalFn = lambda w: funding_squeeze_at_z(w, z_thresh=z_thresh)  # noqa: E731
    recs = walk_trades(
        df, fn, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps
    )
    cut = int(len(df) * split_frac)
    is_r = [r["ret"] for r in recs if r["entry_idx"] < cut]
    oos_r = [r["ret"] for r in recs if r["entry_idx"] >= cut]

    def _stat(rets: list[float]) -> dict:
        return {
            "n": len(rets),
            "win_rate": float(np.mean([x > 0 for x in rets])) if rets else 0.0,
            "sharpe": sharpe_ratio(rets),
            "mean_bps": float(np.mean(rets) * 1e4) if rets else 0.0,
        }

    return {"z": z_thresh, "is": _stat(is_r), "oos": _stat(oos_r)}


# --------------------------------------------------------------------------- #
# dated joint book null (the ensemble gate, vector 3.1)
# --------------------------------------------------------------------------- #
def _daily_pnl(records: list[dict], days: pd.Series) -> pd.Series:
    """Per-calendar-day summed net PnL (de-overlaps intra-asset clustered events —
    the 'dated' in 'joint dated null')."""
    if not records:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([days.iloc[r["entry_idx"]] for r in records])
    return pd.Series([r["ret"] for r in records], index=idx).groupby(level=0).sum()


def joint_dated_null(
    df_by_symbol: dict[str, pd.DataFrame],
    *,
    signal_fn: SignalFn,
    horizon: int,
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    warmup: int = 200,
    window_cap: int = 400,
    no_overlap: bool = True,
    n_boot: int = 1000,
    rng: np.random.Generator,
    weighting: str = "inv_vol",
) -> dict:
    """Dated joint null over the ENSEMBLE of instruments (cross-asset = killer filter).

    Observed = Sharpe of the causal combined BOOK (``combine_book``) of every
    instrument's daily PnL. Null = rebuild each instrument's trades from the same
    number of random LONG entries, combine into a null book the SAME way, and take
    its Sharpe; repeat. ``p = (1 + #{null_sharpe ≥ obs_sharpe}) / (n_boot+1)``.
    Tests whether the *ensemble* result beats chance after the daily de-overlap and
    the cross-asset combination — not just one lucky leg.
    """
    rt = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    obs_daily: dict[str, pd.Series] = {}
    ctx: dict[str, dict] = {}
    for sym, df in df_by_symbol.items():
        recs = walk_trades(
            df, signal_fn, horizon=horizon, fee_bps=fee_bps, slippage_bps=slippage_bps,
            warmup=warmup, window_cap=window_cap, no_overlap=no_overlap,
        )
        days = df["ts"].dt.floor("D")
        obs_daily[sym] = _daily_pnl(recs, days)
        ctx[sym] = {
            "n_trades": len(recs),
            "highs": df["high"].to_numpy(dtype=float),
            "lows": df["low"].to_numpy(dtype=float),
            "closes": df["close"].to_numpy(dtype=float),
            "ap": _atrpct_full(df),
            "days": days.to_numpy(),
            "n": len(df),
        }

    obs_book = combine_book(obs_daily, weighting=weighting)
    obs_sharpe = sharpe_ratio(obs_book.tolist())

    null_sharpes = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        null_daily: dict[str, pd.Series] = {}
        for sym, c in ctx.items():
            k = c["n_trades"]
            if k == 0:
                null_daily[sym] = pd.Series(dtype=float)
                continue
            lo_idx, hi_idx = warmup, c["n"] - 2
            entries = rng.integers(lo_idx, hi_idx + 1, size=k)
            rets, idx = np.empty(k, dtype=float), []
            for i, t in enumerate(entries):
                stop, tp = _long_barrier(c["ap"][t])
                _, gross = triple_barrier_label(
                    c["highs"], c["lows"], c["closes"], int(t),
                    c["closes"][t], "BUY", stop, tp, horizon,
                )
                rets[i] = gross - rt
                idx.append(c["days"][t])
            s = pd.Series(rets, index=pd.DatetimeIndex(idx)).groupby(level=0).sum()
            null_daily[sym] = s
        nb = combine_book(null_daily, weighting=weighting)
        null_sharpes[b] = sharpe_ratio(nb.tolist())

    p = (1 + int(np.sum(null_sharpes >= obs_sharpe))) / (n_boot + 1)
    return {
        "obs_sharpe": float(obs_sharpe),
        "null_sharpe_mean": float(null_sharpes.mean()),
        "p_value": float(p),
        "n_boot": n_boot,
    }
