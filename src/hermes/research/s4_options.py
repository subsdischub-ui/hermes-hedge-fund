"""Stream 4 — crypto options surface (skew 25Δ, dealer-GEX, VRP-DVOL) research primitives.

Data reality is the spine of this module (Amberdata not bought):
  * `deribit_dvol` is BACKFILLABLE (BTC/ETH hourly, 2021-03-24→) — the ONLY backtestable
    options input. Vector 4.3 (VRP as a sizing GATE on a confirmed sleeve) lives here.
  * `deribit_surface_snapshot` (25Δ skew/Greeks) is a FORWARD-LOG only — never backfilled,
    never fabricated. Vector 4.1 is a "chrono démarré" verdict + a hardened cron, not a backtest.
  * `coinglass_options_oi` is AGGREGATE (no strikes, no put/call) — vector 4.2 can only be a
    coarse, UNSIGNED OI-regime CONDITIONER, never a signed dealer-GEX.

Design law (lab journal): the recurring bug is look-ahead, and a sizing gate is read AT trade
entry, so every primitive is causal by construction — DVOL lagged +1 bar, realized vol TRAILING
(never forward). The gate statistics (Sharpe/DSR/PBO/joint-null) are NOT re-implemented here:
they are the committed machinery (`hermes.validation.book`, `hermes.learning.stats`,
`research_fx.fxgate`). Pure functions; the lake I/O is a thin, separately-tested seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Reuse the already-tested causal z-score rather than rebuild it (DRY; tested in
# tests/research/test_s2_positioning.py).
from hermes.research.s2_positioning import causal_zscore

__all__ = [
    "aggregate_options_oi",
    "bs_delta",
    "compute_vrp",
    "dedup_forward_log",
    "gate_trade_returns",
    "gex_oi_regime",
    "oos_mask",
    "realized_vol_cc",
    "realized_vol_parkinson",
    "size_multiplier",
    "vrp_zscore",
]


# --------------------------------------------------------------------------- #
# 4.3 — realized-vol estimators (both annualized to VOL POINTS, same convention
#       as DVOL, so VRP = implied − realized is homogeneous)
# --------------------------------------------------------------------------- #
def realized_vol_cc(
    close: pd.Series, window: int = 30, periods_per_year: float = 365.0
) -> pd.Series:
    """Trailing close-to-close realized vol, annualized to vol points (%).

    ``rv_t = std(logret over the trailing ``window`` bars, ddof=1) * sqrt(ppy) * 100``.
    Causal — the rolling window only looks backward, so ``rv_t`` depends on ``close[..t]``
    alone (the no-look-ahead guarantee the gate's VRP input must satisfy). The first
    ``window`` bars are NaN (warmup); a flat window yields 0.0.
    """
    c = pd.Series(close, dtype=float)
    logret = np.log(c / c.shift(1))
    sd = logret.rolling(window).std(ddof=1)
    return sd * np.sqrt(periods_per_year) * 100.0


def realized_vol_parkinson(
    high: pd.Series, low: pd.Series, window: int = 30, periods_per_year: float = 365.0
) -> pd.Series:
    """Trailing Parkinson (high-low range) realized vol, same units as ``realized_vol_cc``.

    ``pk_t = sqrt( mean( ln(H/L)^2 over window ) / (4 ln2) * ppy ) * 100``. A lower-variance
    estimator than close-to-close (uses the intrabar range), kept as the cross-check so the
    gate verdict cannot depend on the estimator choice. Causal (trailing window).
    """
    h = pd.Series(high, dtype=float)
    lo = pd.Series(low, dtype=float)
    hl2 = np.log(h / lo) ** 2
    mean_hl2 = hl2.rolling(window).mean()
    var = mean_hl2 / (4.0 * np.log(2.0))
    return np.sqrt(var * periods_per_year) * 100.0


def compute_vrp(
    dvol: pd.Series, realized: pd.Series, dvol_lag: int = 1
) -> pd.Series:
    """Causal VRP = implied minus realized, both already in vol points.

    ``VRP_t = DVOL_{t-dvol_lag} - realized_t``. DVOL is lagged +1 bar (the invariant: its ts
    is open-stamped and it must be knowable at the decision bar); ``realized`` is TRAILING
    (built from ``close[..t]``). So the VRP used to size a trade at entry is knowable at entry
    — zero future leak. The ex-post (forward-realized) VRP is a diagnostic only and is NEVER
    fed to the gate. Index-aligned; the first ``dvol_lag`` bars are NaN.
    """
    d = pd.Series(dvol, dtype=float)
    r = pd.Series(realized, dtype=float)
    return d.shift(dvol_lag) - r


def vrp_zscore(vrp: pd.Series, window: int = 365) -> pd.Series:
    """Trailing z-score of the VRP (the stationary gate input). Thin wrapper over the
    committed ``causal_zscore`` so the look-ahead guarantee is the one already tested."""
    return causal_zscore(pd.Series(vrp, dtype=float), window)


# --------------------------------------------------------------------------- #
# 4.3 — the FOUR pre-registered, frozen gate forms (see research_streams/S4/FINDINGS.md).
#       The multiplier centres on 1.0 so the test is conditional TIMING, not leverage
#       (Sharpe is invariant to a constant rescale of the whole multiplier series).
# --------------------------------------------------------------------------- #
def size_multiplier(
    vrp_z: pd.Series,
    form: str,
    sign: int,
    *,
    beta: float = 0.25,
    lo: float = 0.5,
    hi: float = 1.5,
    m_low: float = 0.5,
    m_high: float = 1.5,
    threshold: float = 0.0,
) -> pd.Series:
    """Causal per-bar size multiplier from the VRP z-score (one of the 4 frozen forms).

    ``form='binary'``: ``m = m_high`` in the "up" regime, else ``m_low``. The ``sign=+1``
    up-regime is ``VRP_z >= threshold`` (the mandate's *fear→up* prior); ``sign=-1`` is its
    STRICT complement ``VRP_z < threshold`` (the opposite hypothesis), so the two signs
    partition the line and genuinely test opposite directions.

    ``form='linear'``: ``m = clip(1 + sign*beta*VRP_z, lo, hi)`` — a centred, bounded
    response; ``sign`` flips the slope.

    Warmup (NaN ``VRP_z``) is NEUTRAL 1.0 — never dropped — so the gated series keeps every
    base trade (gating must change SIZE, not the trade set). Params are frozen a-priori.
    """
    z = pd.Series(vrp_z, dtype=float)
    if form == "binary":
        up = z >= threshold  # the sign=+1 ("fear→up") region; NaN -> False
        if sign < 0:
            up = ~up
        m = pd.Series(np.where(up, m_high, m_low), index=z.index, dtype=float)
    elif form == "linear":
        s = 1.0 if sign >= 0 else -1.0
        m = (1.0 + s * beta * z).clip(lo, hi)
    else:
        raise ValueError(f"unknown form: {form!r} (use 'binary' or 'linear')")
    return m.where(z.notna(), 1.0)


# --------------------------------------------------------------------------- #
# 4.3 — apply the gate to a confirmed sleeve's per-trade returns. Per-position
#       sizing: gated_i = m_i * ret_net_i, where m_i is the multiplier from the
#       LAST VRP_z timestamped at/before trade i's entry (no future leak). Cost is
#       already in ret_net at unit size; scaling the position scales PnL and its
#       round-trip cost identically, so there is no incremental turnover term — and
#       Sharpe is scale-invariant, so this tests conditional TIMING, not leverage.
# --------------------------------------------------------------------------- #
def gate_trade_returns(
    entry_ts_ms,
    ret_net,
    vrp_z: pd.Series,
    vrp_z_ts_ms,
    form: str,
    sign: int,
    **kw,
) -> np.ndarray:
    """Scale each trade's net return by the causal VRP size multiplier at its entry.

    For trade *i* entered at ``entry_ts_ms[i]``, the multiplier is ``size_multiplier`` of the
    LAST ``vrp_z`` whose timestamp is <= the entry (an as-of-backward lookup) — so a trade
    never sees a VRP_z value it could not have known at entry. A trade entered before the
    first VRP_z is NEUTRAL (m=1.0), never dropped. ``vrp_z`` must already be
    publication-aligned by the caller (its value at ``vrp_z_ts_ms[k]`` knowable at that ts).
    """
    m_series = size_multiplier(pd.Series(vrp_z, dtype=float), form, sign, **kw)
    z_ts = np.asarray(vrp_z_ts_ms)
    order = np.argsort(z_ts, kind="stable")
    z_ts_sorted = z_ts[order]
    m_sorted = m_series.to_numpy()[order]
    entries = np.asarray(entry_ts_ms)
    rets = np.asarray(ret_net, dtype=float)
    gated = np.empty(rets.shape[0], dtype=float)
    for i, ets in enumerate(entries):
        j = int(np.searchsorted(z_ts_sorted, ets, side="right")) - 1
        m = float(m_sorted[j]) if j >= 0 else 1.0
        gated[i] = m * rets[i]
    return gated


def oos_mask(entry_ts_ms, frac: float = 0.4) -> np.ndarray:
    """Boolean held-out mask: the last ``frac`` of the timeline BY TIME (not by count).

    cutoff = min_ts + (1-frac)*(max_ts-min_ts); a trade is OOS iff its entry >= cutoff. Keyed
    on the timestamp VALUE, so unsorted input is handled correctly. This is the KILL split:
    the gate must lift the sleeve's deflated Sharpe on exactly this out-of-sample tail.
    """
    ts = np.asarray(entry_ts_ms, dtype=float)
    if ts.size == 0:
        return np.zeros(0, dtype=bool)
    lo, hi = float(ts.min()), float(ts.max())
    if hi <= lo:
        return np.zeros(ts.shape[0], dtype=bool)
    cutoff = lo + (1.0 - frac) * (hi - lo)
    return ts >= cutoff


# --------------------------------------------------------------------------- #
# 4.2 — aggregate-OI dealer-GEX REGIME proxy (UNSIGNED). CoinGlass options OI is
#       aggregate-per-exchange only: NO strikes, NO put/call split, NO Greeks, so a
#       SIGNED dealer-GEX is unconstructable. This is a coarse OI-magnitude regime
#       CONDITIONER, never a standalone signal. Default to Deribit+OKX (full history
#       from 2020-06-24) to avoid the structural break when Binance/Bybit join in 2023.
# --------------------------------------------------------------------------- #
def aggregate_options_oi(
    oi_long: pd.DataFrame, exchanges: tuple[str, ...] = ("Deribit", "OKX")
) -> pd.Series:
    """Daily aggregate options OI = sum of ``value`` over the selected exchanges per ts.

    Filters to ``exchanges`` (excludes the 2023-joining venues by default), drops NaN
    ``value`` rows BEFORE summing (so a missing venue never NaN-propagates the day), and
    groups by ``ts``. Returns a Series indexed by ts (epoch-ms), sorted.
    """
    df = oi_long[oi_long["exchange"].isin(exchanges)]
    df = df.dropna(subset=["value"])
    return df.groupby("ts")["value"].sum().sort_index()


def gex_oi_regime(
    oi_long: pd.DataFrame,
    exchanges: tuple[str, ...] = ("Deribit", "OKX"),
    window: int = 90,
) -> pd.Series:
    """UNSIGNED OI-magnitude regime = causal z-score of the aggregate options OI.

    High |z| = an unusually large options OI base (a regime where dealer hedging flow is
    structurally larger), low/negative = a thin-OI regime. It carries NO sign (cannot
    distinguish dealer long- vs short-gamma without strikes/put-call), so it is a regime
    CONDITIONER only. Causal via ``causal_zscore`` (trailing window).
    """
    agg = aggregate_options_oi(oi_long, exchanges)
    return causal_zscore(agg, window)


# --------------------------------------------------------------------------- #
# 4.1 — forward-log cron robustness. The 25Δ skew/RR goes NaN whenever live ticker
#       greeks are unavailable (it bit the lake: rr25_1m/skew_1m = NaN). A
#       Black-Scholes delta computed from (S, K, t, iv) needs NO network, so the
#       wings can always bracket |delta|=0.25 — the forward-log populates offline.
#       This NEVER fabricates history: it only makes each forward snapshot reliable.
# --------------------------------------------------------------------------- #
def bs_delta(
    option_type: str, S: float, K: float, t_years: float, iv: float, r: float = 0.0
) -> float | None:
    """Black-Scholes spot delta (r=0 default). None for a degenerate option.

    ``d1 = (ln(S/K) + (r + iv^2/2)*t) / (iv*sqrt(t))``; call delta = ``N(d1)`` ∈ (0,1),
    put delta = ``N(d1)-1`` ∈ (-1,0). Returns None when ``t<=0`` or ``iv<=0`` (an expired
    or zero-vol option has no usable smooth delta — the caller skips it, never fabricates).
    ``iv`` is a fraction (0.20 = 20%), not vol points.
    """
    if t_years <= 0.0 or iv <= 0.0 or S <= 0.0 or K <= 0.0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * iv * iv) * t_years) / (iv * np.sqrt(t_years))
    n_d1 = float(norm.cdf(d1))
    return n_d1 if option_type.upper().startswith("C") else n_d1 - 1.0


def dedup_forward_log(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    ts_col: str = "ts",
    key_cols: tuple[str, ...] = ("ccy",),
) -> pd.DataFrame:
    """Idempotent forward-log merge: at most one row per (UTC-day, key) — keep last.

    A daily cron re-run on the same UTC day OVERWRITES that day's row instead of appending
    a duplicate (the legacy ``skew_term_log.csv`` was append-only and would accumulate
    duplicates). Mirrors the lake's ``merge_surface_snapshot`` semantics. ``ts`` is epoch-ms.
    """
    frames = [f for f in (existing, new) if f is not None and len(f)]
    if not frames:
        return new if new is not None else pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    day = pd.to_datetime(df[ts_col], unit="ms", utc=True).dt.floor("D")
    df = df.assign(_utc_day=day)
    df = df.drop_duplicates(subset=["_utc_day", *key_cols], keep="last").drop(columns="_utc_day")
    return df.sort_values(ts_col).reset_index(drop=True)
