"""Stream 2 — positioning & liquidations research primitives (crypto).

The mandate: gate the only open crypto lead — **top-trader LSR contrarian** — on
the now-unlocked CoinGlass history, redo the **liquidation cascade** on REAL data,
and probe **OI-acceleration** funding structure. All three are judged by the
committed machinery (``hermes.validation.book`` LOO-Δ-Sharpe + ``hermes.learning.stats``
DSR/PBO + the ``research_fx.fxgate`` joint-null), never by a solo Sharpe.

Design law inherited from the lab journal: **the recurring bug is look-ahead**, so
every signal here is causal by construction (trailing windows, then an explicit
publication lag) and every primitive is unit-tested for it before use. Pure
functions only; the lake I/O is a thin, separately-tested seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing-window z-score: z_t = (x_t - mean) / std over the ``window`` bars
    ending at and INCLUDING t (std ddof=1).

    Causal by construction — a rolling window only looks backward, so z_t depends
    on x[..t] alone and never on a future bar (the look-ahead guard the lab keeps
    getting bitten by). The first ``window-1`` bars are NaN (warmup), and a flat
    window (std=0) yields NaN rather than ±inf.
    """
    s = pd.Series(series, dtype=float)
    roll = s.rolling(window)
    mean = roll.mean()
    std = roll.std(ddof=1)
    return (s - mean) / std.where(std > 0.0)


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward simple return fwd_t = close[t+horizon]/close[t] - 1.

    The last ``horizon`` bars are NaN — no future bar exists to realise against, so
    they can never enter a backtest. This is the prediction TARGET; the signal that
    predicts it is built only from data at or before t.
    """
    s = pd.Series(close, dtype=float)
    return s.shift(-horizon) / s - 1.0


def backtest_positions(
    position: pd.Series, fwd_ret: pd.Series, cost_per_turnover: float
) -> pd.Series:
    """Realise a causal position series into a net per-bar return stream.

    ``net_t = pos_t * fwd_t - cost_per_turnover * |pos_t - pos_{t-1}|`` — a held
    position pays no new cost, only rebalancing (turnover) does. ``pos_t`` is assumed
    already causal (built from data <= t, then publication-lagged). Bars whose forward
    target is NaN (the tail) are dropped, so the realised series never carries NaN.
    """
    pos = pd.Series(position, dtype=float).reset_index(drop=True)
    fwd = pd.Series(fwd_ret, dtype=float).reset_index(drop=True)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    net = pos * fwd - cost_per_turnover * turnover
    return net[fwd.notna()].reset_index(drop=True)


def tag_regime(
    close: pd.Series,
    window: int = 90,
    bull_thr: float = 0.15,
    bear_thr: float = -0.15,
) -> pd.Series:
    """Causal bull/bear/range label from the trailing ``window``-bar return.

    ``tr_t = close[t]/close[t-window] - 1``; ``tr > bull_thr`` -> 'bull',
    ``tr < bear_thr`` -> 'bear', else 'range'. Uses only past closes (shift is
    backward), so the label at t is knowable at t. The first ``window`` bars have
    no trailing window and are 'unknown' (never silently folded into a regime — that
    would be a look-ahead). This is the self-contained cross-regime axis for the §2.1
    KILL test; the driver also reports the literal 2021/2022/2026 calendar split.
    """
    s = pd.Series(close, dtype=float)
    tr = s / s.shift(window) - 1.0
    out = pd.Series("unknown", index=s.index, dtype=object)
    out[tr > bull_thr] = "bull"
    out[tr < bear_thr] = "bear"
    out[(tr <= bull_thr) & (tr >= bear_thr)] = "range"
    return out


def top_lsr_contrarian_signal(
    lsr: pd.Series,
    window: int,
    mode: str = "continuous",
    clip: float = 2.0,
) -> pd.Series:
    """Vector 2.1: the contrarian position from the top-trader long/short ratio.

    The thesis (H3): when top perp traders are net-long at the EXTREME, the forward
    return is negative (crowding/exhaustion). So the position is the OPPOSITE sign of
    the positioning z-score: ``position = f(-z)``.

    ``mode``:
      * ``"sign"``       -> -sign(z): a discrete ±1 fade (robust, low-DoF).
      * ``"continuous"`` -> (-z / clip) clipped to [-1, 1]: scales the fade with the
        extremity and saturates past ``clip`` sigma (more information, more exposed to
        the z estimate).
    Warmup bars (z is NaN) are FLAT (0.0) — no position, never NaN. The z is causal
    (``causal_zscore``); apply a publication lag (``.shift``) before backtesting.
    """
    z = causal_zscore(lsr, window)
    if mode == "sign":
        return (-np.sign(z)).fillna(0.0)
    if mode == "continuous":
        return (-z / clip).clip(-1.0, 1.0).fillna(0.0)
    raise ValueError(f"unknown mode: {mode!r} (use 'sign' or 'continuous')")


def liquidation_flush_events(
    long_liq: pd.Series,
    short_liq: pd.Series,
    window: int,
    k: float = 3.0,
) -> list[tuple[int, float]]:
    """Vector 2.2: detect liquidation FLUSHES and side them to mean-revert.

    A flush = a bar whose TOTAL liquidation (long+short) z-score over the trailing
    ``window`` exceeds ``k``. The trade fades the forced flow:
      * long-dominated flush  -> longs are force-SOLD, price overshoots DOWN -> +1 (long)
      * short-dominated squeeze -> shorts force-BOUGHT, price overshoots UP -> -1 (short)

    Returns ``(bar_index, side)`` events (positional indices into the series), causal
    by construction (the z is trailing). This is a pre-registered modelling choice —
    size/cluster thresholds (``window``, ``k``) are swept and DSR-deflated in the gate.
    """
    ll = pd.Series(long_liq, dtype=float).reset_index(drop=True)
    sl = pd.Series(short_liq, dtype=float).reset_index(drop=True)
    total = ll + sl
    z = causal_zscore(total, window)
    side = np.where(ll.to_numpy() >= sl.to_numpy(), 1.0, -1.0)
    hits = np.where(z.to_numpy() > k)[0]
    return [(int(i), float(side[i])) for i in hits]


def liquidation_flush_side_array(
    long_liq: pd.Series,
    short_liq: pd.Series,
    *,
    window: int,
    k: float,
    lag: int,
) -> np.ndarray:
    """Per-bar entry SIDE array for the flush gate, with z and dominance CONSISTENTLY
    lagged (the composition the ad-hoc driver glue got wrong).

    Builds flush events via the tested ``liquidation_flush_events`` — whose side is the
    dominance AT THE FLUSH BAR (known when the flush is detected) — then places each
    event's side ``lag`` bars later (the entry). The side is therefore NEVER taken from
    the entry bar's own dominance, which would leak the entry bar's contemporaneous price
    move (dominance ~ same-bar return, corr ~ -0.55) and is unknown at the flush. Returns
    a float array of {-1, 0, +1} aligned to the inputs, for ``basket_event_gate``.
    """
    n = len(long_liq)
    side = np.zeros(n, dtype=float)
    for flush_bar, s in liquidation_flush_events(long_liq, short_liq, window, k):
        entry = flush_bar + lag
        if 0 <= entry < n:
            side[entry] = s
    return side


def oi_acceleration_zscore(oi: pd.Series, window: int) -> pd.Series:
    """Vector 2.3: OI ACCELERATION = causal z-score of open-interest growth.

    ``oi.pct_change()`` is the OI growth rate; its trailing z is the acceleration — a
    rising leveraged build. Causal (trailing window). The cross-venue funding
    DISPERSION half of 2.3 is data-gated (the frozen lake is Binance-only), so the
    driver tests only the single-venue OI-acceleration-at-constant-funding mutation.
    """
    growth = pd.Series(oi, dtype=float).pct_change()
    return causal_zscore(growth, window)


# --------------------------------------------------------------------------- #
# lake I/O seam (thin; the frozen DuckDB lake is the only data source)
# --------------------------------------------------------------------------- #
_KLINE_SYMBOL = {  # coinglass-style USDT pair -> binance_klines slash symbol
    "BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "SOLUSDT": "SOL/USDT",
    "AVAXUSDT": "AVAX/USDT", "ARBUSDT": "ARB/USDT", "SUIUSDT": "SUI/USDT",
    "WIFUSDT": "WIF/USDT",
}
_ALLOWED_VIEWS = {
    "coinglass_top_lsr_account", "coinglass_top_lsr_position",
    "coinglass_global_lsr_account", "coinglass_liquidation",
    "coinglass_funding_rate", "coinglass_open_interest",
    "coinglass_aggregated_open_interest", "coinglass_taker_volume",
    "binance_klines",
}


def load_lake(con, view: str, symbol: str, interval: str) -> pd.DataFrame:
    """Load one symbol/interval slice from the frozen DuckDB lake, sorted by ts (ms).

    ``view`` is validated against an allowlist (no arbitrary SQL); ``symbol`` is the
    coinglass-style pair (``BTCUSDT``) and is auto-mapped to the slash form for
    ``binance_klines``. Returns the raw frame — causal lagging is the caller's job.
    """
    if view not in _ALLOWED_VIEWS:
        raise ValueError(f"view not in allowlist: {view!r}")
    sym = _KLINE_SYMBOL.get(symbol, symbol) if view == "binance_klines" else symbol
    # view is validated against _ALLOWED_VIEWS above; symbol/interval are bound params
    return con.execute(
        f"SELECT * FROM {view} WHERE symbol = ? AND interval = ? ORDER BY ts",
        [sym, interval],
    ).df()


# --------------------------------------------------------------------------- #
# basket gate — de-overlapped joint relocation null, in BAR space
# --------------------------------------------------------------------------- #
def _sharpe(x) -> float:
    a = np.asarray(list(x), dtype=float)
    if a.size < 2:
        return 0.0
    sd = float(a.std(ddof=1))
    return float(a.mean() / sd) if sd > 0.0 else 0.0


def _deoverlap_and_joint_null(
    cand, close_by_sym, *, horizon, cost, n_sims, seed, frac_oos
) -> dict:
    """Shared core: greedy de-overlap of ``cand`` = [(bar, side)] to held positions
    (min-gap = horizon), realise the per-event BASKET return (mean over instruments of
    ``side*(close[t+h]/close[t]-1) - cost``), and a JOINT relocation null (one shared
    random anchor per event applied to every instrument, real side sequence preserved →
    cross-asset correlation intact). Single-sourced so the contrarian and event gates
    share one tested null. ``close_by_sym`` arrays are all on the same aligned grid.
    """
    syms = list(close_by_sym)
    closes = {s: np.asarray(close_by_sym[s], dtype=float) for s in syms}
    n_bars = len(closes[syms[0]])

    kept: list[tuple[int, int]] = []
    last = -(10**9)
    for t, d in sorted(cand):
        if t >= last + horizon and t <= n_bars - 2:
            kept.append((t, d))
            last = t

    def basket_ret(entry: int, direction: int):
        rs = []
        for s in syms:
            c = closes[s]
            h = min(horizon, n_bars - 1 - entry)
            if h <= 0:
                continue
            rs.append(direction * (c[entry + h] / c[entry] - 1.0) - cost)
        return float(np.mean(rs)) if rs else None

    real = [(t, d, basket_ret(t, d)) for t, d in kept]
    real = [(t, d, r) for t, d, r in real if r is not None]
    empty = {"n_events": 0, "sleeve_mean": 0.0, "sleeve_sharpe": 0.0,
             "oos_sharpe": 0.0, "mc_p": 1.0, "per_event": [], "event_bars": []}
    if not real:
        return empty

    bars = np.array([t for t, _, _ in real], dtype=float)
    rets = np.array([r for _, _, r in real], dtype=float)
    real_mean = float(rets.mean())
    lo, hi = bars.min(), bars.max()
    oos = _sharpe(rets[bars >= lo + (1.0 - frac_oos) * (hi - lo)]) if hi > lo else 0.0
    event_bars = [int(t) for t, _, _ in real]

    hi_bound = n_bars - 1 - horizon
    if hi_bound <= 0:
        return {**empty, "n_events": len(real), "sleeve_mean": real_mean,
                "sleeve_sharpe": _sharpe(rets), "oos_sharpe": oos,
                "per_event": [float(x) for x in rets], "event_bars": event_bars}
    rng = np.random.default_rng(seed)
    dirs = [d for _, d, _ in real]
    beats = 0
    for _ in range(n_sims):
        anchors = rng.integers(0, hi_bound, size=len(real))
        nr = [basket_ret(int(a), d) for a, d in zip(anchors, dirs)]
        nr = [x for x in nr if x is not None]
        if nr and float(np.mean(nr)) >= real_mean:
            beats += 1
    return {
        "n_events": len(real),
        "sleeve_mean": real_mean,
        "sleeve_sharpe": _sharpe(rets),
        "oos_sharpe": oos,
        "mc_p": beats / n_sims,
        "per_event": [float(x) for x in rets],
        "event_bars": event_bars,
    }


def basket_event_gate(
    side_by_bar,
    close_by_sym: dict,
    *,
    horizon: int,
    cost: float,
    n_sims: int = 2000,
    seed: int = 0,
    frac_oos: float = 0.4,
) -> dict:
    """Joint-null gate for an explicit per-bar SIDE signal on the common grid.

    ``side_by_bar`` is an array of {-1, 0, +1} (0 = no event) aligned to ``close_by_sym``.
    Used by vector 2.2: the liquidation-flush detector decides the trade side per bar
    (mean-revert the forced flow), and this runs the SAME de-overlapped joint relocation
    null as the contrarian gate.
    """
    side = np.asarray(side_by_bar, dtype=float)
    cand = [(int(t), int(side[t])) for t in range(len(side)) if side[t] != 0 and not np.isnan(side[t])]
    return _deoverlap_and_joint_null(
        cand, close_by_sym, horizon=horizon, cost=cost,
        n_sims=n_sims, seed=seed, frac_oos=frac_oos)


def basket_contrarian_gate(
    z_by_sym: dict,
    close_by_sym: dict,
    *,
    thr: float,
    horizon: int,
    cost: float,
    leg: str = "short",
    n_sims: int = 2000,
    seed: int = 0,
    frac_oos: float = 0.4,
) -> dict:
    """De-overlapped, date-level JOINT-null gate for the positioning basket (BAR space).

    The same statistical construction as ``fxgate.event_basket_gate`` — fade the AVERAGE
    crowding across instruments (one return per event), de-overlap a persistent signal to
    a held position (min-gap = ``horizon``), and relocate the event bars JOINTLY (one
    shared random anchor per event, applied to every instrument) so the null preserves
    cross-asset correlation — but in bar space, so it is correct at 6h as well as 1d
    (``event_basket_gate``'s daily-ms arithmetic is not). Inputs are aligned per-symbol
    arrays; ``z`` is assumed already causal AND publication-lagged (the gate never re-lags).

    ``leg``: ``short`` fades crowded longs (dir -1), ``long`` fades crowded shorts (dir +1),
    ``symmetric`` does both. Returns n_events, sleeve_mean/sharpe, oos_sharpe (last frac by
    bar), mc_p (joint null), and the per-event returns (for DSR deflation upstream).
    """
    syms = list(z_by_sym)
    zmat = np.vstack([np.asarray(z_by_sym[s], dtype=float) for s in syms])
    n_bars = zmat.shape[1]
    common_z = np.nanmean(zmat, axis=0)

    cand: list[tuple[int, int]] = []
    for t in range(n_bars):
        cz = common_z[t]
        if np.isnan(cz):
            continue
        if leg in ("short", "symmetric") and cz > thr:
            cand.append((t, -1))
        if leg in ("long", "symmetric") and cz < -thr:
            cand.append((t, 1))

    return _deoverlap_and_joint_null(
        cand, close_by_sym, horizon=horizon, cost=cost,
        n_sims=n_sims, seed=seed, frac_oos=frac_oos)
