"""Stream 5 — combined FX/metals relative-value book: construction core.

The prior arc (``research_fx/HONEST_NEGATIVE_REPORT.md``) proved that every *directional*
free-data gold/FX signal is one of two things: noise (fails the Monte-Carlo random-entry
null) or secular-bull-beta (random long-biased entries already earn gold's drift, so nothing
beats the null). This module exists to test the one structurally-distinct hypothesis that the
directional arc could not reach: **relative-value, market-neutral sleeves where the bull-beta is
removed by construction** (one leg hedged against another), judged not by solo Sharpe but by
their contribution to a combined BOOK (``hermes.validation.book``).

What lives here is the *pure construction core* — no data-lake I/O is required to test it:

* ``causal_rolling_beta`` / ``neutral_spread_returns`` — build a beta-neutral spread whose
  exposure to the common metal factor is ~0 *by construction* (the whole point of S5);
* ``synthetic_nav`` — turn a spread return stream into a price path so the existing
  ``research_fx/fxgate.py`` §3 gate (no-overlap realise, honest 2-leg cost, MC random-entry
  null, OOS-by-date) can be reused verbatim on the spread;
* ``causal_zscore`` / ``threshold_events`` — slow, causal signal generation (the ledger killed
  the FAST gold-silver reversion at ~10d half-life, so every signal here is deliberately slow);
* ``factor_neutrality`` — the neutrality *proof*: measured beta and rho of a sleeve to a factor;
* ``load_lake_close`` — a thin, dependency-injected DuckDB reader for the frozen ``prices`` view.

Causality is the load-bearing invariant: a hedge ratio or z-score applied on day *t* uses only
returns strictly before *t* (rolling-through-(t-1) then ``shift(1)``), so the spread return
series is one a trader could actually have earned — no in-sample weighting leak that would
inflate the very Sharpe the book gate then deflates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "causal_multi_beta",
    "causal_rolling_beta",
    "causal_zscore",
    "factor_neutrality",
    "load_lake_close",
    "log_returns",
    "neutral_spread_returns",
    "neutral_spread_returns_multi",
    "pair_usd_return",
    "pair_usd_value",
    "synthetic_nav",
    "threshold_events",
]


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns of a close series; first element is NaN (no prior bar)."""
    return np.log(close.astype(float)).diff()


def causal_rolling_beta(
    r_a: pd.Series, r_b: pd.Series, lookback: int
) -> pd.Series:
    """Trailing OLS slope of ``r_a`` on ``r_b`` (Cov/Var), known at the open of each bar.

    The beta at bar *t* is estimated over the ``lookback`` window ending at *t-1*
    (rolling-through-*t* then ``.shift(1)``), so it uses only returns strictly before *t*
    — no look-ahead. Warmup bars (before a full window has elapsed and been shifted) are
    NaN. A zero-variance window yields NaN (no defined hedge).
    """
    a = r_a.astype(float).reset_index(drop=True)
    b = r_b.astype(float).reset_index(drop=True)
    cov = a.rolling(lookback).cov(b)
    var = b.rolling(lookback).var()
    beta = (cov / var).shift(1)
    beta.index = r_a.index
    return beta


def neutral_spread_returns(
    r_a: pd.Series, r_b: pd.Series, lookback: int
) -> pd.Series:
    """Beta-neutral spread return: ``r_a - beta_t * r_b`` with a CAUSAL hedge ``beta_t``.

    This is the regression residual of leg A on leg B — long A, short ``beta_t`` units of B —
    so its exposure to B (the common metal factor when B is gold) is ~0 by construction. The
    secular-bull-beta that sank the directional arc is hedged away, not merely hoped away.
    """
    beta = causal_rolling_beta(r_a, r_b, lookback)
    return r_a.astype(float) - beta * r_b.astype(float)


def causal_multi_beta(
    r_a: pd.Series, factors: pd.DataFrame, lookback: int
) -> pd.DataFrame:
    """Trailing multivariate OLS slopes of ``r_a`` on every column of ``factors``, causal.

    For each bar the betas are the no-intercept least-squares fit over the ``lookback`` window
    ending at *t-1* (window ending at *t* then ``.shift(1)``), so they are knowable at *t*'s
    open. Windows with any NaN or a rank-deficient design yield NaN betas. Returns a frame with
    the same columns as ``factors``. (No intercept: return streams are ~zero-mean, and an
    intercept would absorb part of the hedge.)
    """
    cols = list(factors.columns)
    X = factors.to_numpy(dtype=float)
    y = r_a.to_numpy(dtype=float)
    n, k = X.shape
    raw = np.full((n, k), np.nan)
    for t in range(lookback - 1, n):
        xs = X[t - lookback + 1 : t + 1]
        ys = y[t - lookback + 1 : t + 1]
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            continue
        if np.linalg.matrix_rank(xs) < k:
            continue
        beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        raw[t] = beta
    out = pd.DataFrame(raw, index=r_a.index, columns=cols).shift(1)
    return out


def neutral_spread_returns_multi(
    r_a: pd.Series, factors: pd.DataFrame, lookback: int
) -> pd.Series:
    """Residual of ``r_a`` after a CAUSAL multivariate hedge against every column of ``factors``.

    ``r_a - Σ_k beta_{k,t} · factor_{k,t}`` — the idiosyncratic stream left once *all* the
    supplied factors (e.g. BOTH gold and silver) are hedged out, the strongest orthogonality
    a metals-complex sleeve can claim.
    """
    betas = causal_multi_beta(r_a, factors, lookback)
    hedged = (betas.to_numpy() * factors.to_numpy()).sum(axis=1)
    return r_a.astype(float) - pd.Series(hedged, index=r_a.index)


def synthetic_nav(returns: pd.Series) -> pd.Series:
    """Compound a (possibly NaN-leading) return stream into a price path starting at ~1.0.

    A leading NaN return (the first bar has no prior) is treated as FLAT (0.0). The absolute
    base is irrelevant — ``fxgate`` only ever takes ratios ``close[i+h]/close[i]`` — but a
    clean monotone-in-PnL path lets the §3 gate realise the spread as if it were one tradable
    instrument paying the honest 2-leg cost.
    """
    return (1.0 + returns.fillna(0.0)).cumprod()


def causal_zscore(x: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score known at the open of each bar (mean/std over the window ending at *t-1*).

    Like ``causal_rolling_beta`` it rolls through *t* then ``.shift(1)`` so the score driving a
    decision at *t* never peeks at *t*'s own value. Zero-variance windows yield NaN.
    """
    roll = x.astype(float).rolling(lookback)
    z = (x.astype(float) - roll.mean()) / roll.std(ddof=1)
    return z.shift(1)


def threshold_events(x: pd.Series, entry: float, mode: str) -> list[tuple[int, int]]:
    """Generate ``(bar_index, side)`` events wherever ``|x| >= entry``.

    ``mode="trend"`` rides the sign (side = sign(x)); ``mode="revert"`` fades it
    (side = -sign(x)). Events are emitted on every qualifying bar — ``fxgate`` collapses a
    run into one non-overlapping held position, so persistence never double-counts. The bar
    index is positional (0-based), matching ``fxgate``'s expectation.
    """
    if mode not in ("trend", "revert"):
        raise ValueError(f"unknown mode: {mode!r} (use 'trend' or 'revert')")
    sign_mult = 1 if mode == "trend" else -1
    vals = x.to_numpy(dtype=float)
    events: list[tuple[int, int]] = []
    for i, v in enumerate(vals):
        if np.isfinite(v) and abs(v) >= entry:
            events.append((i, sign_mult * (1 if v > 0 else -1)))
    return events


def factor_neutrality(spread_returns: pd.Series, factor_returns: pd.Series) -> dict:
    """Measured exposure of a sleeve to a factor — the neutrality *proof*, not an assumption.

    Aligns the two streams (inner, dropping NaN), returns the OLS ``beta`` (slope of the sleeve
    on the factor), Pearson ``rho``, and the overlap count ``n``. A genuinely neutral sleeve
    shows ``|beta|`` and ``|rho|`` ~0 against the bull factor (gold).
    """
    df = pd.concat([spread_returns, factor_returns], axis=1, keys=["s", "f"]).dropna()
    if len(df) < 2:
        return {"beta": 0.0, "rho": 0.0, "n": len(df)}
    s, f = df["s"].to_numpy(), df["f"].to_numpy()
    var_f = float(np.var(f, ddof=1))
    beta = float(np.cov(s, f, ddof=1)[0, 1] / var_f) if var_f > 0 else 0.0
    sd_s, sd_f = float(np.std(s, ddof=1)), float(np.std(f, ddof=1))
    rho = float(np.corrcoef(s, f)[0, 1]) if sd_s > 0 and sd_f > 0 else 0.0
    return {"beta": beta, "rho": rho, "n": len(df)}


def pair_usd_value(close: pd.Series, quote: str) -> pd.Series:
    """USD value of the FOREIGN currency from a spot FX pair, normalising the quote convention.

    ``quote="XXXUSD"`` (e.g. AUDUSD, EURUSD — price already *is* USD-per-foreign) → the price.
    ``quote="USDXXX"`` (e.g. USDCHF, USDZAR — price is foreign-per-USD) → ``1/price``.
    This lets a cross-rate (X/Y) be built as ``pair_usd_value(X)/pair_usd_value(Y)`` with the USD
    leg cancelling — i.e. dollar-neutral by construction, the prerequisite for FX cross-rate RV.
    """
    if quote == "XXXUSD":
        return close.astype(float)
    if quote == "USDXXX":
        return 1.0 / close.astype(float)
    raise ValueError(f"unknown quote {quote!r} (use 'XXXUSD' or 'USDXXX')")


def pair_usd_return(close: pd.Series, quote: str) -> pd.Series:
    """Log return of the foreign currency's USD value (see :func:`pair_usd_value`).

    For a ``USDXXX`` pair this is the NEGATIVE of the pair's own log return — the sign that makes
    a long-X/short-Y spread genuinely dollar-neutral rather than secretly doubling the USD bet.
    """
    return log_returns(pair_usd_value(close, quote))


def load_lake_close(con, symbols: list[str], interval: str = "1d") -> pd.DataFrame:
    """Wide close-price frame (one column per symbol) from the frozen ``prices`` view.

    ``con`` is an injected DuckDB connection (so the suite tests it against an in-memory
    fixture, never the on-disk lake). Rows are inner-joined on ``ts`` across symbols — only
    bars present for *every* requested symbol survive, which is what a multi-leg spread needs.
    """
    placeholders = ", ".join("?" for _ in symbols)
    q = (
        f"SELECT ts, symbol, close FROM prices "
        f"WHERE interval = ? AND symbol IN ({placeholders}) ORDER BY ts"
    )
    raw = con.execute(q, [interval, *symbols]).df()
    wide = raw.pivot(index="ts", columns="symbol", values="close")
    wide = wide[symbols].dropna(how="any")
    wide.columns.name = None
    return wide
