"""FORWARD — size-0 shadow of the PRE-REGISTERED cross-sectional funding-reversion.

The cell (frozen, sign-inverted = FADE high funding-z; see ``forward.prereg``):
  rank the 23-perp universe by trailing-180-settlement funding-z every 14 days;
  LONG the bottom tercile, SHORT the top tercile, dollar-neutral. Inverted because
  the S2 v-next finding is that cross-sectional funding MEAN-REVERTS (the squeeze
  direction lost; only the slow inverted spread carried a +0.58 post-hoc lead).

The unit under test is the construction, not the (unknowable) edge:
  * DOLLAR-NEUTRAL: a uniform market move cancels (this is what removes the
    bull-beta that contaminates every directional backtest — the raison d'être);
  * the inverted sign fades high funding-z (shorts the top tercile);
  * causality: the period return uses only funding-z known at rebalance and prices
    over [rebalance, rebalance+14d] — nothing after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.forward import shadow_xs_reversion as xsr
from hermes.research.signals import _ztail

_DAY_MS = 86_400_000


def _panel(z_by_sym, ret_by_sym, *, period_days=14, days=15):
    """Daily panel: each symbol flat at 100 until the period close, then *(1+ret).
    funding_z is constant per symbol (the cross-section we want to rank)."""
    ts = (np.arange(days) * _DAY_MS + 1_700_000_000_000).astype("int64")
    panel = {}
    for sym, z in z_by_sym.items():
        close = np.full(days, 100.0)
        close[period_days:] = 100.0 * (1.0 + ret_by_sym[sym])
        panel[sym] = pd.DataFrame(
            {"ts": ts, "close": close, "funding_z": np.full(days, float(z))}
        )
    return panel


def test_causal_funding_z_matches_live_ztail_and_warms_up():
    rng = np.random.default_rng(0)
    f = pd.Series(rng.normal(0, 1e-4, 300))
    z = xsr.causal_funding_z(f, window=180)
    assert z.iloc[:179].isna().all()  # first window-1 are NaN (no premature signal)
    assert np.isclose(float(z.iloc[-1]), _ztail(f, 180))  # same convention as live


def test_causal_funding_z_is_causal():
    f = pd.Series(np.r_[np.zeros(200), np.ones(20)])
    z = xsr.causal_funding_z(f, window=180)
    z_at_190 = float(z.iloc[190])
    f2 = f.copy()
    f2.iloc[210:] = 999.0  # mutate the future
    z2 = xsr.causal_funding_z(f2, window=180)
    assert float(z2.iloc[190]) == z_at_190  # past z unchanged by future data


def test_terciles_invert_shorts_high_funding_z():
    z = pd.Series({f"S{i}": v for i, v in enumerate([-2, -1.5, -1, -0.3, 0, 0.3, 1, 1.5, 2])})
    w = xsr.assign_terciles(z, invert=True, n_terciles=3)
    assert np.isclose(w.sum(), 0.0)  # dollar-neutral
    # highest funding-z (S6,S7,S8) are SHORT (negative); lowest (S0,S1,S2) are LONG
    assert (w[["S6", "S7", "S8"]] < 0).all()
    assert (w[["S0", "S1", "S2"]] > 0).all()
    assert np.isclose(w[["S3", "S4", "S5"]].abs().sum(), 0.0)  # middle flat
    # non-inverted is the exact opposite sign
    w2 = xsr.assign_terciles(z, invert=False, n_terciles=3)
    assert np.allclose(w2.to_numpy(), -w.to_numpy())


def test_uniform_market_move_cancels_dollar_neutral():
    syms = [f"S{i}" for i in range(9)]
    z = {s: i - 4 for i, s in enumerate(syms)}  # distinct z
    ret = {s: 0.05 for s in syms}  # EVERYONE +5% (pure beta)
    book = xsr.xs_reversion_book(_panel(z, ret), period_days=14, fee_bps=0.0, slippage_bps=0.0)
    closed = book[book["closed"]]
    assert len(closed) == 1
    assert np.isclose(float(closed["ret_gross"].iloc[0]), 0.0, atol=1e-12)  # beta removed


def test_inverted_sign_profits_when_high_funding_mean_reverts():
    syms = [f"S{i}" for i in range(9)]
    z = {s: i - 4 for i, s in enumerate(syms)}
    # high-z names fall, low-z names rise (mean reversion) → fading high-z wins
    ret = {s: (-0.04 if z[s] > 0.5 else (0.04 if z[s] < -0.5 else 0.0)) for s in syms}
    book = xsr.xs_reversion_book(_panel(z, ret), period_days=14, fee_bps=0.0, slippage_bps=0.0)
    g = float(book[book["closed"]]["ret_gross"].iloc[0])
    assert g > 0.0  # long bottom (+0.04) minus short top (-0.04) = +0.08


def test_two_leg_cost_reduces_net_below_gross():
    syms = [f"S{i}" for i in range(9)]
    z = {s: i - 4 for i, s in enumerate(syms)}
    ret = {s: (-0.04 if z[s] > 0.5 else (0.04 if z[s] < -0.5 else 0.0)) for s in syms}
    book = xsr.xs_reversion_book(_panel(z, ret), period_days=14, fee_bps=7.0, slippage_bps=3.0)
    row = book[book["closed"]].iloc[0]
    assert float(row["ret_net"]) < float(row["ret_gross"])
    assert float(row["cost"]) > 0.0


def test_period_return_ignores_data_after_exit():
    syms = [f"S{i}" for i in range(9)]
    z = {s: i - 4 for i, s in enumerate(syms)}
    ret = {s: (-0.04 if z[s] > 0.5 else (0.04 if z[s] < -0.5 else 0.0)) for s in syms}
    p1 = _panel(z, ret, period_days=14, days=15)
    # p2: identical through the exit day (14), then WILD bars afterwards
    ts2 = (np.arange(20) * _DAY_MS + 1_700_000_000_000).astype("int64")
    p2 = {}
    for s in syms:
        close = np.full(20, 100.0)
        close[14] = 100.0 * (1.0 + ret[s])  # the exit-day price (must drive period 0)
        close[15:] = 9999.0  # wild, AFTER the exit — must be ignored
        p2[s] = pd.DataFrame({"ts": ts2, "close": close, "funding_z": np.full(20, float(z[s]))})
    g1 = float(xsr.xs_reversion_book(p1, fee_bps=0.0, slippage_bps=0.0)["ret_gross"].iloc[0])
    book2 = xsr.xs_reversion_book(p2, fee_bps=0.0, slippage_bps=0.0)
    r0 = int(p2[syms[0]]["ts"].iloc[0])
    g2 = float(book2[book2["rebalance_ts"] == r0]["ret_gross"].iloc[0])
    assert np.isclose(g1, g2)


def test_is_deterministic():
    syms = [f"S{i}" for i in range(9)]
    z = {s: i - 4 for i, s in enumerate(syms)}
    ret = {s: 0.01 * (i - 4) for i, s in enumerate(syms)}
    a = xsr.xs_reversion_book(_panel(z, ret))
    b = xsr.xs_reversion_book(_panel(z, ret))
    pd.testing.assert_frame_equal(a, b)
