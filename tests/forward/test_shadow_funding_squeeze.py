"""FORWARD — size-0 shadow of the LIVE ``funding_squeeze_continuation`` sleeve.

This is the clean bear/range OOS the frozen lake cannot give: we reproduce the live
rule (|funding-z|>=2 over 180 bars → LONG, ~24x4h hold, no-overlap) on the live feed
in real time and log size-0 theoretical positions + PnL.

The unit under test is faithfulness + causality:

* the entry decision is BYTE-IDENTICAL to the live ``funding_squeeze_continuation``
  signal (we call it, never re-implement it);
* the trade return is BYTE-IDENTICAL to the validated ``harness.evaluate_signal``
  (same triple-barrier, same no-overlap, same costs);
* a position whose 24-bar horizon has not fully elapsed is logged as OPEN (MTM),
  never prematurely closed — so the forward log self-heals when more bars arrive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.forward import shadow_funding_squeeze as sfs
from hermes.research.harness import evaluate_signal
from hermes.research.signals import funding_squeeze_continuation

_4H_MS = 4 * 3_600_000


def _arm_df(n=400, arm_at=(240,), drift_per_bar=0.006) -> pd.DataFrame:
    """Flat tape with a planted funding spike at ``arm_at`` (arms |z|>=2 only there)
    and a positive continuation drift after the first spike so the LONG wins.

    Baseline funding is FLAT (0.0): a constant series has zero std, so ``_ztail``
    returns 0.0 — the signal arms only on the planted spike bars, never on noise.
    """
    ts = (np.arange(n) * _4H_MS + 1_600_000_000_000).astype("int64")
    funding = np.zeros(n)  # flat baseline → z=0 except on planted spikes
    for a in arm_at:
        funding[a] = 0.01  # ~big positive settlement → trailing-180 z >> 2
    close = np.full(n, 100.0)
    first = min(arm_at)
    for i in range(first + 1, n):
        close[i] = close[i - 1] * (1.0 + drift_per_bar)
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 10.0),
            "funding_rate": funding,
        }
    )
    return df


def test_entry_decision_is_byte_identical_to_live_signal():
    df = _arm_df(arm_at=(240,))
    trades = sfs.funding_squeeze_trades(df, "BTC/USDT")
    assert len(trades) >= 1
    tr = trades.iloc[0]
    # the live signal evaluated on the same trailing window must also fire BUY
    live = funding_squeeze_continuation(df.iloc[: int(tr["entry_idx"]) + 1])
    assert live.side == "BUY"
    assert tr["side"] == "BUY"
    assert float(tr["funding_z"]) == float(live.features["funding_z"])
    assert float(tr["funding_z"]) >= 2.0


def test_trade_return_byte_identical_to_validated_harness():
    df = _arm_df(arm_at=(240,))
    mine = sfs.funding_squeeze_trades(df, "BTC/USDT")
    closed = mine[mine["closed"]]
    ledger = evaluate_signal(
        {"BTC/USDT": df},
        funding_squeeze_continuation,
        fee_bps=sfs.FEE_BPS,
        slippage_bps=sfs.SLIPPAGE_BPS,
        horizon=sfs.HOLD_BARS,
        warmup=sfs.WARMUP,
        window_cap=sfs.WINDOW_CAP,
    )
    mine_pairs = sorted((int(r["entry_ts"]), round(float(r["ret_net"]), 12)) for _, r in closed.iterrows())
    theirs = sorted((ev, round(r, 12)) for ev, r in ledger)
    assert mine_pairs == theirs


def test_no_overlap_blocks_second_entry_within_hold():
    # two arms 10 bars apart (< 24-bar hold) → only the first books a position
    df = _arm_df(arm_at=(240, 250))
    trades = sfs.funding_squeeze_trades(df, "BTC/USDT")
    assert len(trades) == 1
    assert int(trades.iloc[0]["entry_idx"]) == 240


def test_no_arm_yields_no_trades():
    n = 400
    # funding oscillates but its z-score never reaches 2 (a pure sine peaks at z≈1.41)
    df = pd.DataFrame(
        {
            "ts": (np.arange(n) * _4H_MS + 1_600_000_000_000).astype("int64"),
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 10.0,
            "funding_rate": 1e-4 * np.sin(np.arange(n)),  # |z| <= ~1.41, never arms
        }
    )
    trades = sfs.funding_squeeze_trades(df, "BTC/USDT")
    assert trades.empty


def test_open_position_not_prematurely_closed():
    # arm 5 bars before the end: horizon (24) cannot elapse → must log OPEN, MTM only
    n = 400
    df = _arm_df(n=n, arm_at=(n - 6,))
    trades = sfs.funding_squeeze_trades(df, "BTC/USDT")
    assert len(trades) == 1
    tr = trades.iloc[0]
    assert bool(tr["closed"]) is False
    assert tr["exit_reason"] == "open"
    # MTM return uses the last available bar, not a fabricated future bar
    last_close = float(df["close"].iloc[-1])
    entry_px = float(tr["entry_px"])
    assert np.isclose(float(tr["ret_gross"]), (last_close - entry_px) / entry_px)


def test_is_deterministic():
    df = _arm_df(arm_at=(240,))
    a = sfs.funding_squeeze_trades(df, "BTC/USDT")
    b = sfs.funding_squeeze_trades(df, "BTC/USDT")
    pd.testing.assert_frame_equal(a, b)
