"""TDD — Boucle-alpha carry-harvest backtester (built ON carry_join, mirrors S3 paper).

The harvest is a FUNDING-SETTLEMENT ACCUMULATOR, not a price-return engine. Two
correctness guards dominate and are TEST #1 / #2:

  #1  L0 leak: the trailing-funding-APR builder is causal (backward window). A forward
      window MUST be caught by carry_join.assert_no_lookahead.
  #2  NO DOUBLE-COUNT: funding attributed onto the 4h grid sums EXACTLY to the native
      settlements (binance_funding_on_4h forward-CARRIES the 8h rate across a variable
      number of 4h bars, so per-bar summation would 2x; native settlements are the
      ground truth -> sum(attributed) == sum(native), each settlement counted once).

Then the on/off mechanics: causal cadence-gated arming (anti-whipsaw), costs charged
once per open + once per close (never per bar), equal-weight-among-armed book.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hermes.datalake import carry_join as cj
from hermes.research import carry_harvest as ch

FOUR_H_MS = 14_400_000
EIGHT_H_MS = 2 * FOUR_H_MS


# =========================================================================== #
# TEST #1 — L0 leak guard on the trailing-funding-APR builder
# =========================================================================== #
def _funding_frame(rates):
    return pd.DataFrame({"ts": [i * FOUR_H_MS for i in range(len(rates))], "funding": rates})


def test_leak_guard_passes_causal_trailing_funding_apr():
    """Backward rolling window: perturbing a FUTURE row leaves earlier trail_apr unchanged."""
    frame = _funding_frame([0.0001] * 8)

    def builder(df: pd.DataFrame) -> pd.DataFrame:
        return ch.add_trailing_funding_apr(df, lookback_bars=3)

    cj.assert_no_lookahead(builder, frame, value_cols=["trail_apr"])


def test_leak_guard_catches_a_forward_window_trailing_apr():
    """A builder that looks FORWARD (future funding in the window) MUST be flagged."""
    frame = _funding_frame([0.0001] * 8)

    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # forward-looking: includes data[t+1] -> leak
        out["trail_apr"] = out["funding"][::-1].rolling(3, min_periods=1).sum()[::-1]
        return out

    with pytest.raises(cj.LookaheadError):
        cj.assert_no_lookahead(leaky, frame, value_cols=["trail_apr"])


# =========================================================================== #
# TEST #2 — NO DOUBLE-COUNT: attributed funding == native settlements, exactly once
# =========================================================================== #
def test_attribute_funding_sums_exactly_to_native_settlements():
    """8h settlements onto a 4h grid: total attributed == total settled, no 2x, no drop."""
    grid_ts = np.array([i * FOUR_H_MS for i in range(8)])         # 8 4h bars
    native_ts = np.array([0, EIGHT_H_MS, 2 * EIGHT_H_MS, 3 * EIGHT_H_MS])  # 4 settlements
    native_rate = np.array([0.0001, 0.0002, -0.0003, 0.0004])
    out = ch.attribute_funding(grid_ts, native_ts, native_rate)
    assert out.shape == grid_ts.shape
    assert abs(out.sum() - native_rate.sum()) < 1e-15          # lossless, no double-count


def test_attribute_funding_puts_each_settlement_in_exactly_one_bar():
    """Non-settlement bars get 0; a settlement lands in the bar that contains it (causal)."""
    grid_ts = np.array([i * FOUR_H_MS for i in range(4)])      # bars at 0,4h,8h,12h
    native_ts = np.array([EIGHT_H_MS])                          # one settlement at 8h
    native_rate = np.array([0.0005])
    out = ch.attribute_funding(grid_ts, native_ts, native_rate)
    assert out.tolist() == [0.0, 0.0, 0.0005, 0.0]             # only the 8h bar


def test_attribute_funding_buckets_intra_bar_settlement_to_its_bar():
    """A settlement strictly inside a bar interval [t, t+4h) attributes to that bar."""
    grid_ts = np.array([0, FOUR_H_MS, 2 * FOUR_H_MS])
    native_ts = np.array([FOUR_H_MS + 1_000])                  # just after the 4h bar opens
    native_rate = np.array([0.0009])
    out = ch.attribute_funding(grid_ts, native_ts, native_rate)
    assert out.tolist() == [0.0, 0.0009, 0.0]


# =========================================================================== #
# trailing funding APR — causal, correctly annualized
# =========================================================================== #
def test_trailing_funding_apr_annualizes_over_the_window():
    """0.0001 per 4h bar over the window -> 0.0001 * 2190 = 21.9% APR (steady state)."""
    frame = _funding_frame([0.0001] * 10)
    out = ch.add_trailing_funding_apr(frame, lookback_bars=5)
    # once the window is full, trailing APR is steady
    assert abs(out["trail_apr"].iloc[-1] - 0.0001 * ch.BARS_PER_YEAR_4H) < 1e-9


# =========================================================================== #
# arm schedule — cadence-gated, causal (anti-whipsaw)
# =========================================================================== #
def test_arm_schedule_only_redecides_at_cadence_boundaries():
    """Exposure changes only at cadence boundaries; the decision governs the NEXT block
    (decision at boundary uses info through that bar, applied from the following bar)."""
    # trailing apr jumps above tau at index 2 and stays high
    trail = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    armed = ch.arm_schedule(trail, tau=0.5, cadence_bars=3, warmup_bars=0)
    # boundaries at 0,3,6. decision at a boundary applies from the NEXT bar.
    # bar0: initial flat. decide@0: trail[0]=0<tau -> stays flat through bar3.
    # decide@3: trail[3]=1>=tau -> armed from bar4. decide@6 keeps it.
    assert armed.tolist() == [False, False, False, False, True, True, True, True]


def test_arm_schedule_decision_is_causal_uses_only_past_boundary():
    """A spike in trailing apr that appears AFTER a boundary must not retro-arm earlier bars."""
    trail = np.zeros(12)
    trail[7:] = 5.0                                   # spike well after boundary 6
    armed = ch.arm_schedule(trail, tau=1.0, cadence_bars=6, warmup_bars=0)
    # boundaries 0,6. decide@6 sees trail[6]=0 -> flat until next boundary (none) -> all flat
    assert not armed.any()


# =========================================================================== #
# gated net returns — cost once per open + once per close, NEVER per bar
# =========================================================================== #
def test_gated_returns_charge_round_trip_cost_once_per_episode():
    basis = np.zeros(8)
    funding = np.full(8, 0.001)
    armed = np.array([False, True, True, True, False, False, False, False])
    rtc = 0.002
    out = ch.gated_net_returns(basis, funding, armed, round_trip_cost=rtc)
    # while armed (bars 1,2,3): each earns funding 0.001; open cost at bar1, close at bar4
    total_cost = (funding[armed].sum()) - out.sum()
    assert abs(total_cost - rtc) < 1e-12              # exactly ONE round trip, not per-bar
    assert out[5] == 0.0 and out[6] == 0.0            # flat bars earn nothing


def test_gated_returns_force_close_cost_if_still_armed_at_end():
    basis = np.zeros(4)
    funding = np.zeros(4)
    armed = np.array([False, True, True, True])       # never closes within the window
    out = ch.gated_net_returns(basis, funding, armed, round_trip_cost=0.002)
    # one open (bar1) + a forced close at the final bar -> total cost == one round trip
    assert abs(out.sum() - (-0.002)) < 1e-12


def test_gated_returns_flat_when_never_armed_costs_nothing():
    out = ch.gated_net_returns(np.zeros(5), np.full(5, 0.001), np.zeros(5, bool), 0.002)
    assert out.tolist() == [0.0] * 5


# =========================================================================== #
# book aggregation — equal-weight among armed coins, flat (0) when none armed
# =========================================================================== #
def test_book_returns_equal_weight_among_armed_coins():
    net = {"BTC": np.array([0.01, 0.02, 0.0]), "ETH": np.array([0.03, 0.0, 0.0])}
    armed = {"BTC": np.array([True, True, False]), "ETH": np.array([True, False, False])}
    book = ch.book_returns(net, armed)
    assert abs(book[0] - (0.01 + 0.03) / 2) < 1e-12   # both armed -> mean
    assert abs(book[1] - 0.02) < 1e-12                # only BTC armed -> its return
    assert book[2] == 0.0                             # none armed -> flat


# =========================================================================== #
# regime decomposition — Axis B core: net conditional on funding sign
# =========================================================================== #
def test_regime_split_separates_positive_and_negative_funding_bars():
    book_ret = np.array([0.002, -0.001, 0.003, -0.004])
    funding_sign = np.array([1.0, -1.0, 1.0, -1.0])   # +,-,+,-
    split = ch.regime_split(book_ret, funding_sign)
    assert abs(split["mean_pos"] - 0.0025) < 1e-12    # mean of bars 0,2
    assert abs(split["mean_neg"] - (-0.0025)) < 1e-12 # mean of bars 1,3
    assert abs(split["frac_pos_funding"] - 0.5) < 1e-12


# =========================================================================== #
# LAKE INTEGRATION — verify the accumulator reproduces independently-measured gross
# (skip if the frozen lake is absent; gitignored/point-in-time)
# =========================================================================== #
_DUCKDB = Path(__file__).resolve().parents[2] / "data" / "hermes_datalake.duckdb"

lake = pytest.mark.skipif(
    not _DUCKDB.exists(),
    reason="lake not materialized (gitignored); run `python -m hermes.datalake.run`",
)


@pytest.fixture(scope="module")
def con():
    import duckdb

    c = duckdb.connect(str(_DUCKDB), read_only=True)
    yield c
    c.close()


@lake
def test_native_funding_accumulates_to_measured_gross_BTC_2024(con):
    """BTC 2024 gross funding (sum of native settled rates) was measured at +11.96%
    by an independent query. The accumulator over the SAME settlements must reproduce it
    (this is the 'verify against reality' anchor — no double-count, no drop)."""
    df = con.execute(
        """
        SELECT ts, funding_rate FROM binance_funding_native
        WHERE symbol = 'BTC/USDT'
          AND EXTRACT(year FROM epoch_ms(ts)) = 2024
        ORDER BY ts
        """
    ).df()
    native_ts = df["ts"].to_numpy()
    native_rate = df["funding_rate"].to_numpy()
    # a 4h grid spanning 2024 that contains every settlement
    grid_ts = np.arange(native_ts.min() - FOUR_H_MS, native_ts.max() + FOUR_H_MS, FOUR_H_MS)
    attributed = ch.attribute_funding(grid_ts, native_ts, native_rate)
    assert abs(attributed.sum() - native_rate.sum()) < 1e-9       # lossless
    assert abs(100.0 * attributed.sum() - 11.96) < 0.05           # matches measured gross


@lake
def test_load_cash_and_carry_basis_is_tiny_vs_funding_BTC(con):
    """The delta-neutral basis term (spot_ret - perp_ret) is near-zero on average
    (corr ~0.999); funding dominates. Confirms the position is genuinely delta-neutral."""
    aligned = cj.carry_returns(cj.cash_and_carry_from_lake(con, symbol="BTC/USDT"))
    basis = (aligned["spot_ret"] - aligned["perp_ret"]).dropna()
    # mean |basis| per 4h bar is tiny (< 10 bps); it is a tail risk, not a mean driver
    assert abs(basis.mean()) < 0.001


# =========================================================================== #
# ORCHESTRATION — per-coin streams + book assembly across the variable universe
# =========================================================================== #
def test_build_book_union_grid_and_pre_listing_flat():
    """Coins list at different times; the book runs on the UNION 4h grid and a coin is
    flat (unarmed, zero net) on bars before it lists."""
    cfg = ch.CarryConfig(tau_apr=0.0, cadence_bars=1, lookback_bars=1, round_trip_cost=0.0)
    btc = pd.DataFrame({"ts": [i * FOUR_H_MS for i in range(4)],
                        "basis_ret": [0.0] * 4, "funding": [0.001] * 4})
    eth = pd.DataFrame({"ts": [2 * FOUR_H_MS, 3 * FOUR_H_MS],
                        "basis_ret": [0.0] * 2, "funding": [0.001] * 2})
    out = ch.build_book({"BTC": btc, "ETH": eth}, cfg)
    assert out["ts"].tolist() == [i * FOUR_H_MS for i in range(4)]
    assert not out["per_coin_armed"]["ETH"][0] and not out["per_coin_armed"]["ETH"][1]
    assert len(out["book_ret"]) == 4
    assert np.isfinite(out["book_ret"]).all()


def test_build_book_costs_reduce_net():
    """Same arming, higher round-trip cost -> strictly lower total book return."""
    btc = pd.DataFrame({"ts": [i * FOUR_H_MS for i in range(20)],
                        "basis_ret": [0.0] * 20, "funding": [0.001] * 20})
    free = ch.CarryConfig(0.0, 5, 5, 0.0)
    costly = ch.CarryConfig(0.0, 5, 5, 0.004)
    b_free = ch.build_book({"BTC": btc}, free)["book_ret"].sum()
    b_cost = ch.build_book({"BTC": btc}, costly)["book_ret"].sum()
    assert b_cost < b_free


@lake
def test_cash_and_carry_stream_is_delta_neutral_with_positive_funding(con):
    """The per-coin stream exposes [ts, basis_ret, funding]; basis is delta-neutral
    (mean ~0) and BTC's lifetime funding is net positive (the harvestable carry)."""
    s = ch.cash_and_carry_stream(con, "BTC/USDT")
    assert {"ts", "basis_ret", "funding"}.issubset(s.columns)
    assert abs(s["basis_ret"].mean()) < 0.001
    assert s["funding"].sum() > 0
