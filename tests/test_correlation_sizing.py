"""Correlation-aware position sizing: a new position in a correlated book is
haircut by 1/sqrt(1 + n_existing*rho) so the AGGREGATE book risk stays ~constant
instead of stacking (BTC/ETH/SOL move together — 3 longs ≈ 3x BTC-beta)."""

import math

import pytest

from hermes.risk.sizing import PositionSizer, SizingInputs, correlation_haircut


def _inp(**over) -> SizingInputs:
    base = dict(
        equity_usd=100.0, confidence=0.8, win_rate=None, avg_win_loss_ratio=None,
        atr=1.0, price=100.0, trades_in_journal=0, stage_cap_pct=2.0, floor_pct=0.25,
    )
    base.update(over)
    return SizingInputs(**base)


def test_haircut_is_one_for_first_or_uncorrelated_position():
    assert correlation_haircut(0, 0.85) == 1.0   # first position
    assert correlation_haircut(1, 0.0) == 1.0     # uncorrelated book


def test_haircut_shrinks_with_correlated_open_positions():
    assert correlation_haircut(1, 0.85) == pytest.approx(1 / math.sqrt(1.85), rel=1e-9)
    assert correlation_haircut(2, 0.85) == pytest.approx(1 / math.sqrt(2.70), rel=1e-9)
    # monotonically decreasing, always in (0, 1]
    prev = 1.0
    for n in range(0, 6):
        h = correlation_haircut(n, 0.9)
        assert 0.0 < h <= 1.0 and h <= prev
        prev = h


def test_sizer_scales_risk_and_notional_by_haircut():
    full = PositionSizer().size(_inp())
    half = PositionSizer().size(_inp(correlation_haircut=0.5))
    assert half.risk_pct == pytest.approx(full.risk_pct * 0.5, rel=1e-9)
    assert half.notional_usd == pytest.approx(full.notional_usd * 0.5, rel=1e-9)
