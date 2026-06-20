"""HAR-RV volatility forecast + vol-targeting scale.

Vol-targeting modulates the RISK BUDGET (risk_pct) by how far forecast vol is
from a target: size DOWN in elevated-vol regimes, UP (bounded) when calm. This
is distinct from the ATR-normalized per-trade stop (which keeps $-risk constant
per trade); here we scale the budget itself.
"""

import pytest

from hermes.risk.sizing import PositionSizer, SizingInputs
from hermes.risk.volforecast import har_rv_forecast, realized_vol, vol_target_scale


def _inp(**over) -> SizingInputs:
    base = dict(
        equity_usd=100.0, confidence=0.8, win_rate=None, avg_win_loss_ratio=None,
        atr=1.0, price=100.0, trades_in_journal=0, stage_cap_pct=2.0, floor_pct=0.25,
    )
    base.update(over)
    return SizingInputs(**base)


def test_realized_vol_population_std():
    assert realized_vol([0.01, -0.01, 0.01, -0.01]) == pytest.approx(0.01, rel=1e-9)
    assert realized_vol([0.02]) == 0.0       # n < 2
    assert realized_vol([]) == 0.0


def test_har_forecast_constant_vol_returns_that_vol():
    assert har_rv_forecast([0.01, -0.01] * 300) == pytest.approx(0.01, rel=1e-6)


def test_har_forecast_blends_short_and_long_horizons():
    # Last 24 bars elevated to 0.02; long horizons stay near 0.01 -> forecast in between.
    rets = [0.01, -0.01] * 300 + [0.02, -0.02] * 12
    f = har_rv_forecast(rets, windows=(24, 120, 480))
    assert 0.01 < f < 0.02


def test_vol_target_scale_sizes_down_in_high_vol_up_in_low():
    assert vol_target_scale(0.01, 0.01) == pytest.approx(1.0)
    assert vol_target_scale(0.02, 0.01) == pytest.approx(0.5)    # high vol -> clamp lo
    assert vol_target_scale(0.005, 0.01) == pytest.approx(1.5)   # low vol -> clamp hi
    assert vol_target_scale(0.0, 0.01) == 1.0                    # degenerate guard


def test_sizer_scales_risk_by_vol_scale():
    full = PositionSizer().size(_inp())
    scaled = PositionSizer().size(_inp(vol_scale=0.5))
    assert scaled.risk_pct == pytest.approx(full.risk_pct * 0.5, rel=1e-9)
