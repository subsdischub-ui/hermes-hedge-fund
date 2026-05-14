from hermes.risk.sizing import PositionSizer, SizingInputs


def test_cold_start_uses_floor_when_no_kelly():
    sizer = PositionSizer()
    out = sizer.size(
        SizingInputs(
            equity_usd=100,
            confidence=0.0,
            win_rate=None,
            avg_win_loss_ratio=None,
            atr=100,        # 0.2% of price
            price=50_000,
            trades_in_journal=0,
            stage_cap_pct=0.5,
            floor_pct=0.25,
        )
    )
    # Floor is 0.25% of $100 = $0.25 of risk; with stop_dist 2*100/50_000 = 0.4%
    # -> notional ≈ 0.25 / 0.004 = $62.5
    assert 0.0025 <= out.risk_pct <= 0.005
    assert out.notional_usd > 0


def test_kelly_kicks_in_after_threshold():
    sizer = PositionSizer()
    out = sizer.size(
        SizingInputs(
            equity_usd=200,
            confidence=0.8,
            win_rate=0.55,
            avg_win_loss_ratio=1.5,
            atr=100,
            price=50_000,
            trades_in_journal=200,
            stage_cap_pct=2.0,
            floor_pct=0.25,
            fractional_kelly=0.20,
        )
    )
    # Kelly = (0.55 * 1.5 - 0.45) / 1.5 = 0.25  -> 0.20*0.25*0.8 = 0.04 = 4% > cap 2%
    assert out.risk_pct == 0.02
