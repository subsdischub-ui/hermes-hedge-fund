"""Per-asset / per-skill PnL breakdown for the dashboard — so the SOL lean (and
any single-asset dependence) is visible live without hard-coding asset cuts."""

from __future__ import annotations

from types import SimpleNamespace

from hermes.memory.breakdown import aggregate_breakdown


def _row(symbol, skill, pnl):
    return SimpleNamespace(symbol=symbol, skill_id=skill, pnl_usd=pnl)


def test_aggregate_by_symbol():
    rows = [
        _row("BTC/USDC", "ts_mom_vol_scaled", 1.0),
        _row("BTC/USDC", "ts_mom_vol_scaled", -0.5),
        _row("SOL/USDC", "mean_reversion_regime", 2.0),
    ]
    b = aggregate_breakdown(rows, "symbol")
    assert b["BTC/USDC"]["n"] == 2
    assert abs(b["BTC/USDC"]["pnl_usd"] - 0.5) < 1e-9
    assert b["BTC/USDC"]["win_rate"] == 0.5
    assert b["SOL/USDC"]["n"] == 1
    assert b["SOL/USDC"]["win_rate"] == 1.0


def test_aggregate_by_skill():
    rows = [
        _row("BTC/USDC", "ts_mom_vol_scaled", 1.0),
        _row("ETH/USDC", "ts_mom_vol_scaled", 1.0),
        _row("SOL/USDC", "mean_reversion_regime", -1.0),
    ]
    b = aggregate_breakdown(rows, "skill_id")
    assert b["ts_mom_vol_scaled"]["n"] == 2
    assert b["ts_mom_vol_scaled"]["win_rate"] == 1.0
    assert b["mean_reversion_regime"]["pnl_usd"] == -1.0


def test_aggregate_handles_none_pnl_and_empty():
    assert aggregate_breakdown([], "symbol") == {}
    b = aggregate_breakdown([_row("BTC/USDC", "x", None)], "symbol")
    assert b["BTC/USDC"]["n"] == 1
    assert b["BTC/USDC"]["pnl_usd"] == 0.0
