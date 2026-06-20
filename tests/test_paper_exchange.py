import json
from pathlib import Path

import pytest

from hermes.execution.base import Order
from hermes.execution.paper import PaperExchange


class _StubMarket:
    """Stub the CCXT ticker so the test stays offline."""

    last_price_value = 50_000.0

    async def ticker(self, symbol):
        return {"last": self.last_price_value}

    async def close(self):
        pass


def _make_px(tmp_data_dir: Path, name: str = "paper_state_test.json") -> PaperExchange:
    state_path = tmp_data_dir / name
    if state_path.exists():
        state_path.unlink()
    px = PaperExchange(state_path=state_path, starting_equity=100.0, venue_for_prices="kraken")
    px.market = _StubMarket()  # type: ignore[assignment]
    return px


@pytest.mark.asyncio
async def test_paper_roundtrip_preserves_equity(tmp_data_dir: Path):
    px = _make_px(tmp_data_dir)
    eq0 = await px.equity_usd()
    assert eq0 == pytest.approx(100.0, rel=1e-9)

    # Buy 0.001 BTC at $50k → roughly $50 notional, $0.025 fee + $0.015 slippage.
    fill_buy = await px.place_order(Order(symbol="BTC/USDC", side="BUY", qty=0.001))
    assert fill_buy.side == "BUY"
    assert fill_buy.price > 50_000.0  # paper slippage applied
    pos = await px.open_positions()
    assert len(pos) == 1

    # Market moves up 1%; close.
    px.market.last_price_value = 50_500.0  # type: ignore[attr-defined]
    fill_close = await px.close_position("BTC/USDC", reason="manual")
    assert fill_close is not None
    pos_after = await px.open_positions()
    assert pos_after == []

    eq1 = await px.equity_usd()
    # Equity should have moved by approximately (close_price - entry_price) * qty,
    # minus two rounds of fees and the slippage cost (≈ $0.03–0.10 each side).
    # We assert equity is back near $100 ± a few cents, not artificially destroyed.
    assert 99.0 <= eq1 <= 101.5, f"equity drifted to {eq1} after a small profitable roundtrip"

    payload = json.loads(Path(px.state_path).read_text())
    assert payload["positions"] == {}
    assert payload["history"], "trade history should be recorded"


@pytest.mark.asyncio
async def test_paper_fees_are_injectable_and_applied(tmp_data_dir: Path):
    """Fees must be configurable (so paper mirrors the real venue, e.g. Hyperliquid
    spot taker = 7 bps, not the old hardcoded 5). With slippage isolated to 0, a
    flat buy+sell round-trip costs EXACTLY two taker fees on the notional."""
    state_path = tmp_data_dir / "paper_fee_inject.json"
    if state_path.exists():
        state_path.unlink()
    px = PaperExchange(
        state_path=state_path,
        starting_equity=100.0,
        venue_for_prices="kraken",
        fee_bps=7,
        slippage_bps=0,
    )
    px.market = _StubMarket()  # type: ignore[assignment]

    eq0 = await px.equity_usd()
    await px.place_order(Order(symbol="BTC/USDC", side="BUY", qty=0.001))   # $50 notional
    await px.close_position("BTC/USDC", reason="manual")
    eq1 = await px.equity_usd()

    # 50 * 0.0007 = 0.035 per side, two sides = 0.07, no slippage.
    assert (eq0 - eq1) == pytest.approx(0.07, abs=1e-6)


@pytest.mark.asyncio
async def test_paper_flat_roundtrip_loses_only_fees_and_slippage(tmp_data_dir: Path):
    """Buy then immediately sell at the same mid — final equity should only be
    short by fees + 2x slippage, NEVER by the notional."""
    px = _make_px(tmp_data_dir, name="paper_flat.json")
    eq0 = await px.equity_usd()

    await px.place_order(Order(symbol="BTC/USDC", side="BUY", qty=0.001))
    await px.close_position("BTC/USDC", reason="manual")

    eq1 = await px.equity_usd()
    # Worst case: 2 * fee + 2 * slippage on $50 notional ≈ $0.08
    loss = eq0 - eq1
    assert 0 <= loss < 0.5, f"flat roundtrip should only cost fees+slippage, lost {loss}"
