import asyncio
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


@pytest.mark.asyncio
async def test_paper_roundtrip(tmp_data_dir: Path, monkeypatch):
    state_path = tmp_data_dir / "paper_state_test.json"
    if state_path.exists():
        state_path.unlink()
    px = PaperExchange(state_path=state_path, starting_equity=100.0, venue_for_prices="binance")
    px.market = _StubMarket()  # type: ignore[assignment]

    eq0 = await px.equity_usd()
    assert eq0 == 100.0

    fill = await px.place_order(Order(symbol="BTC/USDC", side="BUY", qty=0.001))
    assert fill.price == 50_000.0
    pos = await px.open_positions()
    assert len(pos) == 1 and pos[0].symbol == "BTC/USDC"

    # Move the market up 1% and close.
    px.market.last_price_value = 50_500.0  # type: ignore[attr-defined]
    closed = await px.close_position("BTC/USDC", reason="manual")
    assert closed is not None
    pos_after = await px.open_positions()
    assert pos_after == []

    # State should be persisted to disk.
    payload = json.loads(state_path.read_text())
    assert payload["positions"] == {}
    assert payload["history"], "trade history should be recorded"
