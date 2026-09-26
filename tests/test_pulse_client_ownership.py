"""Pulse must never close the market adapter borrowed by the trading graph."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest


@pytest.fixture
def clients(monkeypatch, tmp_path):
    mod = importlib.import_module("hermes.runtime.pulse")
    frame = pd.DataFrame({"close": [100.0] * 60})

    class Adapter:
        def __init__(self):
            self.closed = False
            self.close_count = 0

        async def ohlcv(self, *args, **kwargs):
            if self.closed:
                raise RuntimeError("adapter closed")
            return frame

        async def funding_history(self, *args):
            return pd.DataFrame()

        async def close(self):
            self.closed = True
            self.close_count += 1

    shared, owned = Adapter(), Adapter()
    settings = SimpleNamespace(
        universe=["BTC/USDT:USDT"], data_dir=tmp_path,
        exchange_primary="binanceusdm", yaml=lambda *a, **k: "4h",
    )
    monkeypatch.setattr(mod, "get_settings", lambda: settings)
    monkeypatch.setattr(mod, "load_selective_config", lambda s: object())
    monkeypatch.setattr(mod, "get_skill_library", lambda: SimpleNamespace(
        evaluate_for_symbol=lambda *a: [],
    ))
    monkeypatch.setattr(mod, "get_exchange", lambda: SimpleNamespace(
        open_positions=AsyncMock(return_value=[]),
    ))
    monkeypatch.setattr(mod, "pulse_entry", lambda *a: {"kind": "pulse"})
    # Preserve the old factory injection so reverting the ownership fix is RED.
    monkeypatch.setattr(mod, "get_market", lambda: shared, raising=False)
    created = []

    def factory(venue):
        created.append(venue)
        return owned

    monkeypatch.setattr(mod, "MarketData", factory, raising=False)
    return mod, shared, owned, created


@pytest.mark.asyncio
async def test_pulse_then_market_read_preserves_shared_adapter(clients):
    mod, shared, owned, created = clients
    assert await mod.pulse() == 1
    assert len(await shared.ohlcv("BTC/USDT:USDT")) == 60
    assert created == ["binanceusdm"]
    assert shared.close_count == 0
    assert owned.close_count == 1


@pytest.mark.asyncio
async def test_cancelled_positions_read_closes_owned_adapter(clients, monkeypatch):
    mod, shared, owned, _ = clients
    entered = asyncio.Event()

    async def waiting():
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(mod, "get_exchange", lambda: SimpleNamespace(open_positions=waiting))
    task = asyncio.create_task(mod.pulse())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owned.close_count == 1
    assert shared.close_count == 0


@pytest.mark.asyncio
async def test_pulse_read_failure_closes_only_its_adapter(clients):
    mod, shared, owned, _ = clients
    owned.ohlcv = AsyncMock(side_effect=RuntimeError("market unavailable"))
    assert await mod.pulse() == 0
    assert owned.close_count == 1
    assert shared.close_count == 0


@pytest.mark.asyncio
async def test_cancelled_pulse_closes_owned_adapter(clients):
    mod, shared, owned, _ = clients
    entered = asyncio.Event()

    async def waiting(*a, **k):
        entered.set()
        await asyncio.Event().wait()

    owned.ohlcv = waiting
    # Old code borrows shared, so both wait; failure is resource ownership.
    shared.ohlcv = waiting
    task = asyncio.create_task(mod.pulse())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owned.close_count == 1
    assert shared.close_count == 0
