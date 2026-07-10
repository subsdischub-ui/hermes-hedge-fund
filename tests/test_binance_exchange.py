"""BinanceUSDMExchange behavioral tests. AUCUN appel réseau, AUCUNE clé réelle —
the ccxt client is replaced with mocks after construction (fake credentials
only make ccxt build a local client object; no I/O happens until a method is
called, and every method we exercise is mocked below).

Covers:
- Refuses to instantiate without BOTH BINANCE_API_KEY and BINANCE_API_SECRET.
- place_order rounds qty to exchange precision, places the market entry, then
  attaches reduce-only STOP_MARKET + TAKE_PROFIT_MARKET legs.
- place_order's clientOrderId is deterministic (idempotent across retries).
- place_order refuses a naked order (no stop_price) and the carry skill_id
  (routes to a clean NotImplementedError, never a naked order).
- open_positions is reconstructed from the journal, capped by (never sourced
  from) the raw exchange position list.
- close_position places a reduce-only market order sized from journal ∩ live.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes.settings as settings_mod
from hermes.execution.base import Order
from hermes.execution.binance import BinanceUSDMExchange
from hermes.memory.journal import JournalEntry, TradeJournal
from hermes.memory.store import get_db


@pytest.fixture
def binance_exchange(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    settings_mod.get_settings.cache_clear()
    ex = BinanceUSDMExchange()
    ex.client = MagicMock()
    ex.client.load_markets = AsyncMock(return_value={})
    # Real ccxt truncates to the market's precision; the mock rounds to 3dp so
    # tests can assert the adapter actually calls through amount_to_precision
    # instead of sending the raw, unrounded qty.
    ex.client.amount_to_precision = MagicMock(side_effect=lambda symbol, qty: round(float(qty), 3))
    ex.client.fetch_balance = AsyncMock(return_value={"total": {"USDT": 250.0}})
    ex.client.fetch_ticker = AsyncMock(return_value={"last": 100.0})
    ex.client.fetch_positions = AsyncMock(return_value=[])
    ex.client.create_order = AsyncMock(
        return_value={"id": "exch-1", "filled": 1.0, "average": 100.0, "fee": {"cost": 0.05}}
    )
    yield ex
    settings_mod.get_settings.cache_clear()


# --- credential refusal (safety-critical) -----------------------------------
@pytest.mark.asyncio
async def test_refuses_instantiation_without_any_credentials(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "")
    monkeypatch.setenv("BINANCE_API_SECRET", "")
    settings_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="refusing live mode"):
            BinanceUSDMExchange()
    finally:
        settings_mod.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_refuses_instantiation_missing_secret_only(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "")
    settings_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="refusing live mode"):
            BinanceUSDMExchange()
    finally:
        settings_mod.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_refuses_instantiation_missing_key_only(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    settings_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="refusing live mode"):
            BinanceUSDMExchange()
    finally:
        settings_mod.get_settings.cache_clear()


# --- reads -------------------------------------------------------------
@pytest.mark.asyncio
async def test_equity_usd_reads_futures_usdt_balance(binance_exchange):
    assert await binance_exchange.equity_usd() == pytest.approx(250.0)


# --- place_order ---------------------------------------------------------
@pytest.mark.asyncio
async def test_place_order_rounds_precision_places_entry_then_attaches_stop_and_tp(binance_exchange):
    ex = binance_exchange
    order = Order(
        symbol="BTC/USDT:USDT", side="BUY", qty=0.123456, stop_price=90000.0,
        take_profit=110000.0, client_id="trade-abc",
    )
    fill = await ex.place_order(order)
    assert fill.symbol == "BTC/USDT:USDT" and fill.side == "BUY"

    # 3 create_order calls: the market entry + the SL leg + the TP leg.
    assert ex.client.create_order.await_count == 3
    calls = ex.client.create_order.await_args_list

    entry_kwargs = calls[0].kwargs
    assert entry_kwargs["symbol"] == "BTC/USDT:USDT"
    assert entry_kwargs["type"] == "market"
    assert entry_kwargs["side"] == "buy"
    assert entry_kwargs["amount"] == pytest.approx(0.123)   # rounded via amount_to_precision
    assert "clientOrderId" in entry_kwargs["params"]

    sl_kwargs = next(c.kwargs for c in calls if c.kwargs["params"].get("stopLossPrice") == 90000.0)
    assert sl_kwargs["side"] == "sell"          # closing a long -> SELL leg
    assert sl_kwargs["params"]["reduceOnly"] is True

    tp_kwargs = next(c.kwargs for c in calls if c.kwargs["params"].get("takeProfitPrice") == 110000.0)
    assert tp_kwargs["side"] == "sell"
    assert tp_kwargs["params"]["reduceOnly"] is True


@pytest.mark.asyncio
async def test_place_order_client_order_id_is_deterministic(binance_exchange):
    ex = binance_exchange
    order = Order(
        symbol="ETH/USDT:USDT", side="BUY", qty=1.0, stop_price=1000.0,
        take_profit=2000.0, client_id="trade-idem",
    )
    await ex.place_order(order)
    first_cloid = ex.client.create_order.await_args_list[0].kwargs["params"]["clientOrderId"]

    ex.client.create_order.reset_mock()
    await ex.place_order(order)
    second_cloid = ex.client.create_order.await_args_list[0].kwargs["params"]["clientOrderId"]

    assert first_cloid == second_cloid   # same logical order -> same clientOrderId -> exchange dedups


@pytest.mark.asyncio
async def test_place_order_refuses_naked_order_without_stop(binance_exchange):
    ex = binance_exchange
    order = Order(symbol="BTC/USDT:USDT", side="BUY", qty=0.01, stop_price=None, take_profit=110000.0)
    with pytest.raises(RuntimeError, match="stop_price"):
        await ex.place_order(order)
    ex.client.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_place_order_carry_skill_raises_clean_not_implemented_never_a_naked_order(binance_exchange):
    ex = binance_exchange
    order = Order(
        symbol="BTC/USDT:USDT", side="SELL", qty=0.01, stop_price=90000.0,
        extra={"skill_id": "funding_carry_harvest"},
    )
    with pytest.raises(NotImplementedError, match="carry pair"):
        await ex.place_order(order)
    ex.client.create_order.assert_not_called()


# --- open_positions / close_position (journal-sourced, never raw account) ---
@pytest.mark.asyncio
async def test_open_positions_reconstructed_from_journal_capped_by_live_exchange(binance_exchange):
    await get_db().init()
    journal = TradeJournal()
    now = datetime.now(UTC)
    await journal.open(JournalEntry(
        symbol="SOL/USDT:USDT", side="BUY", qty=2.0, entry_price=100.0, opened_at=now,
        skill_id="mean_reversion_regime",
    ))
    await journal.open(JournalEntry(
        symbol="AVAX/USDT:USDT", side="BUY", qty=5.0, entry_price=20.0, opened_at=now,
        skill_id="mean_reversion_regime",
    ))

    ex = binance_exchange

    async def _fake_fetch_positions(symbols):
        # SOL still carries 1.5 contracts on-exchange (< journal's 2.0 -> capped);
        # AVAX was already closed on-exchange (its TP/SL trigger fired) -> flat.
        return [{"symbol": "SOL/USDT:USDT", "contracts": 1.5}]

    ex.client.fetch_positions = AsyncMock(side_effect=_fake_fetch_positions)
    ex.client.fetch_ticker = AsyncMock(return_value={"last": 105.0})

    positions = await ex.open_positions()
    symbols = {p.symbol for p in positions}
    assert symbols == {"SOL/USDT:USDT"}   # AVAX not reported — exchange already flat
    sol = next(p for p in positions if p.symbol == "SOL/USDT:USDT")
    assert sol.qty == pytest.approx(1.5)   # capped at the live exchange size, not the journal's 2.0
    assert sol.qty > 0                     # BUY -> long -> positive signed qty


@pytest.mark.asyncio
async def test_close_position_places_reduce_only_market_sized_from_journal_and_live(binance_exchange):
    await get_db().init()
    journal = TradeJournal()
    now = datetime.now(UTC)
    await journal.open(JournalEntry(
        symbol="ARB/USDT:USDT", side="SELL", qty=10.0, entry_price=1.0, opened_at=now,
        skill_id="mean_reversion_regime",
    ))

    ex = binance_exchange
    ex.client.fetch_positions = AsyncMock(return_value=[{"symbol": "ARB/USDT:USDT", "contracts": 7.0}])

    fill = await ex.close_position("ARB/USDT:USDT", reason="stop")
    assert fill is not None
    assert fill.side == "BUY"    # closing a SELL (short) via BUY

    close_kwargs = ex.client.create_order.await_args_list[0].kwargs
    assert close_kwargs["side"] == "buy"
    assert close_kwargs["amount"] == pytest.approx(7.0)   # capped by the live exchange size
    assert close_kwargs["params"]["reduceOnly"] is True


@pytest.mark.asyncio
async def test_close_position_returns_none_when_no_open_journal_row(binance_exchange):
    result = await binance_exchange.close_position("DOGE/USDT:USDT", reason="manual")
    assert result is None
    binance_exchange.client.create_order.assert_not_called()
