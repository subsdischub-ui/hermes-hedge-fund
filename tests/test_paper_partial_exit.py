"""Exercise actual paper accounting and persistence with local, explicit prices."""

from copy import deepcopy

import pytest

from hermes.execution import paper
from hermes.execution.base import Order

SYMBOL = "BTC/USDC"
SKILL = "partial_exit_regression"
KEY = f"{SYMBOL}::{SKILL}"


@pytest.fixture
def local_market(monkeypatch):
    class LocalMarket:
        mark = 100.0

        async def ticker(self, symbol):
            assert symbol == SYMBOL
            return {"last": self.mark}

    market = LocalMarket()
    # Never construct a venue client. Only the price boundary is replaced;
    # place_order, equity_usd, open_positions and disk persistence remain real.
    monkeypatch.setattr(paper, "get_market", lambda: market)
    return market


async def open_position(exchange, direction):
    return await exchange.place_order(Order(
        symbol=SYMBOL,
        side="BUY" if direction == 1 else "SELL",
        qty=2,
        price=100,
        stop_price=100 - direction * 20,
        take_profit=100 + direction * 50,
        extra={"skill_id": SKILL, "thesis": "original thesis"},
    ))


async def assert_persisted(exchange, fee_bps, expected_cash, expected_nav):
    expected_state = deepcopy(exchange._state)
    expected_positions = await exchange.open_positions()
    reloaded = paper.PaperExchange(
        exchange.state_path, starting_equity=9999, fee_bps=fee_bps, slippage_bps=0,
    )
    assert reloaded._state == expected_state
    assert reloaded._state["cash_usd"] == pytest.approx(expected_cash)
    assert await reloaded.equity_usd() == pytest.approx(expected_nav)
    assert await reloaded.open_positions() == expected_positions
    assert await reloaded.history() == await exchange.history()
    return reloaded


@pytest.mark.parametrize("direction", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("fee_bps", [0, 10], ids=["zero-fees", "fees"])
@pytest.mark.parametrize("exit_qty", [1, 2, 3], ids=["partial", "full", "flip"])
@pytest.mark.parametrize("new_protection", [False, True], ids=["no-stop-tp", "new-stop-tp"])
async def test_opposite_order_accounting_and_metadata(
    tmp_path, local_market, direction, fee_bps, exit_qty, new_protection,
):
    exchange = paper.PaperExchange(
        tmp_path / "paper.json", starting_equity=1000, fee_bps=fee_bps, slippage_bps=0,
    )
    rate = fee_bps / 10_000
    opening = await open_position(exchange, direction)
    original = deepcopy(exchange._state["positions"][KEY])
    assert opening.fee_usd == pytest.approx(200 * rate)
    assert exchange._state["cash_usd"] == pytest.approx(1000 - direction * 200 - 200 * rate)
    assert await exchange.equity_usd() == pytest.approx(1000 - 200 * rate)

    price = 100 + direction * 10
    local_market.mark = price
    stop = price + direction * 20 if new_protection else None
    take_profit = price - direction * 30 if new_protection else None
    fill = await exchange.place_order(Order(
        symbol=SYMBOL,
        side="SELL" if direction == 1 else "BUY",
        qty=exit_qty,
        price=price,
        stop_price=stop,
        take_profit=take_profit,
        extra={"skill_id": SKILL, "thesis": "must not replace original thesis"},
    ))
    assert fill.price == price
    assert fill.qty == exit_qty
    assert fill.side == ("SELL" if direction == 1 else "BUY")
    assert fill.fee_usd == pytest.approx(exit_qty * price * rate)

    closed_qty = min(exit_qty, 2)
    # Cash pays the full fill fee even on a flip; history allocates only the
    # closed portion's fee to realized PnL.
    booked_fees = (200 + exit_qty * price) * rate
    expected_cash = 1000 - direction * 200 + direction * exit_qty * price - booked_fees
    expected_nav = 1020 - booked_fees
    assert exchange._state["cash_usd"] == pytest.approx(expected_cash)
    assert await exchange.equity_usd() == pytest.approx(expected_nav)
    history = await exchange.history()
    assert len(history) == 1
    assert history[0]["qty_closed"] == closed_qty
    assert history[0]["pnl_usd"] == pytest.approx(closed_qty * (10 - price * rate))
    assert history[0]["skill_id"] == SKILL
    assert history[0]["thesis"] == original["thesis"]

    if exit_qty == 1:
        # Whole-record equality protects entry, stops, TP, timestamp and thesis,
        # both when the reduction omits protection and supplies replacements.
        assert exchange._state["positions"][KEY] == {**original, "qty": direction}
        positions = await exchange.open_positions()
        assert len(positions) == 1
        assert positions[0].entry_price == 100
        assert positions[0].unrealized_pnl_usd == pytest.approx(10)
    elif exit_qty == 2:
        assert exchange._state["positions"] == {}
        assert await exchange.open_positions() == []
    else:
        position = exchange._state["positions"][KEY]
        assert position == {
            **original,
            "qty": -direction,
            "entry_price": price,
            "stop_price": stop,
            "take_profit": take_profit,
            "opened_at": position["opened_at"],
        }
        assert position["opened_at"] > original["opened_at"]
        positions = await exchange.open_positions()
        assert len(positions) == 1
        assert positions[0].unrealized_pnl_usd == 0

    await assert_persisted(exchange, fee_bps, expected_cash, expected_nav)


@pytest.mark.parametrize("direction", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("fee_bps", [0, 10], ids=["zero-fees", "fees"])
async def test_repeated_reductions_across_reloads(tmp_path, local_market, direction, fee_bps):
    exchange = paper.PaperExchange(
        tmp_path / "paper.json", starting_equity=1000, fee_bps=fee_bps, slippage_bps=0,
    )
    await open_position(exchange, direction)
    original = deepcopy(exchange._state["positions"][KEY])
    rate = fee_bps / 10_000
    exit_notional = 0
    gross_realized = 0
    for step in range(1, 5):
        price = 100 + direction * step * 10
        local_market.mark = price
        fill = await exchange.place_order(Order(
            symbol=SYMBOL,
            side="SELL" if direction == 1 else "BUY",
            qty=0.5,
            price=price,
            extra={"skill_id": SKILL, "thesis": "reduction"},
        ))
        assert fill.fee_usd == pytest.approx(0.5 * price * rate)
        exit_notional += 0.5 * price
        gross_realized += 0.5 * step * 10
        remaining = 2 - step * 0.5
        fees = (200 + exit_notional) * rate
        expected_cash = 1000 - direction * 200 + direction * exit_notional - fees
        expected_nav = 1000 + gross_realized + remaining * step * 10 - fees
        assert exchange._state["cash_usd"] == pytest.approx(expected_cash)
        assert await exchange.equity_usd() == pytest.approx(expected_nav)
        if remaining:
            assert exchange._state["positions"][KEY] == {
                **original, "qty": direction * remaining,
            }
        else:
            assert exchange._state["positions"] == {}
        history = await exchange.history()
        assert len(history) == step
        assert history[-1]["qty_closed"] == 0.5
        assert history[-1]["pnl_usd"] == pytest.approx(0.5 * (step * 10 - price * rate))
        assert history[-1]["thesis"] == original["thesis"]
        assert sum(item["pnl_usd"] for item in history) == pytest.approx(
            gross_realized - exit_notional * rate,
        )
        exchange = await assert_persisted(exchange, fee_bps, expected_cash, expected_nav)
