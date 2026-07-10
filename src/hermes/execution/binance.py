"""Binance USDT-M perp adapter via CCXT (production path — the operator's REAL
account, and the same venue the strategies were backtested on).

Safety contract (mirrors hyperliquid.py / bybit.py)
----------------------------------------------------
- Refuses to instantiate without BOTH BINANCE_API_KEY and BINANCE_API_SECRET —
  "refusing live mode", same phrasing/behavior as HyperliquidExchange.
- The delta-neutral carry pair (funding_carry_harvest) has no real spot+perp leg
  wired yet, so place_order raises NotImplementedError with a clear message —
  the engine gets a clean error, never a naked single-leg order. paper.py routes
  this skill_id to `_open_carry` (a hedged two-leg pair); we route it to a
  hard refusal until the real two-leg execution exists AND is validated.
- No order without a stop_price — the same contract the risk engine already
  imposes structurally on every order it builds for the paper exchange
  (guardrails.py always attaches stop/take_profit before calling place_order).

Position tracking
------------------
Like bybit.py, open_positions() is reconstructed from the OPEN journal rows
(ownership) — NEVER from the exchange's raw position list — because the
operator's Binance account may carry positions Hermès never opened. The live
exchange position (fetch_positions) is consulted only to CAP the journal's
claimed quantity, so a TP/SL that already fired on-exchange (attached as a
reduce-only trigger order, outside our own event loop) is never reported as
still open.
"""

from __future__ import annotations

import hashlib
from datetime import UTC
from typing import Any

import ccxt.async_support as ccxt
from sqlalchemy import select

from hermes.execution.base import Exchange, Fill, Order, Position, Side
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


# --- pure helpers (unit-tested without ccxt / network / credentials) --------
def derive_client_order_id(client_id: str, suffix: str = "") -> str:
    """Deterministic Binance-safe clientOrderId (<=36 chars, alnum + '-') from
    our client_id (+ optional suffix for the SL/TP legs), so a network retry of
    the same logical order reuses the id and Binance's own de-dup rejects the
    duplicate instead of opening a second position. Mirrors
    hyperliquid.derive_cloid's idempotency intent."""
    basis = f"{client_id}:{suffix}" if suffix else str(client_id)
    return "hermes-" + hashlib.sha256(basis.encode()).hexdigest()[:28]


def build_reduce_only_legs(
    side: Side, qty: float, stop_price: float | None, take_profit: float | None
) -> list[dict[str, Any]]:
    """Reduce-only STOP_MARKET / TAKE_PROFIT_MARKET legs (as ccxt unified
    create_order params: type='market' + these params) that close the position
    on the exchange, so a live position stays protected even if the agent dies.
    Mirrors hyperliquid.build_tpsl_orders. A BUY (long) is closed by a SELL leg,
    a SELL (short) by a BUY leg. Skips legs with no price."""
    close_side: Side = "SELL" if side == "BUY" else "BUY"
    legs: list[dict[str, Any]] = []
    if stop_price is not None:
        legs.append(
            {
                "side": close_side,
                "amount": float(qty),
                "tag": "sl",
                "params": {"stopLossPrice": float(stop_price), "reduceOnly": True},
            }
        )
    if take_profit is not None:
        legs.append(
            {
                "side": close_side,
                "amount": float(qty),
                "tag": "tp",
                "params": {"takeProfitPrice": float(take_profit), "reduceOnly": True},
            }
        )
    return legs


class BinanceUSDMExchange(Exchange):
    name = "binanceusdm"
    # Same routing key as PaperExchange.CARRY_SKILL — the carry skill's ONLY
    # executable form is a hedged two-leg pair, never a naked single-leg order.
    CARRY_SKILL = "funding_carry_harvest"

    def __init__(self) -> None:
        s = get_settings()
        key = (
            s.binance_api_key.get_secret_value()
            if hasattr(s.binance_api_key, "get_secret_value")
            else str(s.binance_api_key)
        )
        secret = (
            s.binance_api_secret.get_secret_value()
            if hasattr(s.binance_api_secret, "get_secret_value")
            else str(s.binance_api_secret)
        )
        if not key or not secret:
            raise RuntimeError("Binance credentials missing — refusing live mode.")
        self.client = ccxt.binanceusdm(
            {
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
            }
        )
        if s.binance_testnet:
            self.client.set_sandbox_mode(True)

    # --- read paths -----------------------------------------------------
    async def equity_usd(self) -> float:
        bal = await self.client.fetch_balance()
        total = bal.get("total") or {}
        return float(total.get("USDT", 0.0)) + float(total.get("USDC", 0.0))

    async def last_price(self, symbol: str) -> float:
        t = await self.client.fetch_ticker(symbol)
        return float(t["last"])

    async def open_positions(self) -> list[Position]:
        from datetime import datetime

        from hermes.memory.store import JournalRow, get_db

        db = get_db()
        async with db.session() as s:
            rows = (
                await s.execute(select(JournalRow).where(JournalRow.closed_at.is_(None)))
            ).scalars().all()
        if not rows:
            return []

        symbols = list({row.symbol for row in rows})
        try:
            live = await self.client.fetch_positions(symbols)
        except Exception as e:
            log.warning("binanceusdm_fetch_positions_failed", error=str(e))
            live = []
        live_by_symbol = {p.get("symbol"): p for p in live}

        out: list[Position] = []
        for row in rows:
            live_pos = live_by_symbol.get(row.symbol)
            on_book = float((live_pos or {}).get("contracts") or 0.0)
            if on_book <= 0:
                # Exchange already closed this out (TP/SL trigger fired) — never
                # report a journal-open row the exchange no longer carries.
                continue
            qty = min(on_book, float(row.qty))
            signed_qty = qty if row.side == "BUY" else -qty
            try:
                t = await self.client.fetch_ticker(row.symbol)
                px = float(t["last"])
            except Exception:
                px = float(row.entry_price)
            out.append(
                Position(
                    symbol=row.symbol,
                    qty=signed_qty,
                    entry_price=float(row.entry_price),
                    stop_price=None,
                    take_profit=None,
                    opened_at=row.opened_at or datetime.now(UTC),
                    unrealized_pnl_usd=(px - float(row.entry_price)) * signed_qty,
                    skill_id=row.skill_id or "",
                    thesis=row.thesis or "",
                )
            )
        return out

    # --- write paths ------------------------------------------------------
    async def place_order(self, order: Order) -> Fill:
        # Structural safety: the carry skill can ONLY open a hedged two-leg pair
        # (short perp + long spot). That real execution isn't built/validated yet,
        # so the engine gets a clean, explicit error — never a naked perp order.
        if order.extra.get("skill_id") == self.CARRY_SKILL:
            raise NotImplementedError(
                "binanceusdm: carry pair execution réelle non câblée — HOLD-only "
                "tant que la jambe spot+perp réelle n'est pas construite ET validée."
            )
        # Same contract the risk engine already imposes on the paper exchange:
        # every order it builds carries a stop_price. Enforce it here too so a
        # bug upstream can never send a naked order to REAL capital.
        if order.stop_price is None:
            raise RuntimeError(
                "binanceusdm refuses to place an order without a stop_price "
                "(no naked orders on live capital)"
            )

        await self.client.load_markets()
        amount = float(self.client.amount_to_precision(order.symbol, order.qty))
        cloid = derive_client_order_id(order.client_id)
        r = await self.client.create_order(
            symbol=order.symbol,
            type="market",
            side=order.side.lower(),
            amount=amount,
            params={"clientOrderId": cloid},
        )
        filled_qty = float(r.get("filled") or amount)
        fill = Fill(
            order_id=str(r.get("id", cloid)),
            symbol=order.symbol,
            side=order.side,
            qty=filled_qty,
            price=float(r.get("average") or r.get("price") or 0.0),
            fee_usd=float((r.get("fee") or {}).get("cost", 0.0)),
        )

        # Attach reduce-only stop/TP triggers so the position is protected
        # on-exchange even if the agent dies — mirrors hyperliquid.py's tpsl attach.
        for leg in build_reduce_only_legs(order.side, filled_qty, order.stop_price, order.take_profit):
            try:
                leg_amount = float(self.client.amount_to_precision(order.symbol, leg["amount"]))
                leg_cloid = derive_client_order_id(order.client_id, suffix=leg["tag"])
                await self.client.create_order(
                    symbol=order.symbol,
                    type="market",
                    side=leg["side"].lower(),
                    amount=leg_amount,
                    params={**leg["params"], "clientOrderId": leg_cloid},
                )
            except Exception as e:
                log.warning(
                    "binanceusdm_tpsl_attach_failed",
                    symbol=order.symbol,
                    tag=leg["tag"],
                    error=str(e),
                )
        return fill

    async def close_position(
        self, symbol: str, reason: str, skill_id: str | None = None
    ) -> Fill | None:
        from hermes.memory.store import JournalRow, get_db

        db = get_db()
        async with db.session() as s:
            query = select(JournalRow).where(
                JournalRow.symbol == symbol, JournalRow.closed_at.is_(None)
            )
            if skill_id is not None:
                query = query.where(JournalRow.skill_id == skill_id)
            row = (
                await s.execute(query.order_by(JournalRow.id.asc()).limit(1))
            ).scalar_one_or_none()
        if row is None:
            return None

        try:
            live = await self.client.fetch_positions([symbol])
        except Exception as e:
            log.warning("binanceusdm_fetch_positions_failed", symbol=symbol, error=str(e))
            live = []
        live_pos = next((p for p in live if p.get("symbol") == symbol), None)
        on_book = float((live_pos or {}).get("contracts") or 0.0)
        qty = min(on_book, float(row.qty))
        if qty <= 0:
            return None

        close_side: Side = "SELL" if row.side == "BUY" else "BUY"
        await self.client.load_markets()
        amount = float(self.client.amount_to_precision(symbol, qty))
        cloid = derive_client_order_id(f"close-{row.id}", suffix=reason)
        r = await self.client.create_order(
            symbol=symbol,
            type="market",
            side=close_side.lower(),
            amount=amount,
            params={"clientOrderId": cloid, "reduceOnly": True},
        )
        return Fill(
            order_id=str(r.get("id", cloid)),
            symbol=symbol,
            side=close_side,
            qty=float(r.get("filled") or amount),
            price=float(r.get("average") or r.get("price") or 0.0),
            fee_usd=float((r.get("fee") or {}).get("cost", 0.0)),
        )

    async def close(self) -> None:
        await self.client.close()
