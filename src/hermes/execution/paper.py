"""Paper-trading exchange. Persists state in /data so restarts don't reset equity.

Cash accounting model
---------------------
We treat `cash_usd` as the OPERATOR's free cash. A long is "cash spent on
inventory" — on open we debit (cash) and credit a position; on close we
release the cost basis back into cash AND add realized PnL (minus fee).
A short is "cash received, inventory owed" — on open we credit cash by the
notional and credit a negative-qty position; on close we debit cash by
qty*close_price plus fee, leaving (entry-close)*qty as realized.

Concurrency
-----------
All mutations of `_state` are serialized via `self._lock` so the 5-minute
market tick and the 30-second stop watcher cannot interleave writes. The
state file is written atomically via tmp + os.replace().
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hermes.data.market import get_market
from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.logging_setup import get_logger

log = get_logger(__name__)


class PaperExchange(Exchange):
    name = "paper"
    # Conservative fallbacks only. Real fees are venue-/instrument-specific and
    # injected from config (see config/default.yaml:fees and execution.factory).
    # NOTE: 5 bps under-charged vs Hyperliquid spot taker (7 bps) — that gap
    # inflated the paper Sharpe and could open the stage gates on a fictional edge.
    FEE_BPS = 7      # Hyperliquid spot taker
    SLIPPAGE_BPS = 3  # market-impact assumption for paper fills

    def __init__(
        self,
        state_path: Path,
        starting_equity: float,
        venue_for_prices: str | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ):
        self.state_path = state_path
        self.fee_bps = float(self.FEE_BPS if fee_bps is None else fee_bps)
        self.slippage_bps = float(self.SLIPPAGE_BPS if slippage_bps is None else slippage_bps)
        self.market = get_market(venue_for_prices) if venue_for_prices else get_market()
        if state_path.exists():
            self._state = json.loads(state_path.read_text())
        else:
            self._state = {
                "cash_usd": float(starting_equity),
                "positions": {},   # symbol -> dict
                "history": [],
            }
            self._save()
        self._lock = asyncio.Lock()

    # ---- persistence -------------------------------------------------------
    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + os.replace prevents truncated state after a crash.
        fd, tmp = tempfile.mkstemp(
            prefix=self.state_path.name + ".", dir=str(self.state_path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, default=str)
            os.replace(tmp, self.state_path)
        except Exception:
            try:
                os.unlink(tmp)
            finally:
                raise

    # ---- price helpers ------------------------------------------------------
    async def last_price(self, symbol: str) -> float:
        t = await self.market.ticker(symbol)
        return float(t["last"] or t.get("close") or 0.0)

    async def _safe_last_price(self, symbol: str, fallback: float) -> float:
        try:
            return await self.last_price(symbol)
        except Exception as e:
            log.warning("paper_price_fetch_failed", symbol=symbol, fallback=fallback, error=str(e))
            return fallback

    # ---- views --------------------------------------------------------------
    async def equity_usd(self) -> float:
        total = float(self._state["cash_usd"])
        for sym, p in self._state["positions"].items():
            px = await self._safe_last_price(sym, fallback=float(p["entry_price"]))
            # For longs (qty>0): inventory worth qty*px is added.
            # For shorts (qty<0): liability is |qty|*px; entry already credited cash,
            # so we add qty*px (negative) + qty*(entry-px)*(-1) → net = entry*qty for short.
            # Simpler formulation: equity = cash + Σ qty*px for longs - liability for shorts.
            if p["qty"] >= 0:
                total += float(p["qty"]) * px
            else:
                # short: cash was credited entry*|qty|; current liability is |qty|*px.
                # Net contribution = -|qty|*px + |qty|*entry = qty*(entry-px) → qty*entry - qty*px
                # But cash already holds entry*|qty| from open; we only need to add the
                # mark-to-market delta: qty*(entry - px) = (entry-px)*qty (qty<0).
                total += float(p["qty"]) * (px - float(p["entry_price"]))
        return total

    async def open_positions(self) -> list[Position]:
        out: list[Position] = []
        for sym, p in self._state["positions"].items():
            last = await self._safe_last_price(sym, fallback=float(p["entry_price"]))
            opened_at = p["opened_at"]
            if isinstance(opened_at, str):
                opened_at = datetime.fromisoformat(opened_at)
            unrealized = (last - float(p["entry_price"])) * float(p["qty"])
            out.append(
                Position(
                    symbol=sym,
                    qty=float(p["qty"]),
                    entry_price=float(p["entry_price"]),
                    stop_price=p.get("stop_price"),
                    take_profit=p.get("take_profit"),
                    opened_at=opened_at,
                    unrealized_pnl_usd=unrealized,
                    skill_id=p.get("skill_id", ""),
                    thesis=p.get("thesis", ""),
                )
            )
        return out

    # ---- order entry --------------------------------------------------------
    async def place_order(self, order: Order) -> Fill:
        async with self._lock:
            mid = order.price or await self.last_price(order.symbol)
            # Apply paper slippage: BUY pays slightly above mid, SELL receives below.
            slip = self.slippage_bps / 10_000
            px = mid * (1 + slip) if order.side == "BUY" else mid * (1 - slip)
            notional = px * order.qty
            fee = notional * (self.fee_bps / 10_000)
            signed_qty = order.qty if order.side == "BUY" else -order.qty

            positions = self._state["positions"]
            pos = positions.get(order.symbol)

            if pos is None:
                # Fresh position. Cash flow depends on direction.
                positions[order.symbol] = {
                    "qty": signed_qty,
                    "entry_price": px,
                    "stop_price": order.stop_price,
                    "take_profit": order.take_profit,
                    "opened_at": datetime.now(UTC).isoformat(),
                    "skill_id": order.extra.get("skill_id", ""),
                    "thesis": order.extra.get("thesis", ""),
                }
                if signed_qty > 0:
                    self._state["cash_usd"] -= notional + fee   # long: pay
                else:
                    self._state["cash_usd"] += notional - fee   # short: receive
            else:
                old_qty = float(pos["qty"])
                same_side = (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0)
                new_qty = old_qty + signed_qty

                if same_side:
                    # Scale into the position — recompute weighted-average entry.
                    avg_entry = (
                        old_qty * float(pos["entry_price"]) + signed_qty * px
                    ) / new_qty
                    pos["entry_price"] = avg_entry
                    pos["qty"] = new_qty
                    # Update stop/TP if a new one was supplied (don't clobber on None).
                    if order.stop_price is not None:
                        pos["stop_price"] = order.stop_price
                    if order.take_profit is not None:
                        pos["take_profit"] = order.take_profit
                    if signed_qty > 0:
                        self._state["cash_usd"] -= notional + fee
                    else:
                        self._state["cash_usd"] += notional - fee
                else:
                    # Reducing or flipping. Quantity being closed against the existing side.
                    closed_qty = min(abs(signed_qty), abs(old_qty))
                    if old_qty > 0:                                  # closing a long via SELL
                        realized = (px - float(pos["entry_price"])) * closed_qty
                        self._state["cash_usd"] += closed_qty * px - fee * (closed_qty / order.qty)
                    else:                                            # closing a short via BUY
                        realized = (float(pos["entry_price"]) - px) * closed_qty
                        self._state["cash_usd"] -= closed_qty * px + fee * (closed_qty / order.qty)
                    self._state["history"].append(
                        {
                            "symbol": order.symbol,
                            "closed_at": datetime.now(UTC).isoformat(),
                            "qty_closed": float(closed_qty),
                            "pnl_usd": float(realized) - fee * (closed_qty / order.qty),
                            "skill_id": pos.get("skill_id", ""),
                            "thesis": pos.get("thesis", ""),
                        }
                    )
                    if abs(new_qty) < 1e-12:
                        positions.pop(order.symbol, None)
                    else:
                        # Residual flips to the opposite side. New basis = current px,
                        # stops reset (we will NOT keep the old stop on a flipped side).
                        pos["qty"] = new_qty
                        pos["entry_price"] = px
                        pos["stop_price"] = order.stop_price
                        pos["take_profit"] = order.take_profit
                        pos["opened_at"] = datetime.now(UTC).isoformat()
                        # Account for the flipped half's open-leg cash flow.
                        flipped_notional = abs(new_qty) * px
                        if new_qty > 0:
                            self._state["cash_usd"] -= flipped_notional
                        else:
                            self._state["cash_usd"] += flipped_notional

            fill = Fill(
                order_id=order.client_id or f"paper-{uuid4().hex[:8]}",
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=px,
                fee_usd=fee,
            )
            self._save()
        log.info(
            "paper_fill", symbol=order.symbol, side=order.side, qty=order.qty,
            price=px, fee=fee, cash=self._state["cash_usd"],
        )
        return fill

    async def close_position(self, symbol: str, reason: str) -> Fill | None:
        pos = self._state["positions"].get(symbol)
        if not pos:
            return None
        side = "SELL" if pos["qty"] > 0 else "BUY"
        order = Order(symbol=symbol, side=side, qty=abs(float(pos["qty"])), extra={"reason": reason})
        return await self.place_order(order)

    async def history(self) -> list[dict]:
        return list(self._state["history"])

    async def close(self) -> None:
        await self.market.close()
