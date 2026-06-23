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
            self._migrate_position_keys()
        else:
            self._state = {
                "cash_usd": float(starting_equity),
                # key = "{symbol}::{skill_id}" so two strategies (e.g. momentum #1
                # and funding #2) can each hold ONE position on the same symbol
                # concurrently. Each value also stores "symbol" and "skill_id".
                "positions": {},
                # Strategy #3 delta-neutral carry PAIRS (short perp + long spot),
                # keyed by symbol; price-neutral, accrue funding over the hold.
                "carry_positions": {},
                "history": [],
            }
            self._save()
        self._lock = asyncio.Lock()

    # ---- position-book key (symbol x skill) --------------------------------
    @staticmethod
    def _pos_key(symbol: str, skill_id: str | None) -> str:
        return f"{symbol}::{skill_id or ''}"

    def _migrate_position_keys(self) -> None:
        """Upgrade a legacy symbol-keyed book to the (symbol, skill) composite key.
        Idempotent: already-composite keys pass through unchanged."""
        positions = self._state.get("positions", {})
        migrated: dict[str, dict] = {}
        changed = False
        for k, p in positions.items():
            if "::" in k:
                sym, sid = k.split("::", 1)
            else:
                sym, sid = k, p.get("skill_id", "")
                changed = True
            p["symbol"] = p.get("symbol", sym)
            p["skill_id"] = p.get("skill_id", sid)
            migrated[self._pos_key(p["symbol"], p["skill_id"])] = p
        self._state["positions"] = migrated
        if changed:
            self._save()

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
        for p in self._state["positions"].values():
            sym = p["symbol"]
            px = await self._safe_last_price(sym, fallback=float(p["entry_price"]))
            # equity = cash + Σ (signed_qty * mark_price), for BOTH longs and shorts:
            # open debits cash by the long notional / credits it by the short notional,
            # so adding signed_qty*px reconstructs C0 + unrealized PnL in one formula.
            #   long  (qty>0): C0 - |q|*entry + |q|*px = C0 + |q|*(px-entry)  ✓
            #   short (qty<0): C0 + |q|*entry - |q|*px = C0 + |q|*(entry-px)  ✓
            # (The previous short branch added qty*(px-entry), double-counting the
            #  cash credit and inflating equity by the open short's notional.)
            total += float(p["qty"]) * px
        # Delta-neutral carry pairs: price legs cancel, so only the accrued funding
        # (carry collected so far) contributes to equity.
        for cp in self._state.get("carry_positions", {}).values():
            total += float(cp.get("accrued_funding_usd", 0.0))
        return total

    async def open_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._state["positions"].values():
            sym = p["symbol"]
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
        # Carry pairs surface as a delta-neutral (qty=0) position whose unrealized
        # PnL is the funding accrued so far — so the dashboard shows #3 working.
        for cp in self._state.get("carry_positions", {}).values():
            opened_at = cp["opened_at"]
            if isinstance(opened_at, str):
                opened_at = datetime.fromisoformat(opened_at)
            out.append(
                Position(
                    symbol=cp["symbol"], qty=0.0, entry_price=float(cp["entry_price"]),
                    stop_price=None, take_profit=None, opened_at=opened_at,
                    unrealized_pnl_usd=float(cp.get("accrued_funding_usd", 0.0)),
                    skill_id=cp.get("skill_id", "funding_carry_harvest"),
                    thesis=cp.get("thesis", "delta-neutral carry"),
                )
            )
        return out

    # ---- delta-neutral carry (Strategy #3) ----------------------------------
    CARRY_SKILL = "funding_carry_harvest"
    CARRY_MAKER_BPS = 5.0  # per fill; 4 fills/round-trip pair = 20bps = the validated maker cost

    async def _open_carry(self, order: Order) -> Fill:
        """Open a delta-neutral carry PAIR: SHORT perp + LONG spot, equal notional,
        MAKER fees both legs. Price legs net to ~0 delta; only fees touch cash now."""
        async with self._lock:
            mid = order.price or await self.last_price(order.symbol)
            qty = float(order.qty)
            notional = mid * qty
            fee = notional * (self.CARRY_MAKER_BPS / 10_000) * 2  # short + long maker legs
            self._state["cash_usd"] -= fee
            self._state.setdefault("carry_positions", {})[order.symbol] = {
                "symbol": order.symbol, "qty": qty, "entry_price": mid,
                "opened_at": datetime.now(UTC).isoformat(),
                "accrued_funding_usd": 0.0,
                "entry_funding_bps": float(order.extra.get("funding_bps", 0.0)),
                "skill_id": self.CARRY_SKILL,
                "thesis": order.extra.get("thesis", "delta-neutral funding carry"),
            }
            self._save()
        log.info("paper_carry_open", symbol=order.symbol, qty=qty, notional=notional, fee=fee)
        return Fill(order_id=order.client_id or f"carry-{uuid4().hex[:8]}",
                    symbol=order.symbol, side="SELL", qty=qty, price=mid, fee_usd=fee)

    def settle_carry_funding(self, symbol: str, funding_bps: float) -> float:
        """Credit one 8h funding settlement to an open carry pair (short receives
        positive funding). Returns the USD credited. The live 8h loop / the close
        path drives this; pure + unit-testable."""
        cp = self._state.get("carry_positions", {}).get(symbol)
        if not cp:
            return 0.0
        notional = float(cp["qty"]) * float(cp["entry_price"])
        usd = notional * (float(funding_bps) / 10_000.0)
        cp["accrued_funding_usd"] = float(cp.get("accrued_funding_usd", 0.0)) + usd
        self._save()
        return usd

    async def _close_carry(self, symbol: str, reason: str) -> Fill | None:
        """Close BOTH legs atomically; realized PnL = accrued funding - close fees
        (price deltas cancel). Books to history under skill funding_carry_harvest."""
        async with self._lock:
            cp = self._state.get("carry_positions", {}).get(symbol)
            if not cp:
                return None
            mid = await self._safe_last_price(symbol, fallback=float(cp["entry_price"]))
            notional = float(cp["qty"]) * mid
            fee = notional * (self.CARRY_MAKER_BPS / 10_000) * 2
            # If no explicit 8h settlements were credited (the live loop / tests drive
            # settle_carry_funding), book the ACTUAL funding that settled over the hold
            # from the exchange feed (short receives positive funding). Best-effort: on
            # any failure the accrued stays as-is (tests control it via settle_*).
            if float(cp.get("accrued_funding_usd", 0.0)) == 0.0:
                opened = cp["opened_at"]
                if isinstance(opened, str):
                    opened = datetime.fromisoformat(opened)
                entry_notional = float(cp["qty"]) * float(cp["entry_price"])
                try:
                    fdf = await self.market.funding_history(symbol)
                    if fdf is not None and not getattr(fdf, "empty", True):
                        m = (fdf["ts"] > opened) & (fdf["ts"] <= datetime.now(UTC))
                        cp["accrued_funding_usd"] = entry_notional * float(fdf.loc[m, "funding_rate"].sum())
                except Exception as e:
                    log.warning("carry_funding_accrual_failed", symbol=symbol, error=str(e))
            realized = float(cp.get("accrued_funding_usd", 0.0)) - fee
            self._state["cash_usd"] += realized
            self._state["history"].append({
                "symbol": symbol, "closed_at": datetime.now(UTC).isoformat(),
                "qty_closed": float(cp["qty"]), "pnl_usd": float(realized),
                "skill_id": self.CARRY_SKILL, "thesis": cp.get("thesis", ""),
                "pair_type": "carry", "reason": reason,
            })
            self._state["carry_positions"].pop(symbol, None)
            self._save()
        log.info("paper_carry_close", symbol=symbol, realized=realized, reason=reason)
        return Fill(order_id=f"carry-{uuid4().hex[:8]}", symbol=symbol, side="BUY",
                    qty=float(cp["qty"]), price=mid, fee_usd=fee)

    # ---- order entry --------------------------------------------------------
    async def place_order(self, order: Order) -> Fill:
        # Structural safety: the carry skill can ONLY open a hedged two-leg pair —
        # never a naked single-leg order (which is OOS-negative). Routed by skill_id.
        if order.extra.get("skill_id") == self.CARRY_SKILL:
            return await self._open_carry(order)
        async with self._lock:
            mid = order.price or await self.last_price(order.symbol)
            # Apply paper slippage: BUY pays slightly above mid, SELL receives below.
            slip = self.slippage_bps / 10_000
            px = mid * (1 + slip) if order.side == "BUY" else mid * (1 - slip)
            notional = px * order.qty
            fee = notional * (self.fee_bps / 10_000)
            signed_qty = order.qty if order.side == "BUY" else -order.qty

            positions = self._state["positions"]
            key = self._pos_key(order.symbol, order.extra.get("skill_id", ""))
            pos = positions.get(key)

            if pos is None:
                # Fresh position. Cash flow depends on direction.
                positions[key] = {
                    "symbol": order.symbol,
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
                        positions.pop(key, None)
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

    async def close_position(
        self, symbol: str, reason: str, skill_id: str | None = None
    ) -> Fill | None:
        # Carry pairs close as a delta-neutral unit (both legs) via the carry path.
        if (skill_id == self.CARRY_SKILL or skill_id is None) and \
                symbol in self._state.get("carry_positions", {}):
            return await self._close_carry(symbol, reason)
        positions = self._state["positions"]
        if skill_id is not None:
            # Close the SPECIFIC (symbol, skill) position (the stop watcher path).
            pos = positions.get(self._pos_key(symbol, skill_id))
            targets = [pos] if pos else []
        else:
            # No skill given -> close every position on this symbol (legacy/manual).
            targets = [p for p in positions.values() if p.get("symbol") == symbol]
        if not targets:
            return None
        last: Fill | None = None
        for pos in targets:
            side = "SELL" if pos["qty"] > 0 else "BUY"
            order = Order(
                symbol=symbol, side=side, qty=abs(float(pos["qty"])),
                # carry skill_id so place_order resolves the SAME composite key.
                extra={"reason": reason, "skill_id": pos.get("skill_id", "")},
            )
            last = await self.place_order(order)
        return last

    async def history(self) -> list[dict]:
        return list(self._state["history"])

    async def close(self) -> None:
        await self.market.close()
