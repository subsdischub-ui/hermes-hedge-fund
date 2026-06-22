"""Hyperliquid spot/perp adapter (production path).

Wraps `hyperliquid-python-sdk`. The SDK is synchronous, so every call is
dispatched via `asyncio.to_thread` to avoid blocking the event loop. Falls
back to a NotImplementedError when no credentials are configured — the
agent will refuse to leave paper mode in that case.

Response handling
-----------------
HL responses look like:
    {'status': 'ok', 'response': {'data': {'statuses': [{'filled': {...}}]}}}
or
    {'status': 'err', 'response': '...message...'}
We treat anything other than a fully-realized `filled` block as an error and
raise; callers must not interpret zero-fill as "executed".
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


class HyperliquidError(RuntimeError):
    pass


# --- pure helpers (unit-tested without the SDK) -----------------------------
def derive_cloid(client_id: str) -> str:
    """Deterministic 128-bit client order id (0x + 32 hex) from our client_id, so
    a network retry of the same logical order reuses the cloid and the exchange
    deduplicates it instead of opening a second position."""
    return "0x" + hashlib.sha256(str(client_id).encode()).hexdigest()[:32]


def build_tpsl_orders(
    coin: str, is_long: bool, qty: float, stop_px: float | None, tp_px: float | None
) -> list[dict[str, Any]]:
    """Reduce-only stop-loss / take-profit TRIGGER orders that close the position
    on the exchange (so a live position is protected even if the agent dies). A
    long is closed by SELLs, a short by BUYs. Skips legs with no price."""
    close_is_buy = not is_long
    out: list[dict[str, Any]] = []
    for px, tpsl in ((stop_px, "sl"), (tp_px, "tp")):
        if px is None:
            continue
        out.append(
            {
                "coin": coin,
                "is_buy": close_is_buy,
                "sz": float(qty),
                "limit_px": float(px),
                "reduce_only": True,
                "order_type": {"trigger": {"triggerPx": float(px), "isMarket": True, "tpsl": tpsl}},
            }
        )
    return out


class HyperliquidExchange(Exchange):
    name = "hyperliquid"
    DEFAULT_SLIPPAGE = 0.005   # 0.5% — tight enough to avoid bad fills on majors

    def __init__(self) -> None:
        s = get_settings()
        secret = s.hl_secret_key.get_secret_value() if hasattr(s.hl_secret_key, "get_secret_value") else str(s.hl_secret_key)
        if not s.hl_account_address or not secret:
            raise RuntimeError("Hyperliquid credentials missing — refusing live mode.")
        from eth_account import Account  # type: ignore
        from hyperliquid.exchange import Exchange as HLExchange  # type: ignore
        from hyperliquid.info import Info  # type: ignore

        base_url = "https://api.hyperliquid-testnet.xyz" if s.hl_testnet else "https://api.hyperliquid.xyz"
        wallet = Account.from_key(secret)
        self.info = Info(base_url, skip_ws=True)
        self.exchange = HLExchange(wallet, base_url, account_address=s.hl_account_address)
        self.address = s.hl_account_address
        # Stage leverage (1x at micro/normal -> perps behave economically like spot).
        self._leverage = int((s.stage_config or {}).get("leverage_initial", 1))

    # --- sync->async wrapper ------------------------------------------------
    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    # --- read paths ---------------------------------------------------------
    async def equity_usd(self) -> float:
        state = await self._to_thread(self.info.user_state, self.address)
        return float(state.get("marginSummary", {}).get("accountValue", 0.0))

    async def open_positions(self) -> list[Position]:
        state = await self._to_thread(self.info.user_state, self.address)
        out: list[Position] = []
        for asset in state.get("assetPositions", []):
            p = asset["position"]
            qty = float(p["szi"])
            if abs(qty) < 1e-12:
                continue
            out.append(
                Position(
                    symbol=p["coin"] + "/USDC",
                    qty=qty,
                    entry_price=float(p["entryPx"]),
                    stop_price=None,
                    take_profit=None,
                    opened_at=datetime.now(UTC),
                    unrealized_pnl_usd=float(p.get("unrealizedPnl", 0.0)),
                )
            )
        return out

    async def last_price(self, symbol: str) -> float:
        coin = symbol.split("/")[0]
        mids = await self._to_thread(self.info.all_mids)
        if coin not in mids:
            raise HyperliquidError(f"no mid for {coin}")
        return float(mids[coin])

    # --- write paths --------------------------------------------------------
    @staticmethod
    def _extract_filled(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "ok":
            raise HyperliquidError(f"hl error: {result!r}")
        statuses = (
            result.get("response", {}).get("data", {}).get("statuses", [])
        )
        if not statuses:
            raise HyperliquidError(f"hl no statuses: {result!r}")
        st = statuses[0]
        if "error" in st:
            raise HyperliquidError(f"hl rejected: {st['error']}")
        filled = st.get("filled")
        if not filled or float(filled.get("totalSz", 0)) <= 0:
            raise HyperliquidError(f"hl zero fill: {st!r}")
        return filled

    def _cloid(self, order: Order):
        """Best-effort Cloid object from our deterministic id; None if the SDK
        version doesn't expose Cloid (we then fall back to no idempotency key)."""
        hexid = derive_cloid(order.client_id or f"{order.symbol}-{order.qty}-{order.side}")
        try:
            from hyperliquid.utils.types import Cloid  # type: ignore

            return Cloid.from_str(hexid)
        except Exception:
            try:
                from hyperliquid.utils.signing import Cloid  # type: ignore

                return Cloid.from_str(hexid)
            except Exception:
                log.warning("hl_cloid_unavailable")
                return None

    async def place_order(self, order: Order) -> Fill:
        coin = order.symbol.split("/")[0]
        is_buy = order.side == "BUY"
        # Enforce stage leverage (1x = no leverage) before opening.
        try:
            await self._to_thread(self.exchange.update_leverage, self._leverage, coin)
        except Exception as e:
            log.warning("hl_update_leverage_failed", coin=coin, error=str(e))

        cloid = self._cloid(order)
        result = await self._to_thread(
            self.exchange.market_open,
            coin, is_buy, order.qty, None, self.DEFAULT_SLIPPAGE, cloid,
        )
        # Redact the SDK echo down to the bits we need before logging.
        log.info("hl_order_status", status=result.get("status"), symbol=order.symbol, side=order.side)
        filled = self._extract_filled(result)
        fill = Fill(
            order_id=str(filled.get("oid", order.client_id)),
            symbol=order.symbol,
            side=order.side,
            qty=float(filled["totalSz"]),
            price=float(filled.get("avgPx", 0.0)),
            fee_usd=float(filled.get("fee", 0.0)),
        )
        # Attach native reduce-only stop/TP triggers so the position is protected
        # on-exchange even if the agent, the Mac LLM or Tailscale goes down.
        tpsl = build_tpsl_orders(
            coin, is_long=is_buy, qty=float(filled["totalSz"]),
            stop_px=order.stop_price, tp_px=order.take_profit,
        )
        if tpsl:
            try:
                await self._to_thread(self.exchange.bulk_orders, tpsl)
            except Exception as e:
                log.warning("hl_tpsl_attach_failed", symbol=order.symbol, error=str(e))
        return fill

    async def close_position(
        self, symbol: str, reason: str, skill_id: str | None = None
    ) -> Fill | None:
        # skill_id is accepted for interface parity with the paper book (which holds
        # one position per (symbol, skill)); a real perp venue nets to ONE position
        # per symbol, so the whole symbol is closed regardless of skill_id.
        coin = symbol.split("/")[0]
        result = await self._to_thread(self.exchange.market_close, coin)
        log.info("hl_close_status", status=result.get("status"), symbol=symbol, reason=reason)
        try:
            filled = self._extract_filled(result)
        except HyperliquidError as e:
            log.warning("hl_close_failed", symbol=symbol, error=str(e))
            return None
        return Fill(
            order_id=str(filled.get("oid", "")),
            symbol=symbol,
            side="SELL",
            qty=float(filled["totalSz"]),
            price=float(filled.get("avgPx", 0.0)),
            fee_usd=float(filled.get("fee", 0.0)),
        )

    async def close(self) -> None:
        return None
