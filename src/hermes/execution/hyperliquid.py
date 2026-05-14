"""Hyperliquid spot/perp adapter (production path).

Wraps `hyperliquid-python-sdk`. Falls back to a NotImplementedError when no
credentials are configured — the agent will refuse to leave paper mode in
that case.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


class HyperliquidExchange(Exchange):
    name = "hyperliquid"

    def __init__(self) -> None:
        s = get_settings()
        if not s.hl_account_address or not s.hl_secret_key:
            raise RuntimeError("Hyperliquid credentials missing — refusing live mode.")
        from hyperliquid.info import Info  # type: ignore
        from hyperliquid.exchange import Exchange as HLExchange  # type: ignore
        from eth_account import Account  # type: ignore

        base_url = "https://api.hyperliquid-testnet.xyz" if s.hl_testnet else "https://api.hyperliquid.xyz"
        wallet = Account.from_key(s.hl_secret_key)
        self.info = Info(base_url, skip_ws=True)
        self.exchange = HLExchange(wallet, base_url, account_address=s.hl_account_address)
        self.address = s.hl_account_address

    async def equity_usd(self) -> float:
        state = self.info.user_state(self.address)
        return float(state.get("marginSummary", {}).get("accountValue", 0.0))

    async def open_positions(self) -> list[Position]:
        state = self.info.user_state(self.address)
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
                    opened_at=datetime.now(timezone.utc),
                    unrealized_pnl_usd=float(p.get("unrealizedPnl", 0.0)),
                )
            )
        return out

    async def last_price(self, symbol: str) -> float:
        coin = symbol.split("/")[0]
        mids = self.info.all_mids()
        return float(mids[coin])

    async def place_order(self, order: Order) -> Fill:
        coin = order.symbol.split("/")[0]
        is_buy = order.side == "BUY"
        result = self.exchange.market_open(
            name=coin, is_buy=is_buy, sz=order.qty, slippage=0.01
        )
        log.info("hl_order", result=result)
        # SDK returns {'status': 'ok', 'response': {'data': {'statuses': [{'filled': {...}}]}}}
        filled = (
            result.get("response", {})
            .get("data", {})
            .get("statuses", [{}])[0]
            .get("filled", {})
        )
        return Fill(
            order_id=str(filled.get("oid", order.client_id)),
            symbol=order.symbol,
            side=order.side,
            qty=float(filled.get("totalSz", order.qty)),
            price=float(filled.get("avgPx", 0.0)),
            fee_usd=float(filled.get("fee", 0.0)),
        )

    async def close_position(self, symbol: str, reason: str) -> Fill | None:
        coin = symbol.split("/")[0]
        result = self.exchange.market_close(coin)
        log.info("hl_close", symbol=symbol, reason=reason, result=result)
        filled = (
            result.get("response", {})
            .get("data", {})
            .get("statuses", [{}])[0]
            .get("filled", {})
        )
        if not filled:
            return None
        return Fill(
            order_id=str(filled.get("oid", "")),
            symbol=symbol,
            side="SELL",
            qty=float(filled.get("totalSz", 0)),
            price=float(filled.get("avgPx", 0.0)),
            fee_usd=float(filled.get("fee", 0.0)),
        )

    async def close(self) -> None:
        return None
