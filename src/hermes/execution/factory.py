"""Pick the right exchange adapter based on stage + config."""

from __future__ import annotations

from functools import lru_cache

from hermes.execution.base import Exchange
from hermes.execution.paper import PaperExchange
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


def select_fees(
    fees_cfg: dict, instrument: str, maker: bool = False
) -> tuple[float | None, float | None]:
    """Pick (fee_bps, slippage_bps) for the stage's instrument. ``maker`` selects
    the (lower) maker fee — used when strategy.maker_orders posts limit/post-only
    orders. Returns None where the config is silent so PaperExchange falls back to
    its own conservative default."""
    side = "maker" if maker else "taker"
    fee_key = f"{'perp' if instrument == 'perp' else 'spot'}_{side}"
    fee = fees_cfg.get(fee_key)
    slip = fees_cfg.get("slippage_bps")
    return (
        float(fee) if fee is not None else None,
        float(slip) if slip is not None else None,
    )


@lru_cache(maxsize=1)
def get_exchange() -> Exchange:
    s = get_settings()
    stage_cfg = s.stage_config
    venue = stage_cfg.get("venue", "paper")

    if venue == "paper":
        # Maker fees apply when the selective strategy posts limit/post-only orders.
        # In paper this is an OPTIMISTIC assumption (fills aren't guaranteed) — the
        # real maker-fill rate is only measurable on testnet/live.
        strat = s.yaml("strategy", default={}) or {}
        maker = bool(strat.get("selective_mode", False) and strat.get("maker_orders", False))
        fee_bps, slippage_bps = select_fees(
            s.yaml("fees", default={}) or {}, stage_cfg.get("instruments", "spot"), maker=maker
        )
        return PaperExchange(
            state_path=s.data_dir / "paper_state.json",
            starting_equity=s.bankroll_usd,
            venue_for_prices=s.paper_price_venue,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
    if venue == "hyperliquid":
        from hermes.execution.hyperliquid import HyperliquidExchange
        return HyperliquidExchange()
    if venue == "bybit":
        from hermes.execution.bybit import BybitExchange
        return BybitExchange()
    if venue == "binanceusdm":
        from hermes.execution.binance import BinanceUSDMExchange
        return BinanceUSDMExchange()
    raise RuntimeError(f"unknown venue: {venue}")
