"""Pick the right exchange adapter based on stage + config."""

from __future__ import annotations

from functools import lru_cache

from hermes.execution.base import Exchange
from hermes.execution.paper import PaperExchange
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_exchange() -> Exchange:
    s = get_settings()
    stage_cfg = s.stage_config
    venue = stage_cfg.get("venue", "paper")

    if venue == "paper":
        return PaperExchange(
            state_path=s.data_dir / "paper_state.json",
            starting_equity=s.bankroll_usd,
            venue_for_prices=s.paper_price_venue,
        )
    if venue == "hyperliquid":
        from hermes.execution.hyperliquid import HyperliquidExchange
        return HyperliquidExchange()
    if venue == "bybit":
        from hermes.execution.bybit import BybitExchange
        return BybitExchange()
    raise RuntimeError(f"unknown venue: {venue}")
