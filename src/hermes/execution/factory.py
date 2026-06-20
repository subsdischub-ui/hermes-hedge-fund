"""Pick the right exchange adapter based on stage + config."""

from __future__ import annotations

from functools import lru_cache

from hermes.execution.base import Exchange
from hermes.execution.paper import PaperExchange
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


def select_fees(
    fees_cfg: dict, instrument: str
) -> tuple[float | None, float | None]:
    """Pick (taker_bps, slippage_bps) for the stage's instrument. Returns None
    where the config is silent so PaperExchange uses its own conservative default."""
    taker_key = "perp_taker" if instrument == "perp" else "spot_taker"
    fee = fees_cfg.get(taker_key)
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
        fee_bps, slippage_bps = select_fees(
            s.yaml("fees", default={}) or {}, stage_cfg.get("instruments", "spot")
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
    raise RuntimeError(f"unknown venue: {venue}")
