"""APScheduler — wires up market tick, stop watcher, and learning cadences."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from hermes.graph import HermesOrchestrator
from hermes.learning import (
    run_monthly_compile,
    run_nightly_reflection,
    run_weekly_skill_rerank,
)
from hermes.logging_setup import get_logger
from hermes.runtime.activity import ActivityLog
from hermes.runtime.pulse import pulse
from hermes.runtime.stop_watcher import tick as stop_tick
from hermes.settings import get_settings

log = get_logger(__name__)

# La règle validée vit sur les clôtures 4h UTC de Binance (00/04/08/12/16/20).
# Le tick DOIT donc être ancré UTC, pas en HERMES_TZ : avec le cron applicatif
# "0 */4" en Europe/Paris (UTC+2), la prod a tiré à H+2 de chaque barre pendant
# 3 semaines — décision systématique sur une demi-barre jamais backtestée
# (audit 2026-07-16). Minute 1 : marge de consolidation de la clôture côté venue.
_MARKET_TICK_SPEC = "1 0,4,8,12,16,20 * * *"


def market_tick_trigger(spec: str = _MARKET_TICK_SPEC) -> CronTrigger:
    """Trigger du market_tick, ancré UTC quel que soit HERMES_TZ (DST-immune)."""
    return CronTrigger.from_crontab(spec, timezone="UTC")


def build_scheduler() -> AsyncIOScheduler:
    s = get_settings()
    loops = s.yaml("loops", default={}) or {}
    sched = AsyncIOScheduler(timezone=s.tz)

    orch = HermesOrchestrator()

    decision_tf = str(s.yaml("strategy", "decision_timeframe", default="4h") or "4h")
    activity = ActivityLog(s.data_dir / "activity.json")

    async def market_tick() -> None:
        for sym in s.universe:
            try:
                state = await orch.tick(sym, timeframe=decision_tf)
                rd = state.risk_decision or {}
                prop = state.trader_proposal or {}
                activity.record(
                    {
                        "kind": "decision",
                        "symbol": sym,
                        "executed": bool(state.executed),
                        "side": prop.get("side", "?"),
                        "decision": rd.get("decision"),
                        "reason": rd.get("reason") or (state.selective_gate or {}).get("reason"),
                        "skill": prop.get("skill_id", ""),
                        "regime": (state.regime or {}).get("regime"),
                        "selective_gate": state.selective_gate or {},
                    }
                )
            except Exception as e:
                log.exception("market_tick_failed", symbol=sym, error=str(e))
                activity.record({"kind": "error", "symbol": sym, "error": str(e)})

    async def status_heartbeat() -> None:
        """Periodic one-line health/PnL summary so the agent's rhythm is visible
        in the logs and any external watchdog can scrape it."""
        try:
            from hermes.execution import get_exchange
            from hermes.memory.journal import TradeJournal

            ex = get_exchange()
            eq = await ex.equity_usd()
            pos = await ex.open_positions()
            st = await TradeJournal().stats()
            log.info(
                "heartbeat",
                equity_usd=round(eq, 2),
                open_positions=len(pos),
                closed_trades=st["count"],
                win_rate=st["win_rate"],
                realized_pnl_usd=round(st["pnl_usd"], 2),
            )
        except Exception as e:
            log.warning("heartbeat_failed", error=str(e))

    sched.add_job(
        market_tick,
        # UTC ancré aux clôtures 4h Binance — jamais s.tz ici (voir market_tick_trigger).
        market_tick_trigger(loops.get("market_tick", _MARKET_TICK_SPEC)),
        id="market_tick",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        stop_tick,
        IntervalTrigger(seconds=30),
        id="stop_watcher",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        status_heartbeat,
        CronTrigger.from_crontab(loops.get("heartbeat", "0 * * * *"), timezone=s.tz),
        id="status_heartbeat",
        max_instances=1,
        coalesce=True,
    )
    # Hourly LLM-free pulse: records what the agent sees/would-do so the dashboard
    # shows continuous activity between the 4h decisions.
    sched.add_job(
        pulse,
        CronTrigger.from_crontab(loops.get("pulse", "0 * * * *"), timezone=s.tz),
        id="pulse",
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        run_nightly_reflection,
        CronTrigger.from_crontab(loops.get("nightly_reflect", "0 23 * * *"), timezone=s.tz),
        id="nightly_reflection",
    )
    sched.add_job(
        run_weekly_skill_rerank,
        CronTrigger.from_crontab(loops.get("weekly_rerank", "0 22 * * 0"), timezone=s.tz),
        id="weekly_rerank",
    )
    sched.add_job(
        run_monthly_compile,
        CronTrigger.from_crontab(loops.get("monthly_compile", "0 21 1 * *"), timezone=s.tz),
        id="monthly_compile",
    )
    return sched
