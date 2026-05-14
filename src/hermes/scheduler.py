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
from hermes.runtime.stop_watcher import tick as stop_tick
from hermes.settings import get_settings

log = get_logger(__name__)


def build_scheduler() -> AsyncIOScheduler:
    s = get_settings()
    loops = s.yaml("loops", default={}) or {}
    sched = AsyncIOScheduler(timezone=s.tz)

    orch = HermesOrchestrator()

    async def market_tick() -> None:
        for sym in s.universe:
            try:
                await orch.tick(sym, timeframe="1h")
            except Exception as e:
                log.exception("market_tick_failed", symbol=sym, error=str(e))

    sched.add_job(
        market_tick,
        CronTrigger.from_crontab(loops.get("market_tick", "*/5 * * * *"), timezone=s.tz),
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
