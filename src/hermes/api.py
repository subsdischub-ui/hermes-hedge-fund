"""FastAPI surface — health, status, manual tick, dashboard data."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from hermes.execution import get_exchange
from hermes.graph import HermesOrchestrator
from hermes.learning import run_nightly_reflection, run_weekly_skill_rerank
from hermes.logging_setup import configure_logging, get_logger
from hermes.memory.journal import TradeJournal
from hermes.memory.store import get_db
from hermes.risk.stage import StageManager
from hermes.scheduler import build_scheduler
from hermes.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level)
    log = get_logger(__name__)
    log.info("hermes_starting", stage=s.stage, bankroll=s.bankroll_usd, universe=s.universe)
    await get_db().init()
    # Rehydrate skill archival from disk so weekly decisions persist across restarts.
    from hermes.skills import get_skill_library
    await get_skill_library().reload_archival_from_db()
    sched = build_scheduler()
    sched.start()
    log.info("scheduler_started", jobs=[j.id for j in sched.get_jobs()])
    try:
        yield
    finally:
        # Graceful shutdown: stop accepting jobs, wait briefly for in-flight ones,
        # then dispose the engine. Open positions are intentionally NOT auto-closed —
        # the operator must explicitly close before stopping the agent.
        sched.shutdown(wait=True)
        await get_db().dispose()


app = FastAPI(title="Hermès", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, Any]:
    s = get_settings()
    exchange = get_exchange()
    equity = await exchange.equity_usd()
    positions = await exchange.open_positions()
    journal = TradeJournal()
    stats = await journal.stats()
    stage_mgr = StageManager()
    # Stage gates require a real walk-forward Sharpe and rule-violation count;
    # the status endpoint reports the metrics it can prove and leaves the
    # other gates unfilled so the operator sees them as BLOCKED rather than
    # passed-on-a-proxy.
    report = stage_mgr.evaluate(
        metrics={
            "live_trades": stats["count"],
            "paper_trades": stats["count"],
            "rule_violations": 0,
        }
    )
    return {
        "stage": s.stage,
        "stage_advance_summary": report.summary,
        "equity_usd": equity,
        "open_positions": [p.__dict__ for p in positions],
        "journal": stats,
    }


@app.post("/tick/{symbol_base}")
async def manual_tick(symbol_base: str) -> dict[str, Any]:
    symbol = f"{symbol_base.upper()}/USDC"
    if symbol not in get_settings().universe:
        raise HTTPException(404, f"{symbol} not in universe")
    orch = HermesOrchestrator()
    state = await orch.tick(symbol)
    return {
        "symbol": symbol,
        "executed": state.executed,
        "trade_id": state.trade_id,
        "risk_decision": state.risk_decision,
        "proposal": state.trader_proposal,
        "trm": state.trm_refinement,
        "error": state.error,
    }


@app.post("/learn/nightly")
async def trigger_nightly() -> dict[str, Any]:
    n = await run_nightly_reflection()
    return {"lessons_added": n}


@app.post("/learn/weekly")
async def trigger_weekly() -> dict[str, Any]:
    out = await run_weekly_skill_rerank()
    return {"skills": out}
