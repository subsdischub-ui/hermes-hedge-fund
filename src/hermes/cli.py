"""`hermes` CLI — for local dev + ops."""

from __future__ import annotations

import asyncio

import typer

from hermes.execution import get_exchange
from hermes.graph import HermesOrchestrator
from hermes.learning import (
    run_monthly_compile,
    run_nightly_reflection,
    run_weekly_skill_rerank,
)
from hermes.logging_setup import configure_logging
from hermes.memory.journal import TradeJournal
from hermes.memory.store import get_db
from hermes.risk.stage import StageManager
from hermes.settings import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Hermès — autonomous AI hedge-fund agent.")


@app.command()
def status() -> None:
    async def _run() -> None:
        configure_logging("INFO")
        await get_db().init()
        ex = get_exchange()
        eq = await ex.equity_usd()
        pos = await ex.open_positions()
        stats = await TradeJournal().stats()
        s = get_settings()
        typer.echo(f"stage          : {s.stage}")
        typer.echo(f"equity_usd     : ${eq:.2f}")
        typer.echo(f"open_positions : {len(pos)}")
        typer.echo(f"closed_trades  : {stats['count']}")
        typer.echo(f"win_rate       : {stats['win_rate']}")
    asyncio.run(_run())


@app.command()
def tick(symbol: str = typer.Argument(..., help="e.g. BTC/USDC"), timeframe: str = "1h") -> None:
    async def _run() -> None:
        configure_logging("INFO")
        await get_db().init()
        orch = HermesOrchestrator()
        state = await orch.tick(symbol, timeframe=timeframe)
        typer.echo(f"executed={state.executed} trade_id={state.trade_id}")
        typer.echo(f"proposal={state.trader_proposal}")
        typer.echo(f"risk={state.risk_decision.get('decision')} ({state.risk_decision.get('reason')})")
        if state.trm_refinement:
            typer.echo(f"trm={state.trm_refinement}")
    asyncio.run(_run())


@app.command()
def stage() -> None:
    async def _run() -> None:
        await get_db().init()
        stats = await TradeJournal().stats()
        mgr = StageManager()
        rep = mgr.evaluate({
            "live_trades": stats["count"],
            "paper_trades": stats["count"],
            "rule_violations": 0,
        })
        typer.echo(rep.summary)
    asyncio.run(_run())


@app.command()
def learn(loop: str = typer.Argument("nightly")) -> None:
    async def _run() -> None:
        configure_logging("INFO")
        await get_db().init()
        if loop == "nightly":
            n = await run_nightly_reflection(); typer.echo(f"lessons_added={n}")
        elif loop == "weekly":
            out = await run_weekly_skill_rerank(); typer.echo(out)
        elif loop == "monthly":
            path = await run_monthly_compile(); typer.echo(f"trainset={path}")
        else:
            raise typer.BadParameter("loop must be nightly|weekly|monthly")
    asyncio.run(_run())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("hermes.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
