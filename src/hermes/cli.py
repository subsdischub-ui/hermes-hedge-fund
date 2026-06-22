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
            n = await run_nightly_reflection()
            typer.echo(f"lessons_added={n}")
        elif loop == "weekly":
            out = await run_weekly_skill_rerank()
            typer.echo(out)
        elif loop == "monthly":
            path = await run_monthly_compile()
            typer.echo(f"trainset={path}")
        else:
            raise typer.BadParameter("loop must be nightly|weekly|monthly")
    asyncio.run(_run())


@app.command(name="fetch-history")
def fetch_history(
    timeframe: str = typer.Option("1h", help="bar timeframe, e.g. 1h or 4h"),
    years: float = typer.Option(2.0, help="exchange mode: how much history to assemble"),
    max_batches: int = typer.Option(400, help="exchange mode: pagination safety cap"),
    source: str = typer.Option(
        "hf", help="hf (linxy/CryptoCoin deep history) | exchange (live CCXT, shallow)"
    ),
) -> None:
    """Assemble multi-year OHLCV per symbol and save it under
    ${HERMES_DATA_DIR}/history/<SYM>_<tf>.csv for offline backtests.

    Default source `hf` pulls the linxy/CryptoCoin dataset (Binance klines back to
    2017 — a full multi-cycle history). `exchange` paginates the live venue
    (Kraken caps OHLC depth, so it is shallow)."""

    s = get_settings()
    tf_ms = {"1h": 3_600_000, "4h": 14_400_000}.get(timeframe, 3_600_000)
    out_dir = s.data_dir / "history"

    if source == "hf":
        configure_logging("INFO")
        from hermes.data.history import detect_gaps, fetch_hf_history

        for sym in s.universe:
            path = out_dir / f"{sym.replace('/', '_')}_{timeframe}.csv"
            df = fetch_hf_history(sym, timeframe, dest_csv=path)
            gaps = detect_gaps(df, tf_ms)
            span_d = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000 if len(df) else 0
            typer.echo(f"{sym}: {len(df)} bars (~{span_d:.0f}d), {len(gaps)} gaps -> {path}")
        return

    async def _run() -> None:
        configure_logging("INFO")
        import time

        from hermes.data import get_market
        from hermes.data.history import detect_gaps, save_history
        from hermes.data.history import fetch_history as fh

        since_ms = int(time.time() * 1000) - int(years * 365 * 24 * 3_600_000)
        md = get_market(s.paper_price_venue)
        for sym in s.universe:
            df = await fh(sym, timeframe, since_ms, md.client, max_batches=max_batches)
            gaps = detect_gaps(df, tf_ms)
            path = out_dir / f"{sym.replace('/', '_')}_{timeframe}.csv"
            save_history(df, path)
            typer.echo(f"{sym}: {len(df)} bars, {len(gaps)} gaps -> {path}")
        await md.close()

    asyncio.run(_run())


@app.command()
def backtest(
    limit: int = typer.Option(1000, help="bars of 1h history to fetch per symbol (live mode)"),
    horizon: int = typer.Option(24, help="triple-barrier horizon in bars"),
    step: int = typer.Option(1, help="evaluate every Nth bar (speeds up long histories)"),
    timeframe: str = typer.Option("1h", help="bar timeframe"),
    window_cap: int = typer.Option(
        400, help="trailing bars fed to skills per bar (>=200 lookback; O(n^2)->O(n))"
    ),
    history_dir: str = typer.Option(
        "", help="load multi-year CSVs from this dir instead of a live fetch"
    ),
    bootstrap: bool = typer.Option(
        False, help="persist the fitted calibrator + meta-labeler to /data/models"
    ),
) -> None:
    """Replay the engine on historical OHLCV and report Deflated Sharpe / CPCV /
    PBO / win-rate net of fees; optionally bootstrap the live decision models.

    With --history-dir, loads the long history assembled by `hermes fetch-history`
    (so the DSR/PBO verdict spans a full cycle, not just the last ~1000 bars)."""

    async def _run() -> None:
        configure_logging("INFO")
        from pathlib import Path

        import pandas as pd

        from hermes.data import get_market
        from hermes.data.history import load_history
        from hermes.skills import get_skill_library
        from hermes.validation.backtest import run_backtest
        from hermes.validation.cpcv import TrialRegistry
        from hermes.validation.report import backtest_report

        s = get_settings()
        fees = s.yaml("fees", default={}) or {}
        fee_bps = float(fees.get("spot_taker", 7))
        slip = float(fees.get("slippage_bps", 3))

        ohlcv = {}
        if history_dir:
            base = Path(history_dir)
            for sym in s.universe:
                path = base / f"{sym.replace('/', '_')}_{timeframe}.csv"
                if not path.exists():
                    typer.echo(f"skip {sym}: {path} not found")
                    continue
                df = load_history(path)
                # History stores ts as epoch-ms ints; convert so time-of-day skills
                # read real UTC hours (the live path already returns datetime ts).
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                ohlcv[sym] = df
                typer.echo(f"loaded {sym}: {len(df)} bars from {path}")
        else:
            md = get_market(s.paper_price_venue)
            for sym in s.universe:
                df = await md.ohlcv(sym, timeframe=timeframe, limit=limit)
                ohlcv[sym] = df
                typer.echo(f"fetched {sym}: {len(df)} bars")
            await md.close()

        res = run_backtest(
            ohlcv, fee_bps=fee_bps, slippage_bps=slip, horizon=horizon, step=step,
            window_cap=window_cap,
        )

        # True trial count: each active skill is one frozen config tried this run;
        # the registry accumulates across runs so the DSR deflates honestly.
        lib = get_skill_library()
        n_configs = len(lib.list_active()) + len(lib.list_active_cross())
        registry = TrialRegistry(s.data_dir)
        total_trials = registry.bump(n_configs)

        rep = backtest_report(res, n_trials=total_trials)
        o = rep["overall"]
        c = o["cpcv"]
        typer.echo("")
        typer.echo(f"=== Backtest ({o['n']} trades, fees {fee_bps}+{slip}bps, horizon {horizon}) ===")
        typer.echo(
            f"OVERALL  win_rate={o['win_rate']:.3f}  mean_ret={o['mean_ret']:+.4f}  "
            f"DSR={o['dsr']:.3f}  PBO={o['pbo']:.3f}  (n_trials={o['n_trials']})"
        )
        typer.echo(
            f"CPCV     paths={c['n_paths']}  sharpe_mean={c['sharpe_mean']:+.3f}  "
            f"median={c['sharpe_median']:+.3f}  frac_positive={c['frac_positive']:.2f}"
        )
        for sid, st in rep["per_skill"].items():
            typer.echo(
                f"  {sid:26s} n={st['n']:4d}  win={st['win_rate']:.3f}  "
                f"mean={st['mean_ret']:+.4f}  DSR={st['dsr']:.3f}"
            )
        if bootstrap:
            from hermes.validation.bootstrap import bootstrap_models

            typer.echo(f"\nbootstrap: {bootstrap_models(res, s.data_dir / 'models')}")

    asyncio.run(_run())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("hermes.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
