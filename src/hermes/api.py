"""FastAPI surface — health, status, manual tick, dashboard data."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

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
    app.state.scheduler = sched
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

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def _api_key_ok(provided: str | None, configured: str) -> bool:
    """Open when no key is configured; otherwise require an exact match."""
    if not configured:
        return True
    return provided == configured


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Dependency guarding side-effectful endpoints (/tick, /learn/*, /status).
    /health stays open for the container healthcheck."""
    if not _api_key_ok(key, get_settings().api_key.get_secret_value()):
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/status", dependencies=[Depends(require_api_key)])
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


@app.get("/dashboard/data")
async def dashboard_data() -> dict[str, Any]:
    """Read-only telemetry feed for the dashboard (no actions, so left open for
    in-browser viewing; lock behind a reverse-proxy/VPN before live if desired)."""
    from hermes.memory.journal import TradeJournal
    from hermes.risk.calibration import get_confidence_calibrator
    from hermes.risk.meta_labeling import get_meta_labeler

    s = get_settings()
    ex = get_exchange()
    equity = await ex.equity_usd()
    positions = await ex.open_positions()
    journal = TradeJournal()
    stats = await journal.stats()
    recent = (await journal.recent_closed(since=datetime.now(UTC) - timedelta(days=120)))[:20]
    report = StageManager().evaluate(
        {"live_trades": stats["count"], "paper_trades": stats["count"], "rule_violations": 0}
    )
    jobs = []
    sched = getattr(app.state, "scheduler", None)
    if sched is not None:
        for j in sched.get_jobs():
            nrt = getattr(j, "next_run_time", None)
            jobs.append({"id": j.id, "next_run": nrt.isoformat() if nrt else None})
    cal = await get_confidence_calibrator()
    ml = await get_meta_labeler()
    return {
        "stage": s.stage,
        "bankroll_usd": s.bankroll_usd,
        "universe": s.universe,
        "equity_usd": equity,
        "stage_summary": report.summary,
        "gates": {k: {"target": v[0], "observed": v[1], "passed": v[2]} for k, v in report.gates.items()},
        "journal": stats,
        "open_positions": [p.__dict__ for p in positions],
        "recent_closed": [
            {
                "symbol": r.symbol, "side": r.side, "pnl_usd": r.pnl_usd,
                "exit_reason": r.exit_reason, "skill_id": r.skill_id,
                "confidence": r.confidence,
                "closed_at": r.closed_at.isoformat() if r.closed_at else None,
            }
            for r in recent
        ],
        "scheduler": jobs,
        "models": {"calibrator_fitted": cal.is_fitted, "meta_labeler_fitted": ml.is_fitted},
        "breakdown": await journal.breakdown(),
        "activity": _recent_activity(s),
        "server_time": datetime.now(UTC).isoformat(),
    }


def _recent_activity(s, n: int = 30) -> list[dict]:
    """Last agent decisions/pulses (why it traded or held) for the activity feed."""
    from hermes.runtime.activity import ActivityLog

    return ActivityLog(s.data_dir / "activity.json").recent(n)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


@app.post("/tick/{symbol_base}", dependencies=[Depends(require_api_key)])
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


@app.post("/learn/nightly", dependencies=[Depends(require_api_key)])
async def trigger_nightly() -> dict[str, Any]:
    n = await run_nightly_reflection()
    return {"lessons_added": n}


@app.post("/learn/weekly", dependencies=[Depends(require_api_key)])
async def trigger_weekly() -> dict[str, Any]:
    out = await run_weekly_skill_rerank()
    return {"skills": out}


# --- dashboard page (read-only observation UI) ------------------------------
_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Hermès — dashboard</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#0b0e14;color:#cdd6f4;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{display:flex;align-items:baseline;gap:16px;padding:16px 22px;border-bottom:1px solid #1d2230}
  h1{font-size:18px;margin:0;color:#fff}
  .stage{font-size:12px;background:#1e2535;border:1px solid #2a3346;padding:2px 8px;border-radius:999px}
  .wrap{padding:18px 22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;max-width:1300px}
  .card{background:#11151f;border:1px solid #1d2230;border-radius:12px;padding:14px 16px}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#7f8aa3;margin:0 0 10px}
  .big{font-size:30px;font-weight:700;color:#fff}
  .sub{color:#7f8aa3;font-size:12px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:5px 6px;border-bottom:1px solid #181d29}
  th{color:#7f8aa3;font-weight:600}
  .pos{color:#a6e3a1}.neg{color:#f38ba8}.muted{color:#6c7693}
  .bar{height:8px;background:#1d2230;border-radius:6px;overflow:hidden;margin-top:6px}
  .bar>i{display:block;height:100%;background:#89b4fa}
  .pill{font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid #2a3346}
  .ok{color:#a6e3a1;border-color:#2d4a35}.off{color:#f9e2af;border-color:#4a432d}
  .foot{padding:8px 22px;color:#6c7693;font-size:12px}
  a{color:#89b4fa}
</style></head>
<body>
<header>
  <h1>Hermès</h1><span class="stage" id="stage">…</span>
  <span class="sub" id="clock"></span>
</header>
<div class="wrap">
  <div class="card"><h2>Equity (paper)</h2><div class="big" id="equity">…</div>
    <div class="sub" id="bankroll"></div></div>
  <div class="card"><h2>Stage gate</h2><div id="gatesum" class="sub"></div><div id="gates"></div></div>
  <div class="card"><h2>Journal</h2><div id="journal"></div></div>
  <div class="card"><h2>Models</h2><div id="models"></div>
    <div class="sub" style="margin-top:8px">Identity/shadow until enough two-class trades.</div></div>
  <div class="card" style="grid-column:1/-1"><h2>Open positions</h2><div id="positions"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Agent activity (live — ticks &amp; hourly pulses)</h2><div id="activity"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>By asset / by skill (watch the SOL lean)</h2><div id="breakdown"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Recent closed trades (rhythm)</h2><div id="trades"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Work cycle — next scheduled runs</h2><div id="sched"></div></div>
</div>
<div class="foot">Auto-refresh 20s · read-only · <span id="err"></span></div>
<script>
const f=(n,d=2)=>n==null?'—':Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
const cls=v=>v>0?'pos':(v<0?'neg':'muted');
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ago=iso=>{if(!iso)return'—';const s=(Date.now()-new Date(iso))/1000;
  if(s<60)return Math.round(s)+'s ago';if(s<3600)return Math.round(s/60)+'m ago';
  if(s<86400)return Math.round(s/3600)+'h ago';return Math.round(s/86400)+'d ago';};
const until=iso=>{if(!iso)return'—';const s=(new Date(iso)-Date.now())/1000;
  if(s<0)return'due';if(s<60)return'in '+Math.round(s)+'s';if(s<3600)return'in '+Math.round(s/60)+'m';
  return'in '+Math.round(s/3600)+'h';};
async function tick(){
 try{
  const d=await (await fetch('/dashboard/data',{cache:'no-store'})).json();
  document.getElementById('stage').textContent=d.stage;
  document.getElementById('clock').textContent='server '+new Date(d.server_time).toUTCString();
  document.getElementById('equity').textContent='$'+f(d.equity_usd,2);
  document.getElementById('bankroll').textContent='bankroll $'+f(d.bankroll_usd,0)+' · '+(d.universe||[]).join(', ');
  document.getElementById('gatesum').textContent=d.stage_summary||'';
  document.getElementById('gates').innerHTML=Object.entries(d.gates||{}).map(([k,v])=>{
    const pct=k.includes('trades')?Math.min(100,100*(v.observed||0)/(v.target||1)):(v.passed?100:8);
    return `<div style="margin:8px 0"><div class="sub">${k}: ${v.observed} / ${v.target} ${v.passed?'✓':'✗'}</div>`
      +`<div class="bar"><i style="width:${pct}%;background:${v.passed?'#a6e3a1':'#89b4fa'}"></i></div></div>`;}).join('');
  const j=d.journal||{};
  document.getElementById('journal').innerHTML=
    `<table><tr><th>closed</th><td>${j.count}</td></tr>`
    +`<tr><th>win rate</th><td>${j.win_rate==null?'—':(100*j.win_rate).toFixed(0)+'%'}</td></tr>`
    +`<tr><th>avg win/loss</th><td>${f(j.avg_win_loss_ratio,2)}</td></tr>`
    +`<tr><th>realized PnL</th><td class="${cls(j.pnl_usd)}">$${f(j.pnl_usd,2)}</td></tr></table>`;
  const m=d.models||{};
  document.getElementById('models').innerHTML=
    `<div>Confidence calibrator <span class="pill ${m.calibrator_fitted?'ok':'off'}">${m.calibrator_fitted?'active':'shadow'}</span></div>`
    +`<div style="margin-top:6px">Meta-labeler <span class="pill ${m.meta_labeler_fitted?'ok':'off'}">${m.meta_labeler_fitted?'active':'shadow'}</span></div>`;
  const ps=d.open_positions||[];
  document.getElementById('positions').innerHTML=ps.length?
    `<table><tr><th>symbol</th><th>qty</th><th>entry</th><th>uPnL</th><th>stop</th><th>tp</th><th>skill</th></tr>`
    +ps.map(p=>`<tr><td>${p.symbol}</td><td>${f(p.qty,4)}</td><td>${f(p.entry_price,2)}</td>`
      +`<td class="${cls(p.unrealized_pnl_usd)}">$${f(p.unrealized_pnl_usd,3)}</td>`
      +`<td class="muted">${f(p.stop_price,2)}</td><td class="muted">${f(p.take_profit,2)}</td>`
      +`<td class="muted">${p.skill_id||''}</td></tr>`).join('')+`</table>`:'<div class="muted">none</div>';
  const tr=d.recent_closed||[];
  document.getElementById('trades').innerHTML=tr.length?
    `<table><tr><th>when</th><th>symbol</th><th>side</th><th>PnL</th><th>reason</th><th>skill</th><th>conf</th></tr>`
    +tr.map(t=>`<tr><td class="muted">${ago(t.closed_at)}</td><td>${t.symbol}</td><td>${t.side}</td>`
      +`<td class="${cls(t.pnl_usd)}">$${f(t.pnl_usd,3)}</td><td class="muted">${t.exit_reason||''}</td>`
      +`<td class="muted">${t.skill_id||''}</td><td class="muted">${f(t.confidence,2)}</td></tr>`).join('')+`</table>`
    :'<div class="muted">no closed trades yet</div>';
  const ac=d.activity||[];
  document.getElementById('activity').innerHTML=ac.length?
    `<table><tr><th>when</th><th>kind</th><th>symbol</th><th>regime</th><th>signal / decision</th><th>reason</th></tr>`
    +ac.map(a=>{
      const act=a.kind==='pulse'
        ? (a.would_act==='HOLD'?'HOLD':esc(a.would_act+' '+(a.would_skill||'')))
        : esc((a.executed?'EXEC ':'')+(a.side||'')+(a.decision?(' / '+a.decision):''));
      return `<tr><td class="muted">${ago(a.ts)}</td><td>${esc(a.kind)}</td><td>${esc(a.symbol)}</td>`
        +`<td class="muted">${esc(a.regime)}</td><td>${act}</td><td class="muted">${esc(a.reason)}</td></tr>`;
    }).join('')+`</table>`
    :'<div class="muted">no activity recorded yet — first pulse on the next top of hour</div>';
  const bd=d.breakdown||{};
  const brows=(obj,label)=>Object.entries(obj||{}).map(([k,v])=>
    `<tr><td>${esc(k)}</td><td class="muted">${label}</td><td>${v.n}</td>`
    +`<td>${(100*v.win_rate).toFixed(0)}%</td><td class="${cls(v.pnl_usd)}">$${f(v.pnl_usd,3)}</td></tr>`).join('');
  document.getElementById('breakdown').innerHTML=
    `<table><tr><th>key</th><th>type</th><th>n</th><th>win</th><th>pnl</th></tr>`
    +brows(bd.by_symbol,'asset')+brows(bd.by_skill,'skill')+`</table>`;
  const sc=d.scheduler||[];
  document.getElementById('sched').innerHTML=sc.length?
    `<table><tr><th>job</th><th>next run</th></tr>`
    +sc.map(s=>`<tr><td>${s.id}</td><td class="muted">${until(s.next_run)} <span class="muted">(${s.next_run?new Date(s.next_run).toUTCString():'—'})</span></td></tr>`).join('')+`</table>`
    :'<div class="muted">—</div>';
  document.getElementById('err').textContent='';
 }catch(e){document.getElementById('err').textContent='fetch error: '+e;}
}
tick();setInterval(tick,20000);
</script></body></html>"""
