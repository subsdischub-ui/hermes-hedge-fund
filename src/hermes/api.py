"""FastAPI surface — health, status, manual tick, dashboard data."""

from __future__ import annotations

import contextlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from hermes.execution import get_exchange
from hermes.graph import HermesOrchestrator
from hermes.learning import run_nightly_reflection, run_weekly_skill_rerank
from hermes.llm.router import get_router
from hermes.logging_setup import configure_logging, get_logger
from hermes.memory.journal import TradeJournal
from hermes.memory.store import get_db
from hermes.risk.stage import StageManager
from hermes.scheduler import build_scheduler
from hermes.settings import get_settings

log = get_logger(__name__)


# Hermès' identity + rules for the operator-chat. He explains and advises about the
# business; he can NEVER trade, move funds, or change config via the conversation.
_CHAT_SYSTEM = """Tu es Hermès, l'agent de trading crypto AUTONOME de cette entreprise \
(fonds spéculatif personnel, micro-capital, perps Binance USDT-M, paper-first). Tu parles \
avec l'opérateur-fondateur, en français, avec une honnêteté absolue et de la concision \
— pas de jargon gratuit, pas de promesses, pas de spin.

CE QUE TU ES (ne te trompe JAMAIS là-dessus) : tu tournes 24/7 et tu trades en paper de façon \
ENTIÈREMENT AUTONOME. À chaque tick (toutes les ~4h, aligné sur tes barres 4h), TU décides ET \
TU exécutes toi-même tes ordres paper, via le moteur déterministe (selective_mode) + le risk-engine \
+ le journal. PERSONNE ne déclenche tes trades à la main : les trades déjà au journal sont les TIENS. \
Entre deux ticks, un pulse horaire montre l'état des jauges ; si une jauge s'arme entre deux ticks, \
c'est TOI qui la prendras au prochain tick 4h — pas l'opérateur.

CE QUE CE CHAT NE FAIT PAS : cette CONVERSATION est en lecture seule — parler ici ne place aucun \
ordre et ne change aucune config. Mais cela ne veut PAS dire que le trading est manuel : ton moteur \
autonome s'en charge. Ne dis JAMAIS à l'opérateur qu'il doit trader « à la main » — c'est faux. La \
seule chose manuelle est l'approvisionnement en argent RÉEL (étapes live futures), pas le paper.

RÈGLE ABSOLUE ANTI-INVENTION : le contexte ci-dessous (état live, historique des trades, jauges) \
contient TOUTES les données réelles dont tu disposes. Tu n'as ACCÈS À RIEN D'AUTRE — pas de mémoire \
de séances passées, pas de détail au-delà de ce qui est listé. Si on te demande un détail absent du \
contexte (un trade non listé, une raison précise non donnée, un chiffre non fourni), tu DOIS répondre \
explicitement « cette donnée n'est pas dans mon contexte actuel » — JAMAIS l'inventer, même \
partiellement, même pour paraître plus complet. Une réponse honnête et incomplète vaut TOUJOURS mieux \
qu'une réponse plausible et fabriquée. C'est nous qui le découvrons à chaque fois — ne nous y refais \
pas prendre.

TON SLATE LIVE ACTUEL (RESTRUCTURATION 2026-07-02 — toutes SÉLECTIVES, elles ne tradent QUE \
quand leur jauge est armée, sinon elles attendent par discipline) :
- #1 Mean reversion (régime) : range mean-reversion, skill DISTINCT (jamais concerné par le \
retrait de S1 ci-dessous).
- #2 Funding squeeze (funding z>=2, long ~96h) : continuation sur funding extrême, orthogonale à #1 \
— le SEUL edge validé du livre (joint-null p=0.0005 après le fix de mesure du 2026-07-02).
SUSPENDU DU SLATE (2026-07-11, validé opérateur) : #3 Funding carry delta-neutral — contribution \
livre LOO-Δ −0.081 (négative) et edge par trade trop mince au coût taker honnête (+16..23bps, \
OOS-40 ≈ 0 ; sain uniquement en exécution maker, jamais mesurée). Sa jauge reste affichée \
(le signal est calculé mais JAMAIS exécuté) ; réveil : fill-rate maker réel mesuré au stage_1.
RETIRÉ DU SLATE (2026-07-02) : l'ex-Strategy #1 momentum (ts_mom_vol_scaled, |score|>=1.0) — \
mesuré comme un drag bull-beta (book LOO-Δ Sharpe -0.146, joint-null p=0.507 : indiscernable d'une \
entrée aléatoire). Le contexte ci-dessous détaille les chiffres et le nouveau gate \
(edge_paper_trades) — utilise-les si on te pose la question, ne les improvise jamais.
PHILOSOPHIE : discipline > agitation. Forcer des trades dans le bruit détruit l'edge. Un livre à \
événements rares (ex: #2 ~13 évts/an) qui reste calme des semaines est une PROTECTION du capital, \
pas une panne. La croissance vient de la patience, de l'orthogonalité, et (plus tard, après les \
gates) du levier."""


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
    allowed = s.yaml("strategy", "allowed_skills") or []
    edge_count = await journal.count_closed_by_skills(allowed)
    # Stage gates require a real walk-forward Sharpe and rule-violation count;
    # the status endpoint reports the metrics it can prove and leaves the
    # other gates unfilled so the operator sees them as BLOCKED rather than
    # passed-on-a-proxy. edge_paper_trades counts ONLY current-slate trades
    # (restructuration 2026-07-02) — legacy/retired skills never advance a stage.
    report = stage_mgr.evaluate(
        metrics={
            "live_trades": stats["count"],
            "paper_trades": stats["count"],
            "edge_paper_trades": edge_count,
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
    all_closed = await journal.recent_closed(since=datetime.now(UTC) - timedelta(days=365))
    recent = all_closed[:20]
    allowed = s.yaml("strategy", "allowed_skills") or []
    edge_count = await journal.count_closed_by_skills(allowed)
    report = StageManager().evaluate(
        {
            "live_trades": stats["count"],
            "paper_trades": stats["count"],
            "edge_paper_trades": edge_count,
            "rule_violations": 0,
        }
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
        "equity_curve": _equity_curve(all_closed, s.bankroll_usd, equity),
        "strategies": _strategies_ab(all_closed, positions),
        "forward": _forward_status(s),
        "governance": _governance_status(),
        "restructuration": _restructuration_status(s, edge_count),
        "trm_status": _trm_status_safe(),
        "protections": await _protections_status(s, equity),
        "server_time": datetime.now(UTC).isoformat(),
    }


def _recent_activity(s, n: int = 30) -> list[dict]:
    """Last agent decisions/pulses (why it traded or held) for the activity feed."""
    from hermes.runtime.activity import ActivityLog

    return ActivityLog(s.data_dir / "activity.json").recent(n)


# Forward-collection flux names surfaced on the dashboard, mapped to the log name each
# collector writes under (see hermes.forward.run.default_collector_names / _*_collector).
# Kept as a plain dict (not the cron's own list) so the dashboard never depends on
# import-time network seams in hermes.forward.run.
_FORWARD_FLUX_LOGS = {
    "top_lsr": "top_lsr",
    "skew": "skew",
    "comex_curve": "comex_curve",
    "comex_spreads": "comex_spreads",
    "macro_dot_vs_ff": "macro_dot_vs_ff",
    "funding_squeeze": "shadow/funding_squeeze_continuation",
    "xs_reversion": "shadow/xs_funding_reversion",
}


def _forward_status(s) -> dict[str, dict]:
    """Fail-soft per-flux status of the forward/ collection logs.

    Reads ``{HERMES_DATA_DIR}/forward/<flux>/log.parquet`` via the same resolver the
    cron writes through (``hermes.forward.logio``). A flux with no log yet (fresh
    local checkout — production has these under ``/data/forward``) reports
    ``status: "no_data"`` instead of raising. top_lsr additionally breaks its row
    count down per symbol since it is the flux the restructuration named explicitly
    (18 (symbol, ratio_type) pairs across 7 live crypto perps + XAU/XAG)."""
    from hermes.forward import logio

    out: dict[str, dict] = {}
    for flux, log_name in _FORWARD_FLUX_LOGS.items():
        try:
            df = logio.read_log(log_name, data_dir=s.data_dir)
        except Exception as e:  # never let a corrupt/partial parquet break the dashboard
            out[flux] = {"status": "error", "error": str(e)}
            continue
        if df is None or df.empty:
            out[flux] = {"status": "no_data", "rows": 0}
            continue
        last_ts = None
        if "ts" in df.columns:
            try:
                last_ts = datetime.fromtimestamp(
                    float(df["ts"].max()) / 1000.0, tz=UTC
                ).isoformat()
            except Exception:
                last_ts = None
        entry: dict[str, Any] = {
            "status": "collecting",
            "rows": len(df),
            "last_ts": last_ts,
        }
        if flux == "top_lsr" and "symbol" in df.columns:
            with contextlib.suppress(Exception):
                entry["by_symbol"] = {
                    str(sym): int(n) for sym, n in df["symbol"].value_counts().items()
                }
        out[flux] = entry
    return out


def _trm_status_safe() -> str:
    """Fail-soft wrapper around trm.verifier.trm_status_summary — never let a
    dashboard render 500 on a TRM config surprise."""
    try:
        from hermes.trm.verifier import trm_status_summary

        return trm_status_summary()
    except Exception as e:
        return f"error: {e}"


async def _protections_status(s, equity_usd: float) -> dict[str, Any]:
    """Operator-visibility block (mémoire hermes-visibility-rule: a shipped safety
    mechanism must be racontable, not silent) for the two gates added 2026-07-10
    (docs/SYSTEME_AUTONOME.md partie III): macro blackout + rolling drawdown kill.
    Both run SHADOW at stage_0_paper (see config/default.yaml) — this block is what
    lets the operator SEE what the shadow verdict currently is, independent of
    whatever the last live tick happened to observe. Computed fresh from the same
    pure functions/config the RiskEngine gates use, fail-soft per sub-block so a
    corrupt calendar or a DB hiccup never takes down the whole dashboard."""
    from hermes.risk.guardrails import _realized_pnl_window_usd

    out: dict[str, Any] = {}

    macro_cfg = s.yaml("macro_blackout", default={}) or {}
    try:
        from hermes.risk.macro_blackout import (
            blackout_status,
            get_macro_calendar,
            horizon_warning,
        )

        cal = get_macro_calendar()
        now = datetime.now(UTC)
        verdict = blackout_status(
            now, cal,
            pre_h=float(macro_cfg.get("pre_hours", 2.0)),
            post_h=float(macro_cfg.get("post_hours", 2.0)),
        )
        out["macro_blackout"] = {
            "enabled": bool(macro_cfg.get("enabled", True)),
            "mode": "enforce" if s.stage in set(macro_cfg.get("enforce_stages", [])) else "shadow",
            "verdict": verdict,
            "horizon_warning": horizon_warning(now, cal),
        }
    except Exception as e:
        out["macro_blackout"] = {"status": "error", "error": str(e)}

    rolling_cfg = s.yaml("rolling_loss_kill", default={}) or {}
    try:
        window_days = int(rolling_cfg.get("window_days", 30))
        kill_pct = float(rolling_cfg.get("kill_pct", 10.0))
        pnl_window = await _realized_pnl_window_usd(window_days)
        blocked = pnl_window / max(equity_usd, 1e-9) <= -(kill_pct / 100)
        out["rolling_loss_kill"] = {
            "enabled": bool(rolling_cfg.get("enabled", True)),
            "mode": "enforce" if s.stage in set(rolling_cfg.get("enforce_stages", [])) else "shadow",
            "window_days": window_days,
            "kill_pct": kill_pct,
            "pnl_usd": pnl_window,
            "equity_usd": equity_usd,
            "blocked": blocked,
        }
    except Exception as e:
        out["rolling_loss_kill"] = {"status": "error", "error": str(e)}

    return out


def _governance_status() -> dict[str, Any]:
    """Fail-soft governance snapshot: trial-registry counts (deflation truth) plus the
    informative DeadEndsLedger verdict for every pre-registration. Both sources are
    already fail-soft internally, but this wraps them again defensively so a
    dashboard render can never 500 on a missing/corrupt governance file."""
    out: dict[str, Any] = {"trials_by_book": {"status": "no_data"}, "dead_ends_verdicts": {}}
    try:
        from hermes.research import s10_governance as gov

        reg = gov.BookTrialRegistry()
        out["trials_by_book"] = reg.as_dict()
    except Exception as e:
        out["trials_by_book"] = {"status": "no_data", "error": str(e)}
    try:
        from hermes.forward import prereg

        out["dead_ends_verdicts"] = prereg.dead_ends_verdicts()
    except Exception as e:
        out["dead_ends_verdicts"] = {"status": "no_data", "error": str(e)}
    return out


def _restructuration_status(s, edge_paper_trades_observed: int) -> dict[str, Any]:
    """Structured facts from the 2026-07-02 restructuration (docs/RESTRUCTURATION_2026-07.md)
    so the dashboard and the chat share one source of the same numbers. Sourced from
    the live config where possible (slate, gate target) rather than re-hardcoding
    values that config already owns; the historical/dated facts (LOO-Δ, deadlines)
    are not runtime-computed anywhere, so they are recorded here verbatim from the
    dossier — fail-soft to a static fallback if the config read ever breaks."""
    try:
        slate = list(s.yaml("strategy", "allowed_skills") or [])
    except Exception:
        slate = []
    try:
        gate_target = s.yaml("stages", "stage_0_paper", "advance_when", "edge_paper_trades")
        gate_target = int(gate_target) if gate_target is not None else 40
    except Exception:
        gate_target = 40
    return {
        "slate": slate,
        "retired": [
            # p-values post-fix look-ahead du 2026-07-02 (bug fxgate _basket_event_return
            # corrigé, re-run officiel — research_crypto2/JOURNAL_rerun_gatefix.md) :
            # S1 p 0.69→0.507 (toujours indiscernable du hasard), LOO-Δ inchangé.
            {"skill_id": "ts_mom_vol_scaled", "loo_delta_sharpe": -0.146, "joint_null_p": 0.507},
        ],
        "suspended": [
            # 2026-07-11, validé opérateur : LOO-Δ −0.081 (négatif), edge taker mince
            # (+16..23bps/trade, OOS-40 ~0 — sain uniquement en maker, jamais mesuré),
            # 0 trade live. Jauge + shadow conservés. Réveil : fill-rate maker au stage_1.
            {"skill_id": "funding_carry_harvest", "loo_delta_sharpe": -0.081,
             "reveil": "fill-rate maker réel mesuré au stage_1"},
        ],
        "gate": {
            "edge_paper_trades_target": gate_target,
            "edge_paper_trades_observed": edge_paper_trades_observed,
        },
        "metals_collect_now": {
            "symbols": ["XAUUSDT", "XAGUSDT"],
            "retest_deadline": "2027-07-02",
            "kill_rule": "n<30 événements ou aucun bear couvert → axe mort",
        },
        "fx": {"status": "fermé", "reopen_check": "2027-07-02"},
        "global_cutoff": "2028-01-02",
        "dossier": "docs/RESTRUCTURATION_2026-07.md",
        # Inventaire définitif du système (aptitudes/algorithmes/sûreté/venues,
        # audit adversarial 40 vérifications, défauts corrigés le 2026-07-10) :
        "systeme": "docs/SYSTEME_AUTONOME.md (2026-07-10 — 3 défauts de fidélité "
                   "backtest/live corrigés : no-pyramid cross-skill, plancher de "
                   "confiance vs sleeves auto-sélectives, résilience LLM ; adapter "
                   "Binance réel écrit, étages re-pointés sur le compte possédé)",
        # Test S2-largeur du 2026-07-02 soir (PREREG research_breadth) — verdict à la
        # lettre : ÉCHEC (une fenêtre bear n=4 négative), malgré +75.8% d'événements
        # marginaux et joint-null p_max=0.003. Réserves : biais de survie non testé
        # (signe défavorable), 2022 sans aucun événement z>=2 (bear-check aveugle au
        # coeur du crash). Suite honnête = un seul test DURCI pré-enregistré (univers
        # incluant les perps délistés) — pas de re-découpage des fenêtres.
        "breadth_test": {
            "date": "2026-07-02",
            "verdict": "ÉCHEC (lettre du PREREG — bear-window n=4 négative)",
            "marginal_events_pct": 75.8,
            "joint_null_p_max5": 0.003,
            "reserves": ["biais de survie non testé", "2022: zéro événement z>=2"],
            "next": "CLOS (décision opérateur 2026-07-02 : pas de test durci)",
        },
        # Carte lake-vs-ledger du 2026-07-02/03 : les 2 derniers axes non-adjugés du
        # lake CoinGlass payé ont été testés dans la même nuit (pré-enregistrés,
        # registre 1909 puis 1910, chacun doublement reproduit par audits). Tous deux
        # MORTS proprement — le lake est désormais 100 % ADJUGÉ, zéro axe restant.
        "lake_axes": {
            "statut_global": "LAKE 100% ADJUGÉ (2026-07-03) — zéro axe non testé restant ; "
                             "l'abonnement CoinGlass peut être suspendu (données gelées sur "
                             "disque, re-souscription à la demande) — décision opérateur",
            "global_lsr_contrarian": {
                "verdict": "MORT 2026-07-02 (joint-null p 0.12-0.14 x5 seeds ET OOS -0.15 ; DSR≈0)",
                "ledger": "#37",
            },
            "taker_volume_perp": {
                "verdict": "MORT 2026-07-03 (joint-null p 0.62-0.65 x5 seeds — le hasard bat la "
                           "stratégie 63% du temps — ET OOS -0.11 ; garde-fous sains : pas un "
                           "redux, 57 événements en 2022)",
                "ledger": "#38",
            },
            "top_lsr_daily_contrarian": {
                "verdict": "MORT (ledger #25, Stream 2 sur historique complet) — la collecte "
                           "quotidienne continue pour la porte intraday uniquement",
            },
        },
    }


# Strategy attribution: which skill belongs to which live/research strategy.
# RESTRUCTURATION 2026-07-02 (docs/RESTRUCTURATION_2026-07.md §3): ts_mom_vol_scaled
# ("Strategy #1 momentum") was RETIRED from config strategy.allowed_skills (bull-beta
# drag: book LOO-Δ Sharpe -0.146, joint-null p=0.507 post-gatefix). mean_reversion_regime is a
# DISTINCT skill built on a different signal (never part of the S1 audit sleeve) and
# stays live. Before this fix both were folded into one "Strategy 1 (Momentum)" card,
# which would have shown future mean_reversion_regime trades as if they belonged to a
# retired strategy — _STRAT1_SKILLS now names ONLY the live skill; every retired/legacy
# skill moves to _RETIRED_SKILLS so the 7 historical trades stay visible but correctly
# attributed as history, not as an active strategy card.
_STRAT1_SKILLS = {"mean_reversion_regime"}  # live: Strategy 1 = mean reversion (régime)
_RETIRED_SKILLS = {
    "ts_mom_vol_scaled", "breakout_vol_expansion",
    "intraday_tod_btc", "trend_follow_ema", "mean_reversion_atr",
}
_STRAT2_SKILLS = {
    "funding_squeeze_continuation",  # the validated, live Strategy #2
    # refuted/retired research candidates kept so any legacy journal rows still attribute
    "weekend_liquidity_fade", "funding_extreme_fade", "vr_antitrend_fade",
    "flow_exhaustion_fade", "cs_residual_reversal",
}
_STRAT3_SKILLS = {"funding_carry_harvest"}  # validated, display-only (needs hedge leg)


def _equity_curve(closed_rows, start_equity: float, current_equity: float) -> list[dict]:
    """Cumulative realised-PnL equity curve from closed trades (oldest->newest)."""
    rows = sorted(
        (r for r in closed_rows if r.closed_at is not None), key=lambda r: r.closed_at
    )
    eq = float(start_equity)
    pts = [{"t": None, "eq": eq}]
    for r in rows:
        eq += float(r.pnl_usd or 0.0)
        pts.append({"t": r.closed_at.isoformat(), "eq": round(eq, 4)})
    return pts


def _strategy_stats(rows) -> dict:
    pnl = [float(r.pnl_usd or 0.0) for r in rows]
    n = len(pnl)
    wins = sum(1 for p in pnl if p > 0)
    return {
        "n": n,
        "pnl_usd": round(sum(pnl), 4),
        "win_rate": round(wins / n, 3) if n else None,
    }


def _strategies_ab(closed_rows, open_positions=None) -> dict:
    """Side-by-side A/B of the three sleeves: each shows its live (closed) trades plus
    an honest research/status line. Strategy #3 (carry) flips to live the moment a
    delta-neutral pair is OPEN, even before the first one closes. Retired/legacy
    skills (ts_mom_vol_scaled and pre-2026-07 history) are reported separately so they
    never conflate with the still-live mean_reversion_regime sleeve."""
    s1 = [r for r in closed_rows if (r.skill_id or "") in _STRAT1_SKILLS]
    s2 = [r for r in closed_rows if (r.skill_id or "") in _STRAT2_SKILLS]
    s3 = [r for r in closed_rows if (r.skill_id or "") in _STRAT3_SKILLS]
    retired = [r for r in closed_rows if (r.skill_id or "") in _RETIRED_SKILLS]
    carry_open = any((getattr(p, "skill_id", "") or "") in _STRAT3_SKILLS for p in (open_positions or []))
    return {
        "strategy_1": {
            "label": "Mean reversion (régime)",
            "status": "live",
            "live": _strategy_stats(s1),
        },
        "strategy_2": {
            "label": "Funding squeeze (orthogonal)",
            # live once funding trades close; "wired" while the feed runs but no trade yet
            "status": "live" if s2 else "wired",
            "live": _strategy_stats(s2),
            # Honest status: after a systematic ~800-backtest sweep, ONE brick survived
            # the strict gate AND generalized cross-asset: funding_squeeze_continuation
            # (extreme-positive funding -> long, ~96h hold). RESTATED per the 2026-06-25
            # re-audit + 2026-07-02 gate fix: the old "+1.1" headline was the ANNUALIZED
            # number — honest magnitudes are per-trade Sharpe ~0.3, OOS-40 ~0.3; weekly
            # book rho to momentum was 0.35 (not the daily ~0.04). Basket joint-null
            # p=0.0005 post-fix: the ONE real edge of the book.
            "research": {
                "oos_validated_bricks": 1,
                "candidates_tested": 24,
                "note": "funding_squeeze_continuation: le seul edge réel du livre (joint-null "
                        "basket p=0.0005 post-fix 2026-07-02, bear-positive). Magnitudes honnêtes "
                        "restatées par le ré-audit : Sharpe par trade ~0.3, OOS-40 ~0.3 (le "
                        "'+1.1' historique était le chiffre annualisé), ~13 évts/an sur 3 actifs. "
                        "Long-only, s'arme sur funding z>=2 seulement.",
            },
        },
        "strategy_3": {
            "label": "Funding carry (delta-neutral)",
            # live once a carry pair is open or has closed; the two-leg engine is wired
            "status": "live" if (s3 or carry_open) else "validated",
            "live": _strategy_stats(s3),
            "research": {
                "oos_validated_bricks": 1,
                "candidates_tested": 40,
                "note": "SUSPENDU du slate le 2026-07-11 (validé opérateur — LOO-Δ −0.081, "
                        "0 trade live ; jauge conservée, signal jamais exécuté ; réveil : "
                        "fill-rate maker réel au stage_1). "
                        "Delta-neutral funding carry (cash-and-carry), RESTATÉ par le ré-audit "
                        "2026-06-25 : au coût taker honnête (30bps deux jambes) l'edge est mince "
                        "(+16..+23bps/trade) avec OOS-40 ~0 — le '+2.4..+4.5' historique ne "
                        "reproduit pas au coût honnête ; sain seulement en exécution maker "
                        "(~12bps, +34bps/trade), d'où HOLD-only tant qu'un fill-rate réel n'est "
                        "pas mesuré. Route structurellement en paire (short perp + long spot), "
                        "JAMAIS à nu (prouvé perdant sur 9 ans).",
            },
        },
        "retired_skills": {
            "label": "Skills retirés (historique)",
            "status": "retired",
            "live": _strategy_stats(retired),
            "skills": sorted(_RETIRED_SKILLS),
            "research": {
                "note": "ts_mom_vol_scaled (ex-Strategy #1 momentum) retiré le 2026-07-02 : "
                        "book LOO-Δ Sharpe -0.146 (le livre double son Sharpe sans lui), "
                        "joint-null p=0.507 post-fix (son panier de trades est indiscernable d'une entrée "
                        "aléatoire). Les trades historiques ci-dessus (incl. legacy "
                        "breakout/trend/mean_reversion_atr) restent visibles mais n'arment plus "
                        "aucun signal futur — voir docs/RESTRUCTURATION_2026-07.md §3.",
            },
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


# ------------------------------- operator chat ----------------------------- #
class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


async def _chat_context() -> str:
    """Assemble Hermès' live state so the chat answers from reality, not vibes."""
    s = get_settings()
    try:
        ex = get_exchange()
        journal = TradeJournal()
        equity = await ex.equity_usd()
        positions = await ex.open_positions()
        stats = await journal.stats()
        closed = await journal.recent_closed(since=datetime.now(UTC) - timedelta(days=365))
        ab = _strategies_ab(closed, positions)
        allowed = s.yaml("strategy", "allowed_skills") or []
        edge_count = await journal.count_closed_by_skills(allowed)
    except Exception as e:  # never let a context hiccup break the chat
        return f"(état partiellement indisponible: {e})"

    strat = "\n".join(
        f"  - {ab[k]['label']} [{ab[k]['status']}]: {ab[k]['live'].get('n', 0)} trades, "
        f"PnL ${ab[k]['live'].get('pnl_usd', 0)}"
        for k in ("strategy_1", "strategy_2", "strategy_3") if k in ab
    )
    pos = ", ".join(
        f"{p.symbol}/{p.skill_id}(qty={round(p.qty, 5)}, "
        f"uPnL ${round(p.unrealized_pnl_usd, 3) if p.unrealized_pnl_usd is not None else '?'})"
        for p in positions
    ) or "aucune"
    seen: dict[str, int] = {}
    gauges: list[str] = []
    for a in sorted((x for x in _recent_activity(s) if x.get("kind") == "pulse"),
                    key=lambda a: str(a.get("ts") or ""), reverse=True):
        sym = a.get("symbol")
        if sym in seen:
            continue
        seen[sym] = 1
        parts: list[str] = []
        for g in (a.get("gauges") or []):
            # Two gauge shapes coexist in the activity ring (pulse.py): slate gauges
            # ({sleeve, metric, val, thr, armed[, retired]}) and SHADOW gauges for
            # skills outside allowed_skills ({shadow, skill, side, conf, armed}) —
            # the latter started appearing for ts_mom the moment S1 left the slate.
            # Render both, and NEVER crash the chat over a display record.
            if g.get("shadow"):
                parts.append(
                    f"[shadow hors-slate] {g.get('skill', '?')} {g.get('side', '?')} "
                    f"conf {g.get('conf', '?')} (signal calculé, jamais exécuté)"
                )
            elif "sleeve" in g:
                retired_tag = " (retirée)" if g.get("retired") else ""
                parts.append(
                    f"{g.get('sleeve')}{retired_tag} {g.get('metric', '?')} "
                    f"{g.get('val', '?')}/{g.get('thr', '?')} "
                    f"{'ARMÉ' if g.get('armed') else 'attend'}"
                )
        gauges.append(f"  - {sym} [{a.get('regime')}]: {'; '.join(parts)}")
    # Itemised trade history — without this, the LLM has only aggregates (count/
    # win_rate/pnl) and confabulates per-trade detail when asked for it (observed
    # live: invented 4/6 trades with full conviction). Capped so context size stays
    # bounded once the journal grows past the stage_0 200-trade gate.
    _TRADE_DETAIL_CAP = 20
    trade_lines = [
        f"  - {r.closed_at:%Y-%m-%d %H:%M} {r.symbol} {r.side} skill={r.skill_id or '?'} "
        f"PnL ${round(r.pnl_usd or 0.0, 3)} (exit: {r.exit_reason or '?'})"
        for r in closed[:_TRADE_DETAIL_CAP]
    ]
    if len(closed) > _TRADE_DETAIL_CAP:
        trade_lines.append(f"  - ... et {len(closed) - _TRADE_DETAIL_CAP} trade(s) clos plus anciens (non listés ici, voir les stats agrégées ci-dessus pour le total).")
    return (
        f"ÉTAT LIVE DE NOTRE ENTREPRISE ({s.stage}):\n"
        f"- Capital paper: ${equity:.2f} (bankroll ${s.bankroll_usd}); univers {', '.join(s.universe)}\n"
        f"- Journal: {stats.get('count', 0)} trades clos, "
        f"win {stats.get('win_rate')}, PnL ${stats.get('pnl_usd', 0)}\n"
        f"- Historique détaillé des trades clos (les SEULS trades réels — n'en invente JAMAIS d'autres):\n"
        f"{chr(10).join(trade_lines) or '  (aucun trade clos pour le moment)'}\n"
        f"- Stratégies:\n{strat}\n"
        f"- Positions ouvertes: {pos}\n"
        f"- Ce que chaque sleeve voit maintenant:\n{chr(10).join(gauges) or '  (pas de pulse récent)'}\n"
        f"{_restructuration_chat_block(s, edge_count)}"
    )


def _restructuration_chat_block(s, edge_count: int) -> str:
    """Compact restructuration + collecte forward block for the chat context — same
    facts as /dashboard/data's `restructuration`/`forward` blocks, condensed to key
    numbers only (token budget). Lets the agent answer 'quel est ton slate et
    pourquoi S1 a été retiré ?', 'où en est la collecte or/argent/top_lsr ?', 'c'est
    quoi le gate maintenant ?', 'pourquoi si peu de trades ?' (réponse: slate
    rare-événements = protection, pas panne)."""
    r = _restructuration_status(s, edge_count)
    fwd = _forward_status(s)
    top_lsr = fwd.get("top_lsr", {})
    by_sym = top_lsr.get("by_symbol") or {}
    top_lsr_line = (
        f"top_lsr {top_lsr.get('status')}, {top_lsr.get('rows', 0)} lignes"
        + (f" ({', '.join(f'{k}:{v}' for k, v in sorted(by_sym.items()))})" if by_sym else "")
    )
    skew = fwd.get("skew", {})
    skew_line = f"skew {skew.get('status')}, {skew.get('rows', 0)} lignes"
    retired = r["retired"][0]
    gate = r["gate"]
    return (
        "- RESTRUCTURATION 2026-07-02 (docs/RESTRUCTURATION_2026-07.md — source de vérité ; "
        "inventaire système complet : docs/SYSTEME_AUTONOME.md, 2026-07-10):\n"
        f"  - Slate live actuel: {', '.join(r['slate']) or '(vide)'}\n"
        f"  - Retiré: {retired['skill_id']} (ex-Strategy #1 momentum) — book LOO-Δ Sharpe "
        f"{retired['loo_delta_sharpe']} (le livre double son Sharpe sans lui), joint-null p="
        f"{retired['joint_null_p']} (panier indiscernable d'une entrée aléatoire). "
        f"mean_reversion_regime reste (skill distinct, non concerné).\n"
        + (
            f"  - Suspendu (2026-07-11, validé opérateur) : {susp[0]['skill_id']} — LOO-Δ "
            f"{susp[0]['loo_delta_sharpe']} (contribution négative), edge taker trop mince "
            f"(sain uniquement en maker, jamais mesuré) ; jauge et shadow conservés ; "
            f"réveil : {susp[0]['reveil']}.\n"
            if (susp := r.get("suspended")) else ""
        )
        + f"  - Gate stage_0: edge_paper_trades {gate['edge_paper_trades_observed']}/"
        f"{gate['edge_paper_trades_target']} (compte SEULEMENT les trades du slate actuel — "
        f"un livre à événements rares prend des années par construction ; peu de trades = "
        f"protection du capital, pas une panne).\n"
        f"  - Collecte forward: {top_lsr_line} ; {skew_line}.\n"
        f"  - Métaux (XAU/XAG) : collect-now, "
        f"re-test {r['metals_collect_now']['retest_deadline']} (couperet: {r['metals_collect_now']['kill_rule']}).\n"
        f"  - FX : {r['fx']['status']}, réouverture vérifiée {r['fx']['reopen_check']}.\n"
        + (
            f"  - Test de LARGEUR S2 (élargir la règle funding gelée à 12 perps liquides, "
            f"{bt.get('date')}) : verdict {bt.get('verdict')} — malgré "
            f"+{bt.get('marginal_events_pct')}% d'événements marginaux et joint-null "
            f"p={bt.get('joint_null_p_max5')}, le PREREG à zéro tolérance s'applique. "
            f"Réserves : {', '.join(bt.get('reserves', []))}. Suite : {bt.get('next')}.\n"
            if (bt := r.get("breadth_test")) else ""
        )
        + (
            "  - Data lake CoinGlass (payé, inventorié 2026-07-02) : "
            + " ; ".join(
                f"{k} → {v.get('verdict') if isinstance(v, dict) else v}"
                for k, v in la.items()
            )
            + ".\n"
            if (la := r.get("lake_axes")) else ""
        )
        + (
            "  - Système autonome (inventaire 2026-07-10, docs/SYSTEME_AUTONOME.md) : venue "
            "d'exécution des étages RÉELS (1/2/3) = binanceusdm (le compte possédé + le venue "
            "de backtest ; adapter réel écrit, refuse sans clés, carry jamais nu). Trois défauts "
            "de fidélité backtest/live corrigés ce jour : (a) no-pyramid désormais par "
            "(symbole, skill) et non par symbole seul — S2 n'est plus étranglée par un symbole "
            "tenu par mean_reversion ; (b) le plancher de confiance n'étrangle plus les sleeves "
            "auto-sélectives (S2 s'arme bien à z>=2, plus à z>=2.2) ; (c) une panne LLM ne "
            "suspend plus le tick — la règle déterministe décide (LLM advisory). Constats "
            "restants avant tout passage réel : blackout macro FOMC/CPI, circuit-breaker "
            "multi-jours, jambe spot carry réelle, statut TRM.\n"
        )
        + f"  - Couperet global du narratif « revenu par trading » : {r['global_cutoff']}."
    )


@app.post("/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest) -> dict[str, Any]:
    if not (req.message or "").strip():
        raise HTTPException(status_code=400, detail="message vide")
    try:
        context = await _chat_context()
    except Exception as e:  # defense in depth: a context/display bug must degrade, never 500
        log.warning("chat_context_failed", error=str(e))
        context = f"(état live indisponible — détail technique: {e})"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _CHAT_SYSTEM + "\n\n" + context}
    ]
    for h in (req.history or [])[-10:]:
        role, content = h.get("role"), h.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)[:4000]})
    messages.append({"role": "user", "content": req.message[:4000]})
    try:
        out = await get_router().chat(
            agent_role="chat", messages=messages, temperature=0.4, max_tokens=4096
        )
    except Exception as e:
        log.warning("chat_llm_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"LLM indisponible: {e}") from e
    return {"reply": out.get("content") or "(pas de réponse)", "model": out.get("model")}


@app.get("/chat", response_class=HTMLResponse)
async def chat_page() -> str:
    return _CHAT_HTML


@app.post("/tick/{symbol_base}", dependencies=[Depends(require_api_key)])
async def manual_tick(symbol_base: str) -> dict[str, Any]:
    # Resolve the base coin (e.g. "BTC", "SUI") to its FULL universe symbol rather
    # than hardcoding a quote/format — the universe now uses Binance perp symbols
    # ("{COIN}/USDT:USDT"), so a hardcoded "/USDC" would 404 for every pair.
    base = symbol_base.upper()
    symbol = next(
        (s for s in get_settings().universe if s.split("/")[0].upper() == base), None
    )
    if symbol is None:
        raise HTTPException(404, f"{base} not in universe")
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
  <a href="/chat" style="margin-left:auto;font-size:14px">💬 parler à Hermès</a>
</header>
<div class="wrap">
  <div class="card" style="grid-column:1/-1;border-color:#3a4a6a">
    <h2>Live — what each strategy sees right now</h2><div id="live"></div></div>
  <div class="card" style="grid-column:1/-1;border-color:#4a3a6a">
    <h2>🔬 Shadow sleeves — research, log-only (never traded)</h2><div id="shadow"></div></div>
  <div class="card"><h2>Equity (paper)</h2><div class="big" id="equity">…</div>
    <div class="sub" id="bankroll"></div></div>
  <div class="card"><h2>Stage gate</h2><div id="gatesum" class="sub"></div><div id="gates"></div></div>
  <div class="card"><h2>Journal</h2><div id="journal"></div></div>
  <div class="card"><h2>Models</h2><div id="models"></div>
    <div class="sub" style="margin-top:8px">Identity/shadow until enough two-class trades.</div></div>
  <div class="card" style="grid-column:1/-1"><h2>Open positions</h2><div id="positions"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Equity curve (paper, realised)</h2><div id="eqchart"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Strategies — A/B (side by side)</h2><div id="ab"></div></div>
  <div class="card" style="grid-column:1/-1;border-color:#3a4a6a">
    <h2>Restructuration &amp; collecte forward (2026-07-02)</h2><div id="restructuration"></div></div>
  <div class="card" style="grid-column:1/-1">
    <h2>Agent activity (live — ticks &amp; hourly pulses)</h2>
    <div style="margin-bottom:8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <input id="actfilter" placeholder="filter (symbol, skill, reason…)" oninput="render()"
        style="background:#0b0e14;border:1px solid #2a3346;color:#cdd6f4;border-radius:6px;padding:4px 8px;font:13px sans-serif"/>
      <label class="sub" style="cursor:pointer"><input type="checkbox" id="hidehold" onchange="render()"/> hide HOLD pulses</label>
    </div>
    <div id="activity"></div>
  </div>
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
let D={};
async function tick(){
  try{ D=await (await fetch('/dashboard/data',{cache:'no-store'})).json();
       document.getElementById('err').textContent=''; render(); }
  catch(e){ document.getElementById('err').textContent='fetch error: '+e; }
}
function renderEq(pts){
  const el=document.getElementById('eqchart');
  const ys=(pts||[]).map(p=>p.eq);
  if(ys.length<2){ el.innerHTML='<div class="muted">no closed trades yet — curve starts at first close</div>'; return; }
  const W=900,H=160,pad=8, mn=Math.min(...ys), mx=Math.max(...ys), rng=(mx-mn)||1;
  const X=i=>pad+i*(W-2*pad)/(ys.length-1), Y=v=>H-pad-(v-mn)/rng*(H-2*pad);
  const dp=ys.map((v,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)).join(' ');
  const first=ys[0], last=ys[ys.length-1], col=last>=first?'#a6e3a1':'#f38ba8';
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" height="160" preserveAspectRatio="none">`
    +`<path d="${dp}" fill="none" stroke="${col}" stroke-width="2"/></svg>`
    +`<div class="sub">start $${f(first,2)} → now $${f(last,2)} · ${ys.length-1} closed trades</div>`;
}
function renderLive(d){
  const el=document.getElementById('live'); if(!el) return;
  const tick=(d.scheduler||[]).find(x=>x.id==='market_tick');
  const head=`<div class="sub" style="margin-bottom:10px">The agent evaluates all assets every tick. Each sleeve trades ONLY when its gauge is ARMED — otherwise it holds, by design (selective edge).`
    +(tick?` · next decision <b>${until(tick.next_run)}</b>`:'')+`</div>`;
  const acts=(d.activity||[]).filter(a=>a.kind==='pulse').slice()
    .sort((x,y)=>new Date(y.ts)-new Date(x.ts));
  const bySym={}; for(const a of acts){ if(!bySym[a.symbol]) bySym[a.symbol]=a; }
  const syms=(d.universe||[]).filter(s=>bySym[s]);
  if(!syms.length){ el.innerHTML=head+'<div class="muted">no pulse logged yet — waiting for the next heartbeat</div>'; return; }
  const gbar=g=>{ const pct=Math.max(4,Math.min(100,100*(Number(g.val)/(g.thr||1))));
    // Retired gauges (e.g. Momentum #1 / ts_mom_vol_scaled, dropped from allowed_skills
    // 2026-07-02) still compute + display what they SEE, but can never arm — render
    // them muted/dead instead of a live ARMED-capable bar so they aren't mistaken for
    // an active strategy.
    const col=g.retired?'#3a4258':(g.armed?'#a6e3a1':(g.status?'#cba6f7':'#89b4fa'));
    const tag=g.retired?`<span class="pill off">retiré</span>`
      :(g.status?`<span class="pill off">${esc(g.status)}</span>`
      :(g.armed?'<span class="pill ok">ARMED</span>':'<span class="pill off">holding</span>'));
    return `<div style="margin:6px 0${g.retired?';opacity:.55':''}"><div class="sub">${esc(g.sleeve)} — ${esc(g.metric)} <b>${f(g.val,2)}</b> / ${f(g.thr,2)} `
      +tag+`</div>`
      +`<div class="bar"><i style="width:${pct}%;background:${col}"></i></div></div>`; };
  el.innerHTML=head+`<div style="display:flex;gap:16px;flex-wrap:wrap">`
    +syms.map(sym=>{ const a=bySym[sym]; const gs=(a.gauges||[]).filter(g=>!g.shadow);
      return `<div style="flex:1;min-width:240px;border:1px solid #2a3346;border-radius:8px;padding:10px 12px">`
        +`<div style="font-weight:600;color:#fff">${esc(sym)} `
        +`<span class="muted" style="font-weight:400;font-size:12px">${esc(a.regime||'')} · ER ${f(a.efficiency_ratio,3)}</span></div>`
        +(gs.length?gs.map(gbar).join(''):'<div class="muted" style="margin-top:6px">no gauges (skill not reporting)</div>')
        +`<div class="sub" style="margin-top:6px">decision: <b>${esc(a.would_act||'HOLD')}</b> · ${ago(a.ts)}</div>`
        +`</div>`; }).join('')+`</div>`;
}
function renderShadow(d){
  const el=document.getElementById('shadow'); if(!el) return;
  const acts=(d.activity||[]).filter(a=>a.kind==='pulse').slice()
    .sort((x,y)=>new Date(y.ts)-new Date(x.ts));
  const bySym={}; for(const a of acts){ if(!bySym[a.symbol]) bySym[a.symbol]=a; }
  const syms=(d.universe||[]).filter(s=>bySym[s] && (bySym[s].gauges||[]).some(g=>g.shadow));
  const head=`<div class="sub" style="margin-bottom:10px">Confirmed slow-orthogonal edges on <b>new data</b> (options-implied vol · on-chain stablecoin supply). Computed every pulse but <b>never routed to execution</b> — kept out of the live slate, here only to accrue out-of-sample evidence before any promotion. BTC/ETH only.</div>`;
  if(!syms.length){ el.innerHTML=head+'<div class="muted">no shadow gauge logged yet — waiting for the next pulse</div>'; return; }
  const sbar=g=>{ const pct=Math.max(4,Math.min(100,100*(Math.abs(Number(g.val))/(g.thr||1))));
    const tag=g.armed?`<span class="pill" style="color:#cba6f7;border-color:#4a3a6a">would ${esc(g.side||'fire')}</span>`
      :'<span class="pill off">watching</span>';
    return `<div style="margin:6px 0"><div class="sub">${esc(g.sleeve)} — ${esc(g.metric)} <b>${f(g.val,2)}</b> / ±${f(g.thr,2)} `
      +tag+`</div><div class="bar"><i style="width:${pct}%;background:#cba6f7"></i></div></div>`; };
  el.innerHTML=head+`<div style="display:flex;gap:16px;flex-wrap:wrap">`
    +syms.map(sym=>{ const a=bySym[sym]; const gs=(a.gauges||[]).filter(g=>g.shadow);
      return `<div style="flex:1;min-width:240px;border:1px solid #4a3a6a;border-radius:8px;padding:10px 12px">`
        +`<div style="font-weight:600;color:#fff">${esc(sym)} <span class="pill off">shadow</span></div>`
        +gs.map(sbar).join('')
        +`<div class="sub" style="margin-top:6px">log-only · never trades · ${ago(a.ts)}</div>`
        +`</div>`; }).join('')+`</div>`;
}
function renderAB(st){
  const card=s=>{ if(!s) return '';
    const L=s.live||{};
    const badge=s.status==='live'?'<span class="pill ok">live</span>':`<span class="pill off">${esc(s.status||'research')}</span>`;
    let h=`<div style="flex:1;min-width:260px${s.status==='retired'?';opacity:.7':''}"><div style="font-weight:600;color:#fff;margin-bottom:6px">${esc(s.label)} ${badge}</div>`
      +`<table><tr><th>${s.status==='retired'?'trades historiques':'live trades'}</th><td>${L.n||0}</td></tr>`
      +`<tr><th>win</th><td>${L.win_rate==null?'—':(100*L.win_rate).toFixed(0)+'%'}</td></tr>`
      +`<tr><th>PnL</th><td class="${cls(L.pnl_usd)}">$${f(L.pnl_usd,3)}</td></tr></table>`;
    if(s.skills){ h+=`<div class="sub" style="margin-top:4px">${s.skills.map(esc).join(', ')}</div>`; }
    if(s.research){ const r=s.research;
      h+=`<div class="sub" style="margin-top:6px">`
        +(r.oos_validated_bricks!=null?`research: ${r.oos_validated_bricks} OOS-validated bricks / ${r.candidates_tested||0} tested<br>`:'')
        +`<span class="muted">${esc(r.note||'')}</span></div>`; }
    return h+`</div>`; };
  document.getElementById('ab').innerHTML=
    `<div style="display:flex;gap:20px;flex-wrap:wrap">`
    +card(st.strategy_1)+card(st.strategy_2)+card(st.strategy_3)+card(st.retired_skills)+`</div>`;
}
function renderRestructuration(d){
  const el=document.getElementById('restructuration'); if(!el) return;
  const r=d.restructuration||{}, fwd=d.forward||{}, gov=d.governance||{};
  const fluxRow=(name,label)=>{ const x=fwd[name]||{};
    const bySym=x.by_symbol?Object.entries(x.by_symbol).map(([k,v])=>`${esc(k)}:${v}`).join(' · '):'';
    return `<tr><td>${esc(label)}</td><td class="muted">${esc(x.status||'no_data')}</td>`
      +`<td>${x.rows||0}</td><td class="muted">${x.last_ts?ago(x.last_ts):'—'}${bySym?' · '+bySym:''}</td></tr>`; };
  const gate=r.gate||{};
  const gt=gate.edge_paper_trades_target||40, go=gate.edge_paper_trades_observed||0;
  const gpct=Math.min(100,100*go/(gt||1));
  const retired=(r.retired||[]).map(x=>`${esc(x.skill_id)} (LOO-Δ Sharpe ${f(x.loo_delta_sharpe,3)}, joint-null p=${f(x.joint_null_p,3)})`).join(', ');
  el.innerHTML=
    `<div style="display:flex;gap:20px;flex-wrap:wrap">`
    +`<div style="flex:1;min-width:260px">`
      +`<div class="sub" style="margin-bottom:6px"><b>Slate live</b>: ${(r.slate||[]).map(esc).join(', ')||'—'}</div>`
      +`<div class="sub" style="margin-bottom:6px"><b>Retiré</b>: ${retired||'—'}</div>`
      +`<div class="sub"><b>Gate stage_0</b>: edge_paper_trades ${go} / ${gt}</div>`
      +`<div class="bar"><i style="width:${gpct}%;background:#89b4fa"></i></div>`
    +`</div>`
    +`<div style="flex:1;min-width:280px">`
      +`<table><tr><th>flux forward</th><th>statut</th><th>lignes</th><th>dernier / détail</th></tr>`
      +fluxRow('top_lsr','top_lsr')+fluxRow('skew','skew 25Δ')
      +fluxRow('funding_squeeze','funding_squeeze shadow')+fluxRow('comex_curve','comex')
      +`</table>`
      +`<div class="sub" style="margin-top:6px">gouvernance: trials_by_book ${gov.trials_by_book&&gov.trials_by_book.counts?JSON.stringify(gov.trials_by_book.counts):'—'}</div>`
    +`</div>`
    +`<div style="flex:1;min-width:240px">`
      +`<div class="sub" style="margin-bottom:6px"><b>Échéances</b></div>`
      +`<div class="sub">Métaux (XAU/XAG) collect-now → re-test <b>${esc((r.metals_collect_now||{}).retest_deadline||'—')}</b></div>`
      +`<div class="sub">FX ${esc((r.fx||{}).status||'—')} → vérif <b>${esc((r.fx||{}).reopen_check||'—')}</b></div>`
      +`<div class="sub">Couperet global narratif « revenu » → <b>${esc(r.global_cutoff||'—')}</b></div>`
      +`<div class="sub" style="margin-top:6px"><a href="/${esc(r.dossier||'')}">${esc(r.dossier||'')}</a></div>`
    +`</div>`
    +`</div>`;
}
function render(){
  const d=D; if(!d||!d.server_time) return;
  document.getElementById('stage').textContent=d.stage;
  document.getElementById('clock').textContent='server '+new Date(d.server_time).toUTCString();
  document.getElementById('equity').textContent='$'+f(d.equity_usd,2);
  document.getElementById('bankroll').textContent='bankroll $'+f(d.bankroll_usd,0)+' · '+(d.universe||[]).join(', ');
  document.getElementById('gatesum').textContent=d.stage_summary||'';
  document.getElementById('gates').innerHTML=Object.entries(d.gates||{}).map(([k,v])=>{
    const pct=k.includes('trades')?Math.min(100,100*(v.observed||0)/(v.target||1)):(v.passed?100:8);
    return `<div style="margin:8px 0"><div class="sub">${esc(k)}: ${v.observed} / ${v.target} ${v.passed?'✓':'✗'}</div>`
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
  renderLive(d);
  renderShadow(d);
  renderEq(d.equity_curve);
  renderAB(d.strategies||{});
  renderRestructuration(d);
  const ps=d.open_positions||[];
  document.getElementById('positions').innerHTML=ps.length?
    `<table><tr><th>symbol</th><th>qty</th><th>entry</th><th>uPnL</th><th>stop</th><th>tp</th><th>skill</th></tr>`
    +ps.map(p=>`<tr><td>${esc(p.symbol)}</td><td>${f(p.qty,4)}</td><td>${f(p.entry_price,2)}</td>`
      +`<td class="${cls(p.unrealized_pnl_usd)}">$${f(p.unrealized_pnl_usd,3)}</td>`
      +`<td class="muted">${f(p.stop_price,2)}</td><td class="muted">${f(p.take_profit,2)}</td>`
      +`<td class="muted">${esc(p.skill_id)}</td></tr>`).join('')+`</table>`:'<div class="muted">none</div>';
  const tr=d.recent_closed||[];
  document.getElementById('trades').innerHTML=tr.length?
    `<table><tr><th>when</th><th>symbol</th><th>side</th><th>PnL</th><th>reason</th><th>skill</th><th>conf</th></tr>`
    +tr.map(t=>`<tr><td class="muted">${ago(t.closed_at)}</td><td>${esc(t.symbol)}</td><td>${esc(t.side)}</td>`
      +`<td class="${cls(t.pnl_usd)}">$${f(t.pnl_usd,3)}</td><td class="muted">${esc(t.exit_reason)}</td>`
      +`<td class="muted">${esc(t.skill_id)}</td><td class="muted">${f(t.confidence,2)}</td></tr>`).join('')+`</table>`
    :'<div class="muted">no closed trades yet</div>';
  let ac=(d.activity||[]).slice();
  if(document.getElementById('hidehold').checked) ac=ac.filter(a=>!(a.kind==='pulse'&&a.would_act==='HOLD'));
  const flt=(document.getElementById('actfilter').value||'').toLowerCase();
  if(flt) ac=ac.filter(a=>JSON.stringify(a).toLowerCase().includes(flt));
  document.getElementById('activity').innerHTML=ac.length?
    `<table><tr><th>when</th><th>kind</th><th>symbol</th><th>regime</th><th>signal / decision</th><th>reason</th></tr>`
    +ac.map(a=>{
      const act=a.kind==='pulse'
        ? (a.would_act==='HOLD'?'HOLD':esc(a.would_act+' '+(a.would_skill||'')))
        : esc((a.executed?'EXEC ':'')+(a.side||'')+(a.decision?(' / '+a.decision):''));
      return `<tr><td class="muted">${ago(a.ts)}</td><td>${esc(a.kind)}</td><td>${esc(a.symbol)}</td>`
        +`<td class="muted">${esc(a.regime)}</td><td>${act}</td><td class="muted">${esc(a.reason)}</td></tr>`;
    }).join('')+`</table>`
    :'<div class="muted">no activity matches</div>';
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
    +sc.map(s=>`<tr><td>${esc(s.id)}</td><td class="muted">${until(s.next_run)} <span class="muted">(${s.next_run?new Date(s.next_run).toUTCString():'—'})</span></td></tr>`).join('')+`</table>`
    :'<div class="muted">—</div>';
}
tick();setInterval(tick,20000);
</script></body></html>"""


_CHAT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Hermès — chat</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#0b0e14;color:#cdd6f4;font:15px/1.5 system-ui,sans-serif;
    display:flex;flex-direction:column;height:100vh}
  header{padding:12px 18px;border-bottom:1px solid #1d2230;display:flex;gap:12px;align-items:center}
  header h1{font-size:18px;margin:0}.sub{color:#6c7693;font-size:13px}
  a{color:#89b4fa}
  #log{flex:1;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:12px;max-width:820px;width:100%;margin:0 auto;box-sizing:border-box}
  .msg{padding:10px 14px;border-radius:12px;max-width:85%;white-space:pre-wrap;word-wrap:break-word}
  .me{align-self:flex-end;background:#1e2a44;border:1px solid #2a3a5e}
  .bot{align-self:flex-start;background:#11161f;border:1px solid #1d2230}
  .meta{color:#6c7693;font-size:11px;margin-top:4px}
  form{display:flex;gap:8px;padding:12px 18px;border-top:1px solid #1d2230;max-width:820px;width:100%;margin:0 auto;box-sizing:border-box}
  input,textarea{background:#0b0e14;border:1px solid #2a3346;color:#cdd6f4;border-radius:8px;padding:9px 11px;font:15px sans-serif}
  #msg{flex:1;resize:none}
  button{background:#89b4fa;color:#0b0e14;border:0;border-radius:8px;padding:0 16px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
</style></head>
<body>
<header><h1>Hermès</h1><span class="sub">parle avec l'agent de notre entreprise</span>
  <a href="/dashboard" style="margin-left:auto">← dashboard</a></header>
<div id="log"></div>
<form id="f">
  <textarea id="msg" rows="2" placeholder="Demande-lui où en est l'entreprise, pourquoi il ne trade pas, sa stratégie…"></textarea>
  <button id="send" type="submit">envoyer</button>
</form>
<script>
const log=document.getElementById('log');
const hist=[];
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function add(who,text,meta){const d=document.createElement('div');d.className='msg '+(who==='me'?'me':'bot');
  d.innerHTML=esc(text)+(meta?`<div class="meta">${esc(meta)}</div>`:'');log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
add('bot',"Salut. Je suis Hermès. Demande-moi où en est notre entreprise, ce que voient les stratégies, ou pourquoi je tiens ma position. (Je discute seulement — je n'exécute aucune action ici.)");
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
  const m=document.getElementById('msg').value.trim();
  if(!m)return;
  add('me',m); document.getElementById('msg').value=''; const btn=document.getElementById('send'); btn.disabled=true;
  const think=add('bot','…');
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({message:m,history:hist.slice(-10)})});
    const j=await r.json();
    if(!r.ok){think.textContent='Erreur: '+(j.detail||r.status);}
    else{think.innerHTML=esc(j.reply)+`<div class="meta">${esc(j.model||'')}</div>`;
      hist.push({role:'user',content:m}); hist.push({role:'assistant',content:j.reply});}
  }catch(err){think.textContent='Erreur réseau: '+err;}
  finally{btn.disabled=false; document.getElementById('msg').focus();}
};
document.getElementById('msg').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit();}});
</script></body></html>"""
