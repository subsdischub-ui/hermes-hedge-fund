"""Hourly pulse — a lightweight, LLM-free heartbeat that records what the agent
currently SEES and WOULD do, so the dashboard shows continuous activity between
the (4h-aligned) real decisions. Computes regime + deterministic skill signals +
the selective gate's verdict per symbol; no orders are placed here.
"""

from __future__ import annotations

from hermes.data import get_market
from hermes.data.regime import detect_regime
from hermes.execution import get_exchange
from hermes.logging_setup import get_logger
from hermes.risk.selective import SelectiveConfig, load_selective_config, selective_gate
from hermes.runtime.activity import ActivityLog
from hermes.settings import get_settings
from hermes.skills import get_skill_library
from hermes.skills.library import SkillSignal

log = get_logger(__name__)


def pulse_entry(
    symbol: str,
    regime: dict,
    signals: list[SkillSignal],
    open_symbols: set[str],
    cfg: SelectiveConfig,
) -> dict:
    """Pure summary of one symbol's current state for the activity feed."""
    surfaced = [
        {"skill": s.skill_id, "side": s.side, "conf": round(float(s.confidence), 2)}
        for s in signals
        if s.side != "HOLD"
    ]
    would_act, would_skill, reason = "HOLD", "", "no actionable signal"
    for s in signals:
        if s.side == "HOLD":
            continue
        ok, why = selective_gate(
            s.side, symbol, s.skill_id,
            [{"skill_id": s.skill_id, "confidence_raw": float(s.confidence)}],
            open_symbols, cfg,
        )
        if ok:
            would_act, would_skill, reason = s.side, s.skill_id, "would enter (passes selective gate)"
            break
        reason = why  # remember the last rejection reason
    return {
        "kind": "pulse",
        "symbol": symbol,
        "regime": regime.get("regime"),
        "efficiency_ratio": round(float(regime.get("efficiency_ratio", 0.0)), 3),
        "signals": surfaced,
        "would_act": would_act,
        "would_skill": would_skill,
        "reason": reason,
    }


async def pulse() -> int:
    """Record a per-symbol signal/regime snapshot to the activity feed. No LLM, no
    orders — safe to run frequently and independent of the LLM being reachable."""
    s = get_settings()
    tf = str(s.yaml("strategy", "decision_timeframe", default="4h") or "4h")
    cfg = load_selective_config(s)
    lib = get_skill_library()
    activity = ActivityLog(s.data_dir / "activity.json")
    md = get_market()
    try:
        open_symbols = {p.symbol for p in await get_exchange().open_positions()}
    except Exception:
        open_symbols = set()
    n = 0
    for sym in s.universe:
        try:
            df = await md.ohlcv(sym, timeframe=tf, limit=300)
            regime = detect_regime(df)
            sigs = lib.evaluate_for_symbol(sym, df)
            activity.record(pulse_entry(sym, regime, sigs, open_symbols, cfg))
            n += 1
        except Exception as e:
            log.warning("pulse_failed", symbol=sym, error=str(e))
    await md.close()
    if n:
        log.info("pulse", symbols=n)
    return n
