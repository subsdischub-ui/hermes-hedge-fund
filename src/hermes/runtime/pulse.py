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
    open_keys: set[tuple[str, str]],
    cfg: SelectiveConfig,
) -> dict:
    """Pure summary of one symbol's current state for the activity feed."""
    surfaced = [
        {"skill": s.skill_id, "side": s.side, "conf": round(float(s.confidence), 2)}
        for s in signals
        if s.side != "HOLD"
    ]
    # Per-sleeve gauges shown live even while HOLDing, so the dashboard always
    # answers "what is each strategy seeing and how far is it from firing?".
    gauges = []
    for s in signals:
        if s.skill_id == "ts_mom_vol_scaled":
            # RESTRUCTURATION 2026-07-02: ts_mom_vol_scaled was retired from
            # strategy.allowed_skills (bull-beta drag, book LOO-Δ Sharpe -0.146,
            # joint-null p=0.69 — docs/RESTRUCTURATION_2026-07.md §3). The skill
            # library still computes this signal (so the gauge keeps showing what
            # it SEES for transparency), but selective_gate now rejects it
            # unconditionally — it can NEVER win would_act. Force armed=False and
            # flag `retired` so the dashboard renders this as a dead gauge instead
            # of implying it could still fire.
            v = round(float(s.confidence), 2)
            gauges.append({"sleeve": "Momentum #1", "metric": "conviction",
                           "val": v, "thr": 1.0, "armed": False, "retired": True,
                           "status": "retired (hors slate depuis 2026-07-02)"})
        elif s.skill_id == "funding_squeeze_continuation":
            fz = round(float(s.features.get("funding_z", 0.0)), 2)
            gauges.append({"sleeve": "Funding #2", "metric": "funding z",
                           "val": fz, "thr": 2.0, "armed": fz >= 2.0})
        elif s.skill_id == "funding_carry_harvest":
            bps = round(float(s.features.get("funding_bps", 0.0)), 2)
            arm = float(s.features.get("arm_bps", 2.0))
            # Now executable as a delta-neutral pair (short perp + long spot): arms
            # like the others when funding clears the threshold.
            gauges.append({"sleeve": "Carry #3 (delta-neutral)", "metric": "funding bps",
                           "val": bps, "thr": arm, "armed": bps >= arm})
    would_act, would_skill, reason = "HOLD", "", "no actionable signal"
    for s in signals:
        if s.side == "HOLD":
            continue
        ok, why = selective_gate(
            s.side, symbol, s.skill_id,
            [{"skill_id": s.skill_id, "confidence_raw": float(s.confidence)}],
            open_keys, cfg,
        )
        if ok:
            would_act, would_skill, reason = s.side, s.skill_id, "would enter (passes selective gate)"
            break
        reason = why  # remember the last rejection reason

    # SHADOW gauges (read-only): skills registered in the library but NOT in the live
    # slate (allowed_skills). They are computed + displayed every tick so the dashboard
    # shows them in a separate panel, but they are excluded from would_act BY
    # CONSTRUCTION — selective_gate rejects any skill not in the slate (see
    # test_selective_gate_rejects_shadow_skill_even_with_strong_buy), so this loop only
    # ADDS display gauges and never touches the would_act decision computed above.
    allowed = set(cfg.allowed_skills)
    for s in signals:
        if s.skill_id in allowed or s.side == "HOLD":
            continue
        gauges.append({
            "shadow": True,
            "skill": s.skill_id,
            "side": s.side,
            "conf": round(float(s.confidence), 2),
            "armed": True,  # a non-HOLD shadow signal is firing (size 0, log-only)
        })
    return {
        "kind": "pulse",
        "symbol": symbol,
        "regime": regime.get("regime"),
        "efficiency_ratio": round(float(regime.get("efficiency_ratio", 0.0)), 3),
        "signals": surfaced,
        "gauges": gauges,
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
        open_keys = {(p.symbol, p.skill_id) for p in await get_exchange().open_positions()}
    except Exception:
        open_keys = set()
    n = 0
    for sym in s.universe:
        try:
            df = await md.ohlcv(sym, timeframe=tf, limit=300)
            # Attach funding so the funding_squeeze_continuation sleeve is visible in
            # the pulse (matches what the live decision tick sees).
            try:
                from hermes.research.funding import attach_funding

                fdf = await md.funding_history(sym)
                if not fdf.empty:
                    df = attach_funding(df, fdf)
            except Exception as e:
                log.warning("pulse_funding_attach_failed", symbol=sym, error=str(e))
            regime = detect_regime(df)
            sigs = lib.evaluate_for_symbol(sym, df)
            activity.record(pulse_entry(sym, regime, sigs, open_keys, cfg))
            n += 1
        except Exception as e:
            log.warning("pulse_failed", symbol=sym, error=str(e))
    await md.close()
    if n:
        log.info("pulse", symbols=n)
    return n
