"""Stream 7 — central-bank tone DRIFT (not the print).

Two layers, kept apart on purpose:

* **The classifier** (``ToneScore`` / ``parse_tone`` / ``classify_one`` / ``classify_via_router``):
  the reusable, LLM-native hawk↔dove scorer. The LLM is *injected* (``score_fn``) so it is unit
  testable offline; ``classify_via_router`` wires the real ``hermes.llm.router`` in production. The
  scorer is run on a **date-blinded** excerpt (date + speaker masked) — a deliberate mitigation of
  the lab's hardening principle #1 ("NLP tone must never be a *future-knowing* pretrained model").
  A frontier LLM still carries residual era awareness, so a *pass* of the gate is only provisional;
  a *fail* is robust (look-ahead bias inflates apparent edge, so a contaminated scorer that still
  fails is doubly dead).

* **The signal** (``daily_tone`` → ``ewma_drift`` → ``causal_z`` → ``tone_events`` → ``drop_blackout``):
  pure, causal functions that turn per-speech scores into a slow per-bank drift and then into
  non-overlapping entry events that are **never held through a policy print**. Everything is
  strictly backward-looking; ``causal_z`` is the keystone (no look-ahead). The realised gate is
  ``research_fx/fxgate.event_basket_gate`` (date-level joint MC null) — the same machinery that
  re-killed E3 — wired by the analysis runner, not here.

Targets live in the frozen lake: ``gold``/``EURUSD``/``USDJPY`` (the USD/gold price channel — note
EUR+JPY ≈ one DXY bet, the colinear-channel rule) and ``fred DGS10`` (the rates channel, traded via
``bond_proxy``). BoE has no GBP/UK-rates target in the lake → a documented data lock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "HAWK_DOVE_SCHEMA",
    "ToneScore",
    "bond_proxy",
    "build_messages",
    "causal_z",
    "classify_one",
    "classify_via_router",
    "daily_tone",
    "drop_blackout",
    "ewma_drift",
    "fed_market_disagreement",
    "parse_tone",
    "tone_events",
]

_DAY_MS = 86_400_000
_STANCES = ("hawkish", "dovish", "neutral")


# --------------------------------------------------------------------------- #
# classifier (LLM injected)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToneScore:
    """One speech's hawk↔dove score. ``score`` in [-1 dove, +1 hawk]; ``confidence`` in [0, 1]."""

    id: str
    score: float
    confidence: float
    stance: str


HAWK_DOVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": -1, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "stance": {"type": "string", "enum": list(_STANCES)},
    },
    "required": ["score", "confidence", "stance"],
}

_SYSTEM = (
    "You are a central-bank monetary-policy tone classifier. Score the HAWK<->DOVE stance expressed "
    "in the speech on a continuous scale: +1 = maximally hawkish (inflation/overheating risk, tighten, "
    "raise rates, restrictive, vigilant); 0 = balanced/two-handed or non-monetary (regulation, payments, "
    "supervision); -1 = maximally dovish (downside/growth risk, ease, cut, accommodation, patience, slack). "
    "Judge ONLY the policy lean in THIS text — never use outside knowledge of what happened afterward. "
    "Dates and the speaker name are masked ([Y]/[SPK]) on purpose. confidence is low (<=0.3) for "
    "non-monetary/two-handed speeches, high (>=0.7) when the lean is explicit."
)


def build_messages(excerpt: str) -> list[dict[str, str]]:
    """The reusable chat prompt for the router's ``structured`` JSON-mode call."""
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Classify the monetary-policy tone of this central-bank speech excerpt. Return JSON "
                '{"score": <float -1..1>, "confidence": <float 0..1>, "stance": '
                '"hawkish"|"dovish"|"neutral"}.\n\n' + excerpt
            ),
        },
    ]


def _clamp(x: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def parse_tone(raw: dict[str, Any], id: str) -> ToneScore:
    """Validate one LLM output into a ``ToneScore`` — clamped and FAIL-SOFT.

    A missing/garbage score collapses to a neutral 0.0 with zero confidence, so a single bad LLM
    response never poisons the drift (it simply contributes nothing).
    """
    try:
        score = _clamp(raw["score"], -1.0, 1.0)
    except (KeyError, TypeError, ValueError):
        return ToneScore(id, 0.0, 0.0, "neutral")
    try:
        confidence = _clamp(raw.get("confidence", 0.0), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    stance = raw.get("stance")
    if stance not in _STANCES:
        stance = "neutral"
    return ToneScore(id, score, confidence, stance)


def classify_one(excerpt: str, id: str, *, score_fn: Callable[[str], dict[str, Any]]) -> ToneScore:
    """Classify one excerpt with an injected scorer ``score_fn(excerpt) -> dict`` (testable offline)."""
    return parse_tone(score_fn(excerpt), id)


async def classify_via_router(excerpt: str, id: str, *, agent_role: str = "research") -> ToneScore:
    """Production scorer: route the excerpt through the Hermès LLM brain (``hermes.llm.router``).

    Degrades gracefully (the router already falls back json_schema -> json_object -> none and
    extracts the first JSON block), and ``parse_tone`` fail-softs anything malformed to neutral.
    """
    from hermes.llm.router import get_router

    out = await get_router().structured(
        agent_role=agent_role, messages=build_messages(excerpt), schema=HAWK_DOVE_SCHEMA
    )
    return parse_tone(out or {}, id)


# --------------------------------------------------------------------------- #
# signal (pure, causal)
# --------------------------------------------------------------------------- #
def daily_tone(scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse all speeches on a (bank, day) to ONE confidence-weighted mean tone.

    ``scores`` needs columns date, bank, score, confidence. Output columns: date (Timestamp,
    normalised), bank, tone, n. When a day's confidences are all zero we fall back to the simple
    mean so the speech is not silently dropped.
    """
    cols = ["date", "bank", "tone", "n"]
    if scores is None or len(scores) == 0:
        return pd.DataFrame(columns=cols)
    df = scores.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    def _agg(g: pd.DataFrame) -> float:
        w = g["confidence"].to_numpy()
        s = g["score"].to_numpy()
        return float(np.average(s, weights=w)) if w.sum() > 0 else float(s.mean())

    rows = [
        {"date": d, "bank": b, "tone": _agg(g), "n": len(g)}
        for (b, d), g in df.groupby(["bank", "date"])
    ]
    return pd.DataFrame(rows).sort_values(["bank", "date"]).reset_index(drop=True)


def ewma_drift(tone: pd.Series, halflife_days: float) -> pd.Series:
    """Time-aware EWMA of a datetime-indexed tone series = the slow per-bank DRIFT.

    Uses the real elapsed time between speeches (``times=``), so a quiet stretch decays the drift
    just as wall-clock would — the honest treatment of an irregular speech calendar.
    """
    s = tone.sort_index()
    return s.ewm(halflife=pd.Timedelta(days=halflife_days), times=s.index).mean()


def causal_z(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Trailing standardised drift: z_t = (x_t - mean) / std over the last ``window`` observations.

    STRICTLY backward-looking — the window at t ends at t, so z_t cannot move when any future value
    changes (the look-ahead guard). NaN until ``min_obs`` observations exist, or when the trailing
    std is degenerate.
    """

    def _z(w: np.ndarray) -> float:
        sd = w.std(ddof=1)
        return (w[-1] - w.mean()) / sd if sd > 0 else np.nan

    return series.rolling(window=window, min_periods=min_obs).apply(_z, raw=True)


def tone_events(z: pd.Series, thr: float) -> list[tuple[int, int]]:
    """Fire (entry_date_ms, direction) whenever |z| >= ``thr``. direction = sign(z): +1 hawkish drift,
    -1 dovish drift. NaN never fires."""
    out: list[tuple[int, int]] = []
    for ts, val in z.items():
        if pd.isna(val) or abs(val) < thr:
            continue
        ms = int(pd.Timestamp(ts).value // 1_000_000)
        out.append((ms, 1 if val > 0 else -1))
    return out


def drop_blackout(
    events: list[tuple[int, int]],
    meeting_ms: list[int],
    horizon_days: int,
    pre_days: int = 0,
) -> list[tuple[int, int]]:
    """Drop any entry whose [entry - pre_days, entry + horizon_days] window contains a policy print.

    This is the hard invariant: we never *hold a position through* a meeting/decision (the $12k
    news-blackout lesson). ``meeting_ms`` are decision-day timestamps (FOMC from the lake; ECB/BoE
    from the known schedule). ``pre_days`` optionally also excludes the pre-meeting purdah window.
    """
    meets = sorted(int(m) for m in meeting_ms)
    if not meets:
        return list(events)
    kept: list[tuple[int, int]] = []
    for ts, d in events:
        lo = ts - pre_days * _DAY_MS
        hi = ts + horizon_days * _DAY_MS
        if any(lo <= m <= hi for m in meets):
            continue
        kept.append((ts, d))
    return kept


def bond_proxy(yield_pct: pd.Series) -> pd.Series:
    """A monotone 'price' for the rates channel: ``100 - yield``. Hawkish drift -> yields up ->
    proxy down, so the gate's ratio-return machinery scores the SIGN of the rates response. This is
    a sign-test proxy, NOT a tradable bond return (no DV01/convexity)."""
    return 100.0 - pd.to_numeric(yield_pct, errors="coerce")


# --------------------------------------------------------------------------- #
# 7.3 conditioner — dot-plot vs Fed-funds-futures disagreement (DATA-TO-COLLECT)
# --------------------------------------------------------------------------- #
def fed_market_disagreement(dot_median: pd.Series, ff_implied: pd.Series) -> pd.Series:
    """Slow exogenous macro state: ``SEP dot-plot median - Fed-funds-futures-implied rate`` per
    horizon/date. A POSITIVE gap = the Fed projects HIGHER rates than the market prices (the Fed is
    more hawkish than the curve) — policy-path *uncertainty*, not price-on-price.

    Pure and aligned on the shared index; NaN wherever either side is missing. This is a CONDITIONER
    only (interaction / shuffle-null in Stream 8), never a standalone signal. Both inputs are absent
    from the frozen lake (no SEP, no FF futures) — see research_streams/S7/features/ for the
    collection plan; this function is the ready-to-wire formula.
    """
    a, b = dot_median.align(ff_implied, join="outer")
    return a - b
