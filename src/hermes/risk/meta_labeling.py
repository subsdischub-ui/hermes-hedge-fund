"""Meta-labeling (López de Prado): a secondary model that predicts P(win) from the
decision-time feature snapshot, used to FILTER the primary (LLM) signal.

Run in SHADOW first — logged alongside every decision, never gating — until it
demonstrates value out-of-sample. Identity (None) until enough two-class trades
exist, so it can never silently start blocking trades on noise.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.linear_model import LogisticRegression

_REGIMES = ("trend_up", "trend_down", "range")


def meta_features(snapshot: dict, confidence: float) -> dict[str, float]:
    """Build a fixed numeric feature vector from a decision-time snapshot."""
    vol_avg = float(snapshot.get("vol_avg_20", 0.0) or 0.0)
    vol_last = float(snapshot.get("vol_last", 0.0) or 0.0)
    vol_ratio = (vol_last / vol_avg) if vol_avg > 0 else 1.0
    regime = str(snapshot.get("regime", "range"))
    feats = {
        "confidence": float(confidence),
        "atr_pct": float(snapshot.get("atr_pct", 0.0) or 0.0),
        "ret_1": float(snapshot.get("ret_1", 0.0) or 0.0),
        "regime_efficiency_ratio": float(snapshot.get("regime_efficiency_ratio", 0.0) or 0.0),
        "vol_ratio": vol_ratio,
    }
    for r in _REGIMES:
        feats[f"regime_{r}"] = 1.0 if regime == r else 0.0
    return feats


class MetaLabeler:
    def __init__(self, min_samples: int = 50):
        self.min_samples = min_samples
        self._model: LogisticRegression | None = None
        self._feature_names: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self, feature_dicts: Sequence[dict], outcomes: Sequence[int]
    ) -> MetaLabeler:
        if len(feature_dicts) < self.min_samples or len(set(outcomes)) < 2:
            self._model = None
            return self
        names = sorted({k for d in feature_dicts for k in d})
        x = [[float(d.get(k, 0.0)) for k in names] for d in feature_dicts]
        model = LogisticRegression(max_iter=1000)
        model.fit(x, list(outcomes))
        self._model = model
        self._feature_names = names
        return self

    def predict_proba(self, features: dict) -> float | None:
        if self._model is None or self._feature_names is None:
            return None
        x = [[float(features.get(k, 0.0)) for k in self._feature_names]]
        return float(self._model.predict_proba(x)[0][1])


# --- process-cached loader (fit from the journal; refit nightly) ------------
_CACHE: MetaLabeler | None = None


async def get_meta_labeler(min_samples: int = 50) -> MetaLabeler:
    """Lazily fit and cache a meta-labeler from the journal's (snapshot, win)
    history. Unfitted (shadow no-op) until min_samples two-class trades exist."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    from sqlalchemy import select

    from hermes.memory.store import JournalRow, get_db

    db = get_db()
    async with db.session() as s:
        rows = (
            await s.execute(
                select(JournalRow).where(
                    JournalRow.closed_at.is_not(None),
                    JournalRow.pnl_usd.is_not(None),
                )
            )
        ).scalars().all()
    feats = [meta_features(r.market_state or {}, float(r.confidence or 0.0)) for r in rows]
    outs = [1 if (r.pnl_usd or 0) > 0 else 0 for r in rows]
    _CACHE = MetaLabeler(min_samples=min_samples).fit(feats, outs)
    return _CACHE


def reset_meta_labeler() -> None:
    global _CACHE
    _CACHE = None
