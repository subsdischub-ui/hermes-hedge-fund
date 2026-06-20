"""Confidence calibration — map the LLM's self-reported confidence to a calibrated
P(win) learned from realized trade outcomes (isotonic regression).

The LLM is systematically over-confident (audit finding): a raw 0.8 may win far
less than 80% of the time, yet that same 0.8 feeds BOTH the decision gate and the
Kelly position size. Calibrating it before the risk engine makes the whole
downstream honest. This realises the role originally intended for the TRM verifier
(lowering over-confident proposals) with a simpler, inspectable, falsifiable model.

Isotonic regression is monotone-non-decreasing and clipped to [0, 1]: it only
re-orders/compresses confidences. We keep it as IDENTITY until there is enough
two-class data, so it can never INFLATE an unproven confidence — it only corrects
proven over-confidence downward.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ConfidenceCalibrator:
    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self._iso: IsotonicRegression | None = None

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    def fit(
        self, confidences: Sequence[float], outcomes: Sequence[int]
    ) -> ConfidenceCalibrator:
        x = np.asarray(list(confidences), dtype=float)
        y = np.asarray(list(outcomes), dtype=float)
        # Need enough data AND both win/loss classes present, else a calibrator
        # would be degenerate (e.g. predict 1.0 everywhere) — stay identity.
        if x.size < self.min_samples or np.unique(y).size < 2:
            self._iso = None
            return self
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x, y)
        self._iso = iso
        return self

    def calibrate(self, confidence: float) -> float:
        if self._iso is None:
            return float(confidence)
        return float(self._iso.predict([float(confidence)])[0])


# --- process-cached loader (fit from the journal; refit nightly) ------------
_CACHE: ConfidenceCalibrator | None = None


async def get_confidence_calibrator(min_samples: int = 30) -> ConfidenceCalibrator:
    """Lazily fit (and cache) a calibrator from the journal's (confidence, win)
    history. Identity until min_samples two-class trades exist. Cached for the
    process; call reset_confidence_calibrator() (the nightly job does) to refit."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    from hermes.memory.journal import TradeJournal

    confs, outs = await TradeJournal().confidence_outcomes()
    _CACHE = ConfidenceCalibrator(min_samples=min_samples).fit(confs, outs)
    return _CACHE


def reset_confidence_calibrator() -> None:
    """Drop the cached calibrator so the next call refits on fresh trade history."""
    global _CACHE
    _CACHE = None
