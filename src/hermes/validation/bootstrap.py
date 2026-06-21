"""Bootstrap the live decision models from backtest samples.

Fits the confidence calibrator and the meta-labeler on the historical backtest
(state, action, outcome) triples and persists them, so the live agent starts with
calibrated models instead of waiting weeks for live trades to accumulate.

SECURITY: the persisted files are joblib pickles written by THIS process into the
private /data volume and re-loaded only from there — a trusted, self-produced
source, never a download. Do not point the loaders at externally-supplied pickles.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from hermes.risk.calibration import ConfidenceCalibrator
from hermes.risk.meta_labeling import MetaLabeler
from hermes.validation.backtest import BacktestResult

CALIBRATOR_FILE = "calibrator.pkl"
META_LABELER_FILE = "meta_labeler.pkl"


def bootstrap_models(
    result: BacktestResult,
    out_dir: Path,
    min_samples: int = 50,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confs = [float(s["confidence"]) for s in result.samples]
    outs = [int(s["outcome"]) for s in result.samples]
    feats = [s["features"] for s in result.samples]

    cal = ConfidenceCalibrator(min_samples=min_samples).fit(confs, outs)
    ml = MetaLabeler(min_samples=min_samples).fit(feats, outs)

    if cal.is_fitted:
        joblib.dump(cal, out_dir / CALIBRATOR_FILE)
    if ml.is_fitted:
        joblib.dump(ml, out_dir / META_LABELER_FILE)

    return {
        "n": len(result.samples),
        "calibrator_fitted": cal.is_fitted,
        "meta_fitted": ml.is_fitted,
        "out_dir": str(out_dir),
    }
