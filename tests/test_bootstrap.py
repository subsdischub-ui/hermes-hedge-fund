"""Bootstrap the live decision models (confidence calibrator + meta-labeler) from
backtest samples and persist them, so the agent starts with calibrated models
instead of waiting weeks for live trades."""

import joblib

from hermes.validation.backtest import BacktestResult
from hermes.validation.bootstrap import bootstrap_models


def _samples():
    s = []
    for i in range(30):
        s.append({"confidence": 0.9, "outcome": 1, "features": {"x": 1.0},
                  "ret_net": 0.02, "skill_id": "a", "seq": i})
    for i in range(30):
        s.append({"confidence": 0.3, "outcome": 0, "features": {"x": 0.0},
                  "ret_net": -0.02, "skill_id": "a", "seq": 30 + i})
    return s


def test_bootstrap_fits_and_persists(tmp_path):
    res = BacktestResult(samples=_samples(), returns_by_skill={"a": [0.02] * 30 + [-0.02] * 30})
    out = bootstrap_models(res, tmp_path, min_samples=50)

    assert out["calibrator_fitted"] is True
    assert out["meta_fitted"] is True
    assert (tmp_path / "calibrator.pkl").exists()
    assert (tmp_path / "meta_labeler.pkl").exists()

    cal = joblib.load(tmp_path / "calibrator.pkl")
    assert cal.calibrate(0.9) >= cal.calibrate(0.3)
    ml = joblib.load(tmp_path / "meta_labeler.pkl")
    assert ml.predict_proba({"x": 1.0}) > ml.predict_proba({"x": 0.0})
