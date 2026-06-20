"""Meta-labeling (shadow): a model that predicts P(win) from the decision-time
feature snapshot, to LATER filter the LLM's trades. Run in shadow (logged, never
gating) until it proves out. Identity/None until enough two-class data exists.
"""

from hermes.risk.meta_labeling import MetaLabeler, meta_features


def test_meta_features_extracts_fixed_numeric_schema():
    snap = {
        "atr_pct": 0.02, "ret_1": 0.01, "regime": "trend_up",
        "regime_efficiency_ratio": 0.8, "vol_last": 12.0, "vol_avg_20": 10.0,
    }
    f = meta_features(snap, confidence=0.7)
    assert f["confidence"] == 0.7
    assert f["atr_pct"] == 0.02
    assert f["regime_efficiency_ratio"] == 0.8
    assert f["vol_ratio"] == 1.2
    assert f["regime_trend_up"] == 1.0
    assert f["regime_range"] == 0.0


def test_unfitted_returns_none():
    assert MetaLabeler().is_fitted is False
    assert MetaLabeler().predict_proba({"x": 1.0}) is None


def test_insufficient_or_single_class_stays_unfitted():
    feats = [{"x": 1.0} for _ in range(10)]
    assert MetaLabeler(min_samples=50).fit(feats, [1, 0] * 5).is_fitted is False  # too few
    feats2 = [{"x": 1.0} for _ in range(60)]
    assert MetaLabeler(min_samples=50).fit(feats2, [1] * 60).is_fitted is False   # one class


def test_learns_separating_feature():
    feats = [{"x": 1.0}] * 30 + [{"x": 0.0}] * 30
    outcomes = ([1] * 25 + [0] * 5) + ([1] * 5 + [0] * 25)  # x=1 mostly wins, x=0 mostly loses
    ml = MetaLabeler(min_samples=50).fit(feats, outcomes)
    assert ml.is_fitted is True
    p_hi, p_lo = ml.predict_proba({"x": 1.0}), ml.predict_proba({"x": 0.0})
    assert 0.0 <= p_lo <= 1.0 and 0.0 <= p_hi <= 1.0
    assert p_hi > p_lo
