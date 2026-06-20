"""ConfidenceCalibrator: turn the LLM's (over-confident) self-reported confidence
into a calibrated P(win) learned from the journal, via isotonic regression.

Identity until enough data exists (shadow phase), so it can never *raise* an
unproven confidence — it only corrects proven over-confidence downward.
"""

from hermes.risk.calibration import ConfidenceCalibrator


def test_identity_when_unfitted():
    cal = ConfidenceCalibrator()
    assert cal.is_fitted is False
    assert cal.calibrate(0.8) == 0.8
    assert cal.calibrate(0.2) == 0.2


def test_does_not_fit_on_insufficient_samples():
    cal = ConfidenceCalibrator(min_samples=30)
    cal.fit([0.8] * 5, [1, 0, 1, 0, 1])
    assert cal.is_fitted is False
    assert cal.calibrate(0.8) == 0.8


def test_does_not_fit_when_only_one_outcome_class():
    # All wins -> cannot learn a mapping, must NOT inflate confidence to 1.0.
    cal = ConfidenceCalibrator(min_samples=10)
    cal.fit([0.7] * 40, [1] * 40)
    assert cal.is_fitted is False
    assert cal.calibrate(0.95) == 0.95


def test_overconfidence_is_corrected_downward():
    # Stated 0.9 confidence only wins ~50% of the time -> calibrated well below 0.9.
    confidences = [0.6] * 20 + [0.9] * 20
    outcomes = ([1] * 12 + [0] * 8) + ([1] * 10 + [0] * 10)
    cal = ConfidenceCalibrator(min_samples=20).fit(confidences, outcomes)
    assert cal.is_fitted is True
    assert cal.calibrate(0.9) < 0.7


def test_monotone_mapping_preserves_order_and_bounds():
    confidences = [0.5] * 20 + [0.9] * 20
    outcomes = ([1] * 6 + [0] * 14) + ([1] * 18 + [0] * 2)  # 0.3 vs 0.9 win rates
    cal = ConfidenceCalibrator(min_samples=20).fit(confidences, outcomes)
    lo, hi = cal.calibrate(0.5), cal.calibrate(0.9)
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi > lo
    assert hi > 0.7 and lo < 0.5
