"""Stream-1 §1.3 KILL on REAL S2 data — the meta-labeled fractional-Kelly sizing
layer, leave-one-out across BTC/ETH/SOL, judged on DEFLATED Sharpe vs the un-sized
baseline.

The verdict is an HONEST NEGATIVE for *adoption*: the sizing improves the deflated
Sharpe on the two deep majors (BTC, ETH) but DEGRADES the thin instrument (SOL) — the
"helps the easy ones, tanks the hard one" signature of size-overfit at this data
scale. So the gate does not pass, and S2 stays un-sized. The machinery is the
deliverable; the gate correctly refuses to bless it.

The fixture is the real S2 funding-squeeze event set (funding_z>=2, 24x4h hold,
15bps RT); provenance + regeneration in
``research_streams/S1/_regenerate_s2_sizing_fixture.py``. CI never touches the
gitignored lake/scratch — it locks the honest verdict against captured reality.
"""

from __future__ import annotations

import json
from pathlib import Path

from hermes.validation.meta_sizing import loo_sizing_gate

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "s1" / "s2_sizing_events.json"
)


def _instruments() -> dict:
    return json.loads(FIXTURE.read_text())["instruments"]


def test_s2_sizing_fixture_has_the_three_deep_instruments():
    inst = _instruments()
    assert set(inst) == {"BTC", "ETH", "SOL"}
    # Real S2 event counts — the thin instrument (SOL) is where overfit bites.
    assert len(inst["BTC"]["gross"]) == 73
    assert len(inst["ETH"]["gross"]) == 77
    assert len(inst["SOL"]["gross"]) == 47


def test_s2_sizing_gate_is_an_honest_negative_overfit_on_the_thin_instrument():
    gate = loo_sizing_gate(_instruments(), fraction=0.20, min_samples=50)
    # All folds fit (training on two instruments clears min_samples)...
    assert gate["all_fitted"] is True
    # ...sizing helps the two deep majors but DEGRADES the thin one -> NOT adopted.
    assert gate["improved_folds"] >= 2
    assert gate["per_fold"]["SOL"]["degraded"] is True
    assert gate["passed"] is False  # §1.3 KILL: size-overfit rejected, S2 stays un-sized
