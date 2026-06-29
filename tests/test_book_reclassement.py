"""Stream-1 KILL gate — the harness must RE-CLASSIFY the live crypto strategies
correctly against REAL events, not merely pass synthetic green tests.

The unit of analysis is the BOOK, not the signal. The dead momentum strategy
(S1 = ts_mom_vol_scaled) has a *positive solo Sharpe* (~+0.088) yet a NEGATIVE
leave-one-out contribution to the book (~-0.146): it is redundant bull-beta the
book is strictly better off without. The funding-squeeze sleeve (S2) is the only
positive contributor (~+0.015). If the committed ``hermes.validation.book``
machinery cannot reproduce ``S1 < 0 < S2`` on the REAL captured per-sleeve return
streams, the harness is false (mandate §1.1 KILL) and every downstream
accept/reject verdict in the program is untrustworthy.

Why a fixture, not a live reconstruction: the lake (``data/``) and the audit
scratch (``research_crypto2/``, ``research_fx/fxgate.py``) are gitignored, so a
CI test cannot regenerate the events. Instead the fixture is the EXACT input the
honest re-audit (``research_crypto2/audit_book.py``) feeds to THIS same machinery
— provenance + regeneration in ``research_streams/S1/_regenerate_fixture.py``.
The test therefore locks the verdict against captured reality
(``vert-en-test ≠ vrai-en-prod``), with no dependency on local data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hermes.validation.book import book_report, leave_one_out_delta_sharpe

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "s1" / "reclassement_sleeve_returns.json"
)


def _load() -> tuple[dict, dict[str, pd.Series]]:
    payload = json.loads(FIXTURE.read_text())
    sleeves = {
        sid: pd.Series(
            {pd.Timestamp(k): float(v) for k, v in series.items()}
        ).sort_index()
        for sid, series in payload["sleeves"].items()
    }
    return payload, sleeves


def test_reclassement_S1_negative_S2_positive_against_real_events():
    """THE KILL CRITERION: leave-one-out Δ-Sharpe ranks the dead momentum sleeve
    negative and the funding sleeve positive on the real captured book."""
    _, sleeves = _load()
    loo = leave_one_out_delta_sharpe(sleeves, weighting="inv_vol", min_periods=8)

    # Dead momentum hurts the book; the funding squeeze is the only positive leg.
    assert loo["S1_ts_mom"] < 0.0 < loo["S2_funding_squeeze"]
    # Full ordering: S1 worst, S3 in between (carry net drag), S2 best.
    assert loo["S1_ts_mom"] < loo["S3_carry_hedged"] < loo["S2_funding_squeeze"]


def test_reclassement_unit_is_loo_not_solo_sharpe():
    """The re-classement is real only because the UNIT is the book contribution,
    not the solo Sharpe: S1's solo Sharpe is POSITIVE, so any regression back to
    solo-Sharpe ranking (the error the whole stream rejects) would flip the
    verdict and break the KILL test above."""
    _, sleeves = _load()
    rep = book_report(sleeves, weighting="inv_vol", min_periods=8, n_trials=3)
    s1 = rep["sleeves"]["S1_ts_mom"]
    # Solo says "keep it" (>0); the book says "drop it" (<0). That gap is the point.
    assert s1["solo_sharpe"] > 0.0
    assert s1["loo_delta_sharpe"] < 0.0


def test_reclassement_reproduces_documented_loo_delta_sharpe():
    """Regression-lock the exact documented verdict (research_crypto2/_book.json)
    and the heterogeneous-cadence alignment (non-zero marks per sleeve)."""
    payload, sleeves = _load()
    rep = book_report(sleeves, weighting="inv_vol", min_periods=8, n_trials=3)

    documented = payload["provenance"]["documented_loo_delta_sharpe"]
    for sid, expected in documented.items():
        assert rep["sleeves"][sid]["loo_delta_sharpe"] == pytest.approx(
            expected, abs=1e-9
        )

    # The book is marked on the union grid (461 weeks); sparse sleeves are flat
    # (0.0) in their quiet weeks — the honest heterogeneous-cadence treatment.
    assert rep["book"]["n"] == 461
    assert rep["sleeves"]["S1_ts_mom"]["n_marks"] == 461
    assert rep["sleeves"]["S2_funding_squeeze"]["n_marks"] == 94
    assert rep["sleeves"]["S3_carry_hedged"]["n_marks"] == 104
