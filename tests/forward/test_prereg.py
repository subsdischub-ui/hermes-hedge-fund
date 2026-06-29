"""FORWARD — the dated pre-registration records.

Pre-registration is what makes the eventual forward OOS honest: the exact cell and
the full deflation count are frozen BEFORE any forward data accrues, so the verdict
cannot be reverse-engineered. Two invariants are pinned here:

* the funding-squeeze pre-reg stays byte-identical to the LIVE ``signals.py``
  constants (it must shadow the live cell, not a drifted copy);
* the XS reversion pre-reg encodes the inverted-slow cell and the full trial count.
"""

from __future__ import annotations

from hermes.forward import prereg
from hermes.research import s10_governance as gov
from hermes.research import signals


def test_funding_squeeze_prereg_matches_live_constants():
    pr = prereg.FUNDING_SQUEEZE_CONTINUATION
    assert pr.rule["z_threshold"] == signals._FUNDC_Z
    assert pr.rule["window_bars"] == signals._FUNDC_WIN
    assert pr.rule["hold_bars"] == signals._FUNDC_HOLD_BARS
    assert pr.rule["direction"] == "LONG"
    # shadowing an already-counted live cell → NOT a new trial
    assert pr.is_new_trial is False
    assert pr.deflation_trial_count == 1869


def test_xs_reversion_prereg_is_the_inverted_slow_cell():
    pr = prereg.XS_FUNDING_REVERSION
    assert pr.sign == "INVERTED"
    assert pr.rule["cadence_days"] == 14
    assert pr.rule["funding_z_window_settlements"] == 180
    assert pr.rule["buckets"] == "terciles"
    assert pr.rule["neutrality"].startswith("dollar-neutral")
    assert pr.rule["universe_size"] == 23
    # a genuinely new pre-registered cell → +1 trial; deflate at the FULL count
    assert pr.is_new_trial is True
    assert pr.deflation_trial_count == 1892  # 1869 + 22 (S2 v-next) + 1 (this)


def test_prereg_is_dated_and_flags_contamination():
    for pr in (prereg.FUNDING_SQUEEZE_CONTINUATION, prereg.XS_FUNDING_REVERSION):
        assert pr.registered_utc == "2026-06-29"
        assert pr.book == "crypto"
        assert "contaminat" in pr.contamination.lower() or "in-sample" in pr.contamination.lower()


def test_register_trials_bumps_new_trials_only_and_is_idempotent(tmp_path):
    # the committed, tested mechanism (replacing the manual one-off bump)
    res = prereg.register_trials(data_dir=tmp_path)
    reg = gov.BookTrialRegistry(tmp_path)
    assert reg.count(gov.Book.CRYPTO) == 1870  # 1869 seed + 1 (xs_reversion only)
    assert "xs_funding_reversion" in res["registered"]
    assert "funding_squeeze_continuation" not in res["registered"]  # is_new_trial=False
    sources = {e["source"] for e in reg.provenance(gov.Book.CRYPTO)}
    assert "forward_xs_funding_reversion_prereg" in sources
    # idempotent: re-running must NOT double-count (bump() has no guard of its own)
    prereg.register_trials(data_dir=tmp_path)
    assert gov.BookTrialRegistry(tmp_path).count(gov.Book.CRYPTO) == 1870
