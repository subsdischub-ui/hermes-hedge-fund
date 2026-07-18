"""FORWARD — the unified collection cron (one idempotent, fail-soft job, many flux).

The runner must: write each collector's snapshot through the idempotent log; keep
going when one flux fails (fail-soft, recording the blocker); and be safe to re-run
on the same UTC day (no duplicate rows). The XS reversion shadow is gated OFF by
default (fragile in-sample — see the sanity gate); funding-squeeze always runs.
"""

from __future__ import annotations

import sys

import pandas as pd

from hermes.forward import logio, run
from hermes.research import s10_governance as gov

_DAY_MS = 86_400_000
_NOW = 100 * _DAY_MS + 3_600_000


def _ok_collector():
    def c():
        df = pd.DataFrame({"ts": [_NOW], "k": ["A"], "v": [1.0]})
        return {"logs": [("flux_ok", df, ("k",), "ts")], "blockers": {}}

    c.flux_name = "ok"
    return c


def _raising_collector():
    def c():
        raise RuntimeError("feed down")

    c.flux_name = "boom"
    return c


def test_runner_writes_logs_and_is_fail_soft(tmp_path):
    manifest = run.run_forward(
        data_dir=tmp_path, now_ms=_NOW, collectors=[_ok_collector(), _raising_collector()]
    )
    assert manifest["flux"]["ok"]["status"] == "ok"
    assert manifest["flux"]["ok"]["rows_written"] == 1
    assert manifest["flux"]["boom"]["status"] == "error"  # one failure never aborts the run
    back = logio.read_log("flux_ok", data_dir=tmp_path)
    assert list(back["v"]) == [1.0]


def test_runner_is_idempotent_same_day(tmp_path):
    run.run_forward(data_dir=tmp_path, now_ms=_NOW, collectors=[_ok_collector()])
    run.run_forward(data_dir=tmp_path, now_ms=_NOW + 60_000, collectors=[_ok_collector()])
    back = logio.read_log("flux_ok", data_dir=tmp_path)
    assert len(back) == 1  # re-running the same UTC day overwrites, never duplicates


def _stub_bybit_network(monkeypatch):
    """Neutralise la SEULE couture réseau coûteuse des collecteurs par défaut.

    Les 5 collecteurs historiques ne fetchent qu'un snapshot du jour (secondes) ;
    `bybit_funding`, lui, fait un BACKFILL COMPLET ~6 ans x 11 symboles au premier
    run sur un data_dir vierge — or ces tests CLI utilisent un tmp_path frais :
    mesuré à 140 s PAR TEST (2026-07-18) avant ce stub, en frappant l'API réelle
    depuis la suite. Le stub vide garde le collecteur EXERCÉ (enregistrement,
    fail-soft, manifest) sans réseau ; le backfill réel de prod est intact.
    """
    from hermes.forward import collect_bybit_funding as cbf

    monkeypatch.setattr(cbf, "fetch_funding_page", lambda *a, **k: [])


def test_cli_main_keeps_trial_registry_decoupled_from_forward_data_dir(tmp_path, monkeypatch):
    """Regression guard: main() used to pass --data-dir straight through to the trial
    registry, so production's BookTrialRegistry silently forked onto an ad-hoc path
    instead of its one persistent, deployment-scoped home (HERMES_GOVERNANCE_DIR /
    governance_registry_dir()). The forward-logs root and the registry's root are
    DIFFERENT concerns and must resolve independently — pin that here so a future
    edit can't quietly reintroduce the same fork."""
    forward_root = tmp_path / "forward_logs"
    registry_root = tmp_path / "registry"
    monkeypatch.setenv("HERMES_GOVERNANCE_DIR", str(registry_root))
    _stub_bybit_network(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--data-dir", str(forward_root)]
    )
    run.main()
    # The registry must land under HERMES_GOVERNANCE_DIR, NOT under --data-dir.
    assert (registry_root / "trials_by_book.json").exists()
    assert not (forward_root / "trials_by_book.json").exists()
    assert gov.BookTrialRegistry().path == registry_root / "trials_by_book.json"
    # The forward flux logs still respect --data-dir, untouched by this fix.
    assert (forward_root / "forward").exists()


def test_default_collectors_gate_xs_reversion_off_by_default():
    names_off = run.default_collector_names(wire_xs_reversion=False)
    names_on = run.default_collector_names(wire_xs_reversion=True)
    assert "funding_squeeze_shadow" in names_off  # the live edge always accrues
    assert "xs_reversion_shadow" not in names_off  # fragile → not wired by default
    assert "xs_reversion_shadow" in names_on  # but activation is a one-flag change
    for flux in ("skew", "comex", "macro", "top_lsr"):
        assert flux in names_off


def test_cli_main_manifest_includes_dead_ends_check(tmp_path, monkeypatch, capsys):
    """S10's DeadEndsLedger.check() existed but nothing in prod ever called it. main()
    must now print an informative-only verdict per pre-registration in the manifest —
    and a missing/corrupt ledger must never crash the cron (graceful fallback to the
    module's own defaults, exercised separately in test_prereg.py)."""
    import json

    monkeypatch.setenv("HERMES_GOVERNANCE_DIR", str(tmp_path / "registry"))
    _stub_bybit_network(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py", "--data-dir", str(tmp_path / "forward_logs")])
    run.main()
    printed = capsys.readouterr().out
    manifest = json.loads(printed)
    assert "dead_ends_check" in manifest
    from hermes.forward import prereg

    assert set(manifest["dead_ends_check"]) == {pr.name for pr in prereg.ALL}
    for v in manifest["dead_ends_check"].values():
        assert v["verdict"] in {
            gov.LedgerVerdict.BLOCKED.value,
            gov.LedgerVerdict.ALLOWED_NEW_AXIS.value,
            gov.LedgerVerdict.NOT_A_DEAD_END.value,
        }
