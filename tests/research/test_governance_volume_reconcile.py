"""Réconciliation du volume de gouvernance prod ← repo committé (2026-07-16).

Le problème mesuré en prod ce jour : le dashboard servait `crypto: 1892` quand le
repo canonique comptait 1911 — écart = exactement les 19 essais pré-enregistrés en
local et jamais miroités vers le volume (`HERMES_GOVERNANCE_DIR=/data/governance`).
Un dashboard qui affiche un compte d'essais faux viole la règle de visibilité de
l'opérateur ; et le compte d'essais est ce qui fixe la barre de déflation.

Pourquoi une UNION et pas un écrasement : `forward.prereg.register_preregs()` bumpe
un registre côté prod. Le repo se trouve être aujourd'hui un sur-ensemble strict
(vérifié : aucune source prod absente du repo), mais un écrasement perdrait
silencieusement le premier prereg enregistré en prod et jamais miroité. La fusion
prend donc, par source, le max des deux côtés — monotone, idempotente, sans perte.
"""

from __future__ import annotations

import json

from hermes.research import s10_governance as gov


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _stale_volume(tmp_path):
    """Un volume périmé, façonné comme la vraie prod : un sous-ensemble du repo."""
    vol = tmp_path / "volume"
    _write(
        vol / "trials_by_book.json",
        {
            "counts": {"crypto": 1892, "fx_metals": 53, "cross": 5},
            "provenance": {
                "crypto": [
                    {"source": "funding_discovery_walkforward_sweep", "n": 648},
                    {"source": "options_surface_grid_VRP_TERM_VOV_RSKEW", "n": 1024},
                    {"source": "dvol_reversion_config_grid", "n": 96},
                    {"source": "stablecoin_growth_config_grid", "n": 96},
                    {"source": "positioning_flow_hunts_H1_H5", "n": 5},
                    {"source": "S2_xsec_neutral_sleeves_4prereg_plus_18exploratory", "n": 22},
                    {"source": "forward_xs_funding_reversion_prereg", "n": 1},
                ],
                "fx_metals": [{"source": "seed", "n": 53}],
                "cross": [{"source": "seed", "n": 5}],
            },
            "seeded": True,
        },
    )
    return vol


def test_reconcile_pulls_the_repo_trials_into_a_stale_volume(tmp_path):
    """Le cas réel : le volume périmé doit remonter au compte canonique du repo."""
    vol = _stale_volume(tmp_path)
    summary = gov.reconcile_governance_volume(vol)

    reg = gov.BookTrialRegistry(vol)
    repo_count = gov.BookTrialRegistry(gov.repo_governance_dir()).count(gov.Book.CRYPTO)
    assert reg.count(gov.Book.CRYPTO) == repo_count
    assert summary["trials_changed"] is True
    assert summary["counts_after"]["crypto"] == repo_count


def test_reconcile_never_loses_a_volume_only_source(tmp_path):
    """Le prereg enregistré EN PROD et jamais miroité ne doit pas disparaître.

    C'est la raison d'être de l'union : `register_preregs()` bumpe côté prod.
    """
    vol = _stale_volume(tmp_path)
    gov.BookTrialRegistry(vol).bump(gov.Book.CRYPTO, 3, source="prereg_ne_vivant_quen_prod")

    gov.reconcile_governance_volume(vol)

    prov = {e["source"]: e["n"] for e in gov.BookTrialRegistry(vol).provenance(gov.Book.CRYPTO)}
    assert prov["prereg_ne_vivant_quen_prod"] == 3, "un essai compté en prod a été effacé"
    repo_count = gov.BookTrialRegistry(gov.repo_governance_dir()).count(gov.Book.CRYPTO)
    assert gov.BookTrialRegistry(vol).count(gov.Book.CRYPTO) == repo_count + 3


def test_reconcile_is_idempotent(tmp_path):
    """Rejouer la réconciliation (chaque redéploiement !) ne doit RIEN double-compter."""
    vol = _stale_volume(tmp_path)
    gov.reconcile_governance_volume(vol)
    first = json.loads((vol / "trials_by_book.json").read_text())

    second_summary = gov.reconcile_governance_volume(vol)
    second = json.loads((vol / "trials_by_book.json").read_text())

    assert first == second
    assert second_summary["trials_changed"] is False  # rien à faire au 2e passage


def test_reconciled_counts_always_reconcile_with_provenance(tmp_path):
    """L'invariant du registre (cf. test_committed_registry_provenance_reconciles_with_counts)
    doit tenir APRÈS fusion — c'est le défaut exact commis à la main le 2026-07-16."""
    vol = _stale_volume(tmp_path)
    gov.BookTrialRegistry(vol).bump(gov.Book.CRYPTO, 2, source="volume_only")
    gov.reconcile_governance_volume(vol)

    payload = json.loads((vol / "trials_by_book.json").read_text())
    for book, count in payload["counts"].items():
        traced = sum(e["n"] for e in payload["provenance"].get(book, []))
        assert traced == count, f"{book}: counts={count} vs provenance={traced}"


def test_reconcile_copies_the_committed_dead_ends_ledger(tmp_path):
    """Le ledger est code-as-truth : le volume doit recevoir les 41 entrées committées."""
    vol = _stale_volume(tmp_path)
    _write(vol / "dead_ends_ledger.json", [{"name": "une entrée périmée", "tags": []}])

    gov.reconcile_governance_volume(vol)

    names = {e["name"] for e in json.loads((vol / "dead_ends_ledger.json").read_text())}
    assert names == {e.name for e in gov.DEFAULT_DEAD_ENDS}
    assert any("BREADTH" in n for n in names)


def test_reconcile_is_a_noop_when_volume_is_the_repo_itself(tmp_path, monkeypatch):
    """Sans HERMES_GOVERNANCE_DIR, le dir canonique EST le repo : ne jamais se réécrire."""
    monkeypatch.delenv("HERMES_GOVERNANCE_DIR", raising=False)
    summary = gov.reconcile_governance_volume(gov.repo_governance_dir())
    assert summary["skipped"] == "volume is the repo itself"


def test_repo_governance_dir_ignores_the_env_override(tmp_path, monkeypatch):
    """`repo_governance_dir()` doit TOUJOURS désigner le committé — c'est la source de
    vérité de la fusion, sinon on réconcilierait le volume avec lui-même."""
    monkeypatch.setenv("HERMES_GOVERNANCE_DIR", str(tmp_path / "ailleurs"))
    assert gov.repo_governance_dir() != gov.governance_registry_dir()
    assert (gov.repo_governance_dir() / "trials_by_book.json").exists()
