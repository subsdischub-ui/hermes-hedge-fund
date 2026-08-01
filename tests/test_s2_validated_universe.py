"""S2 ne doit trader QUE l'univers sur lequel il a été validé (2026-08-01).

Défaut de fidélité mesuré ce jour, même classe que les trois du 2026-07-10 (frais maker,
no-pyramid par symbole, plancher de confiance) : le live diverge de ce qui a été validé.

LE FAIT, vérifié à la source :
  research_crypto2/audit_s2_funding_squeeze.py:45  ->  ASSETS = ["BTC", "ETH", "SOL"]
  research_crypto2/_s2_funding_squeeze.json        ->  per_asset = {BTC, ETH, SOL} seulement
Le joint-null panier p=0.0005 — l'unique preuve statistique du livre — porte sur TROIS actifs.

Or `funding_squeeze_continuation` était enregistré SANS `symbols=`, donc `symbols=None`, donc
il tournait sur les 7 perps de l'univers live. Le mécanisme de restriction existe pourtant et
est utilisé pour trois autres skills (intraday_tod_btc -> BTC ; options_iv_dvol_reversion et
onchain_stablecoin_growth -> BTC/ETH) : il a simplement été oublié sur le seul edge validé.

MESURE (règle gelée, horizon 24 barres, coût 15 bps RT, données Binance réelles, fenêtre
2026-02 -> 2026-08) :
  - univers VALIDÉ (BTC/ETH/SOL)      : n=13, moyenne  -82.1 bps/événement
  - univers HORS validation (4 alts)  : n= 9, moyenne -739.6 bps/événement
Les alts non validés sont ~9x pires. Ils n'auraient jamais dû être tradés par ce signal.

Note de gouvernance : l'entrée #41 du ledger (« S2 funding-squeeze BREADTH ») a CLOS la piste
d'élargissement d'univers après deux échecs pré-enregistrés. Le live élargissait déjà, sans
que personne ne le voie.
"""

from __future__ import annotations

from hermes.skills import get_skill_library
from hermes.skills.library import _applies_to

# Verbatim de research_crypto2/audit_s2_funding_squeeze.py:45
VALIDATED_ASSETS = ("BTC", "ETH", "SOL")


def _s2():
    skill = get_skill_library().get("funding_squeeze_continuation")
    assert skill is not None, "funding_squeeze_continuation absent du registre"
    return skill


def test_s2_declares_its_validated_universe():
    """`symbols=None` = 'tourne partout' — inacceptable pour le seul edge validé du livre."""
    skill = _s2()
    assert skill.symbols is not None, (
        "funding_squeeze_continuation tourne sur TOUT l'univers alors qu'il n'est validé "
        "que sur BTC/ETH/SOL (audit_s2_funding_squeeze.py:45)"
    )
    assert set(skill.symbols) == set(VALIDATED_ASSETS)


def test_s2_runs_on_each_validated_symbol():
    skill = _s2()
    for base in VALIDATED_ASSETS:
        assert _applies_to(skill, f"{base}/USDT:USDT"), f"S2 doit tourner sur {base}"


def test_s2_never_runs_outside_its_validation():
    """Les 4 alts mesurés à -739.6 bps/événement ne doivent plus jamais recevoir ce signal."""
    skill = _s2()
    for base in ("AVAX", "SUI", "ARB", "WIF"):
        assert not _applies_to(skill, f"{base}/USDT:USDT"), (
            f"S2 tourne sur {base}, hors de son univers de validation"
        )


def test_the_validated_universe_matches_the_audit_file():
    """Garde-fou anti-dérive : si un jour le fichier de validation couvre d'autres actifs,
    ce test échoue et force la mise à jour CONSCIENTE de la restriction (jamais l'inverse)."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "research_crypto2" / "_s2_funding_squeeze.json"
    if not p.exists():  # artefact de recherche gitignoré — absent en CI
        return
    per_asset = json.loads(p.read_text())["per_asset"]
    assert set(per_asset) == set(VALIDATED_ASSETS), (
        f"le fichier de validation couvre {sorted(per_asset)} — la restriction du skill "
        f"({sorted(VALIDATED_ASSETS)}) doit être mise à jour EN CONSCIENCE"
    )
