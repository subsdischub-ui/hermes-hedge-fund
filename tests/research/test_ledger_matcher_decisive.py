"""Le matcher du ledger doit BLOQUER les morts — mesuré à 73 % de faux PASS (2026-07-16).

Histoire des deux incidents, qui gouverne toute la conception :

1. **2026-07-03 — faux BLOCAGE.** La règle originale (une sous-chaîne suffit) a bloqué une
   proposition d'axe DISJOINT qui contenait juste la phrase générique
   « short-horizon continuation » (un tag de #15). Correctif appliqué alors : frontières de
   mots + exiger « nom verbatim OU >=2 tags OU un tag >=4 mots ».
2. **2026-07-16 — 8 faux PASS sur 11.** Ce durcissement a sur-corrigé. Mesuré :
   `donchian breakout on 4h klines`, `momentum`, `trend following`, `CVD order flow imbalance`,
   `funding term structure`, `maker execution free lunch`, `channel breakout`,
   `intraday time of day seasonality` — TOUS notoirement morts — passaient le gate.

**Le défaut de conception** : la règle utilise la LONGUEUR du tag comme proxy de sa
SPÉCIFICITÉ. C'est faux, et #15 en porte la preuve dans ses propres tags : `donchian`
(un mot, ultra-spécifique) et `short-horizon continuation` (deux mots, générique — le
coupable de 2026-07-03) y sont traités à l'identique : aucun ne peut mordre seul.

**Le correctif** : la décisivité devient une propriété DÉCLARÉE du tag (`decisive_tags`),
pas une fonction de son nombre de mots. Un tag décisif bloque seul ; un tag corroborant
garde la règle des >=2.

**Asymétrie des coûts, qui tranche les cas douteux** : un faux BLOCAGE est RÉCUPÉRABLE
(nommer un `new_data_axis` rend ALLOWED-NEW-AXIS, ou on édite l'entrée) ; un faux PASS
coûte un essai dépensé et, au pire, une stratégie morte déployée. À doute égal, on bloque.
"""

from __future__ import annotations

import pytest

from hermes.research import s10_governance as gov

# Les 11 sondes exactes du scan du 2026-07-16. 8 passaient à tort.
MORTS_NOTOIRES = [
    ("donchian breakout on 4h klines", "Fast/any-cadence"),
    ("channel breakout", None),
    ("momentum", None),
    ("trend following", None),
    ("CVD order flow imbalance", None),
    ("funding term structure", None),
    ("maker execution free lunch", "Maker"),
    ("intraday time of day seasonality", None),
    ("liquidation cascade reversion", None),
    ("cross-venue funding dispersion", None),
    ("ml feature zoo", None),
]


@pytest.mark.parametrize(("proposal", "expect_fragment"), MORTS_NOTOIRES)
def test_ledger_blocks_every_notoriously_dead_family(proposal, expect_fragment):
    """Une famille morte doit être bloquée, même énoncée en deux mots.

    C'est le mécanisme (R3) : sans lui, la gouvernance S10 est un détecteur sans
    conséquence — le prochain agent re-dépense un essai sur du Donchian EN CROYANT
    L'OUTIL.
    """
    verdict = gov.DeadEndsLedger().check(proposal)
    assert verdict["verdict"] is gov.LedgerVerdict.BLOCKED, (
        f"FAUX PASS sur '{proposal}' — famille notoirement morte non bloquée"
    )
    if expect_fragment:
        assert expect_fragment in verdict["matched"]


# --------------------------------------------------------------------------- #
# Le pendant : ne PAS re-créer l'incident de 2026-07-03
# --------------------------------------------------------------------------- #
TRAVAIL_LEGITIME = [
    # L'incident historique EXACT : phrase générique, axe disjoint. Doit passer.
    "short-horizon continuation of a disjoint positioning axis never tested here",
    # Le slate vivant ne doit JAMAIS être bloqué par sa propre gouvernance.
    "funding squeeze continuation on the validated basket",
    "mean reversion regime RSI2 on SOL",
    # Les revive_conditions explicitement nommées par le ledger lui-même.
    "cross-venue bybit funding as a new data axis",
    "hyperliquid hourly funding requalified from scratch",
    # Infra / mesure — ni un pari ni une famille.
    "extend the lake to delisted perps for point-in-time universes",
    "build a crowding monitor from funding and open interest as a dashboard",
]


@pytest.mark.parametrize("proposal", TRAVAIL_LEGITIME)
def test_ledger_does_not_false_block_legitimate_work(proposal):
    """Le durcissement ne doit pas ressusciter le faux blocage de 2026-07-03."""
    verdict = gov.DeadEndsLedger().check(proposal)
    assert verdict["verdict"] is gov.LedgerVerdict.NOT_A_DEAD_END, (
        f"FAUX BLOCAGE sur '{proposal}' -> {verdict['matched']}"
    )


def test_generic_tag_still_cannot_fire_alone():
    """La règle des >=2 tags reste en vigueur pour les tags CORROBORANTS.

    `short-horizon continuation` est un tag de #15 et le coupable de 2026-07-03 : il ne
    doit jamais être promu décisif. C'est ce test qui empêche de « réparer » les faux
    PASS en rendant tous les tags décisifs.
    """
    entry = next(e for e in gov.DEFAULT_DEAD_ENDS if "Fast/any-cadence" in e.name)
    assert "short-horizon continuation" in entry.tags
    assert "short-horizon continuation" not in getattr(entry, "decisive_tags", ())


def test_decisive_tags_survive_the_json_round_trip(tmp_path):
    """Le ledger est code-as-truth : les tags décisifs doivent survivre au JSON committé,
    sinon la prod (qui lit le JSON) aurait un matcher plus permissif que les tests."""
    import json

    path = tmp_path / "led.json"
    path.write_text(json.dumps([e.as_dict() for e in gov.DEFAULT_DEAD_ENDS]))
    reloaded = gov.DeadEndsLedger.load(path)
    assert reloaded.check("donchian breakout on 4h klines")["verdict"] is gov.LedgerVerdict.BLOCKED


# --------------------------------------------------------------------------- #
# Entrée #42 — Boucle β / vendeur de bilan (2026-07-18) : le 4e verdict orphelin
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "proposal",
    [
        # Les 5 formulations mesurées passantes AVANT l'entrée (trou vérifié 2x par
        # exécution : par le panel adversarial du workflow, puis par l'orchestrateur).
        "harvest delta-neutral funding carry on illiquid tail perps",
        "long the illiquid niche basket to collect funding where funds cannot go",
        "delta-neutral cash-and-carry basis on small-cap perps",
        "deposit into HLP vault as counterparty of the traders",
        "sell balance-sheet insurance via medium carry on alts",
        # Reformulations naturelles supplémentaires.
        "reverse carry: long perp short spot on negative-funding tail coins",
        "the small player collects funding in the illiquid tail where funds cannot go",
    ],
)
def test_ledger_blocks_illiquid_balance_sheet_carry(proposal):
    """Le verdict Boucle β ne vivait qu'en prose (FINDINGS_BETA.md) — cas R3.

    Chiffres du kill (vérifiés à la source avant d'écrire l'entrée) : pente forward
    2025+ SIGNÉE +0.047 (t=+2.72) — la récolte monte AVEC la liquidité, l'inverse de
    la thèse « le petit va où les fonds ne peuvent pas » ; queue Binance médiane
    -2.3 %, moyenne -10.8 % ; reverse-carry +15 % brut = prime de borrow, -1.9 % au
    borrow no-arb λ=1.0 (0 % des marchés positifs) ; 3 red-teams unanimes.
    """
    verdict = gov.DeadEndsLedger().check(proposal)
    assert verdict["verdict"] is gov.LedgerVerdict.BLOCKED, (
        f"FAUX PASS sur '{proposal}' — le verdict Boucle β doit bloquer"
    )
    assert "Illiquid-tail" in verdict["matched"]


@pytest.mark.parametrize(
    "proposal",
    [
        # S3 est SUSPENDU (pas mort) : son réveil nommé — fill-rate maker mesuré au
        # stage_1 — ne doit JAMAIS être bloqué par l'entrée Boucle β. C'est le piège
        # exact de cette entrée : S3 EST un cash-and-carry, mais sur le panier
        # LIQUIDE validé, gouverné par sa suspension, pas par le ledger.
        "wake funding_carry_harvest once the real maker fill-rate is measured at stage_1",
        "réveil de funding_carry_harvest au fill-rate maker réel mesuré",
        # P1/P2 (positions survivantes) et l'ancre restent libres.
        "shadow-measure maker fill-rate on our own S2 and MR entries",
        "funding squeeze continuation on the validated basket",
        "cross-venue bybit funding as a new data axis",
        # Le moniteur dormant de P4 lui-même est de l'infra, pas un pari.
        "dormant monitor of annualized basis with a pre-written contango threshold",
    ],
)
def test_beta_entry_does_not_block_s3_revival_nor_survivor_positions(proposal):
    verdict = gov.DeadEndsLedger().check(proposal)
    assert verdict["verdict"] is gov.LedgerVerdict.NOT_A_DEAD_END, (
        f"FAUX BLOCAGE sur '{proposal}' -> {verdict['matched']}"
    )
