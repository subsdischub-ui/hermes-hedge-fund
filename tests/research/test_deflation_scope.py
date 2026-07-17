"""La déflation doit être SYMÉTRIQUE et INFORMATIVE (2026-07-16).

Deux défauts mesurés le même jour, sur les fichiers réels du repo :

1. **VERROU DE STATU QUO.** `research_crypto2/_book.json` déflate le livre live (le
   titulaire) avec `n_trials: 3` — les 3 sleeves. Les challengers, eux, sont déflatés au
   compte du livre : #41 à N=1911, #37 à 1909, #38 à 1910. Mesuré avec leurs sr_variance
   RÉELLES : le livre passe à N=3 (E[max]=0.040 < Sharpe=0.0739) et **échoue dès N=10**
   (E[max]=0.0742). Une barre que le titulaire ne franchit qu'en changeant de règle n'est
   pas un filtre de qualité.

2. **SATURATION = ZÉRO INFORMATION.** Au compte du livre, E[max] dépasse le Sharpe de tous
   les candidats — le DSR tombe alors sous 0.5 *par construction*, sans hypothèse sur les
   moments. Mesuré : #37 → 0.0000253, #38 → 0.000215, #41 → 0.000272. Trois nombres
   quasi identiques et quasi nuls : ils ne discriminent RIEN. Le DSR a pourtant été
   rapporté comme une mesure (le verdict du test durci l'affichait). Il discriminait quand
   N était spécifique à la famille : #14 → 0.287, #31 → 0.483, #23 → grid-max 0.1934 vs
   E[max|1024]=0.1950.

**La cause est une erreur de spécification, pas une prudence.** Le DSR (Bailey & López de
Prado) déflate par E[max] sur les essais **desquels le candidat a été SÉLECTIONNÉ**. Or
53.6 % du compte crypto (1024/1911) est la grille options VRP/TERM/VOV/RSKEW — une famille,
des données et un livre DISJOINTS de S2. Déflater S2 par une grille qui n'aurait jamais pu
le produire n'ajoute pas de rigueur : ça détruit l'information.

**Le correctif ne choisit pas une convention — il rend le choix EXPLICITE et symétrique**,
et refuse une comparaison entre deux conventions différentes. C'est le mécanisme (R3) :
avant, la règle était un paramètre libre que chaque script fixait dans son coin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hermes.research import s10_governance as gov

# Le livre live, verbatim de research_crypto2/_book.json (2026-07-16). Ces constantes
# sont figées ICI parce que research_*/ est gitignoré : le CI n'a PAS les artefacts de
# recherche. Le gros des tests ci-dessous porte donc sur la MÉCANIQUE (qui, elle, est
# versionnée) ; seul le test d'ancrage lit le fichier réel, et il skippe proprement là
# où il n'existe pas — même motif que `lake` dans test_carry_harvest.py.
BOOK_SHARPE = 0.07388835755253846
BOOK_SR_VAR = 0.00222

_BOOK_JSON = Path(__file__).resolve().parents[2] / "research_crypto2" / "_book.json"
needs_book = pytest.mark.skipif(
    not _BOOK_JSON.exists(),
    reason="research_crypto2/_book.json est gitignoré (artefact de recherche local)",
)


def _returns_with_sharpe(sharpe: float, n: int = 461) -> list[float]:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    x = (x - x.mean()) / x.std(ddof=1)  # Sharpe exactement `sharpe` par observation
    return (sharpe + x).tolist()


@needs_book
def test_the_committed_book_really_is_deflated_at_three_trials():
    """Le fait de départ, lu sur le fichier réel — pas une allégation.

    Skippe en CI (research_*/ gitignoré) : c'est le seul test du fichier qui dépende
    d'un artefact non versionné, et il vérifie que les constantes figées ci-dessus
    n'ont pas dérivé du réel. Chemin ABSOLU — un chemin relatif dépendait du cwd de
    pytest, ce qui est précisément ce qui a fait rougir le CI (92fae3b).
    """
    book = json.loads(_BOOK_JSON.read_text())
    assert book["n_trials"] == 3
    assert book["book"]["sharpe"] == pytest.approx(BOOK_SHARPE)


def test_incumbent_passes_at_three_and_fails_at_the_challenger_bar():
    """Le verrou de statu quo, épinglé : la même performance, deux verdicts opposés,
    selon la seule convention de N. C'est le défaut, pas une curiosité."""
    at_3 = gov.deflate(_returns_with_sharpe(BOOK_SHARPE), BOOK_SR_VAR,
                       n_trials=3, scope=gov.TrialScope.SELECTION_SET)
    at_book = gov.deflate(_returns_with_sharpe(BOOK_SHARPE), BOOK_SR_VAR,
                          n_trials=1969, scope=gov.TrialScope.BOOK_WIDE)
    assert not at_3.saturated, "à N=3 le titulaire domine son E[max] — il passe"
    assert at_book.saturated, "au compte du livre le titulaire est sous son E[max] — il échoue"
    assert at_book.dsr < at_3.dsr


def test_comparing_two_scopes_is_refused():
    """Le mécanisme central : on ne peut PLUS juger un challenger à 1911 en gardant le
    titulaire à 3. Avant, rien dans le code ne l'interdisait — c'est ainsi que
    l'asymétrie a vécu."""
    incumbent = gov.deflate(_returns_with_sharpe(BOOK_SHARPE), BOOK_SR_VAR,
                            n_trials=3, scope=gov.TrialScope.SELECTION_SET)
    challenger = gov.deflate(_returns_with_sharpe(0.2727), 0.02323,
                             n_trials=1911, scope=gov.TrialScope.BOOK_WIDE)
    with pytest.raises(ValueError, match="non comparable"):
        gov.assert_comparable(incumbent, challenger)


def test_same_scope_compares_fine():
    a = gov.deflate(_returns_with_sharpe(0.30), 0.02, n_trials=27,
                    scope=gov.TrialScope.SELECTION_SET)
    b = gov.deflate(_returns_with_sharpe(0.10), 0.02, n_trials=27,
                    scope=gov.TrialScope.SELECTION_SET)
    gov.assert_comparable(a, b)  # ne lève pas


def test_saturation_is_flagged_and_explained():
    """Un DSR saturé doit se DÉCLARER sans information, pas se faire passer pour une mesure.

    Les 3 challengers du livre (#37/#38/#41) rendaient 0.0000253 / 0.000215 / 0.000272 —
    présentés comme des mesures alors qu'ils étaient une constante.
    """
    r = gov.deflate(_returns_with_sharpe(0.2727), 0.02323, n_trials=1911,
                    scope=gov.TrialScope.BOOK_WIDE)
    assert r.saturated
    assert r.dsr < 0.5
    assert "saturé" in r.reason.lower() or "satur" in r.reason.lower()
    assert "E[max]" in r.reason


def test_selection_set_scope_still_discriminates():
    """L'instrument doit rendre des verdicts DIFFÉRENTS quand N est spécifique à la famille —
    c'est ce qu'il faisait (#14 DSR 0.287, #31 0.483) avant le forfait."""
    fort = gov.deflate(_returns_with_sharpe(0.45), 0.02, n_trials=27,
                       scope=gov.TrialScope.SELECTION_SET)
    faible = gov.deflate(_returns_with_sharpe(0.05), 0.02, n_trials=27,
                         scope=gov.TrialScope.SELECTION_SET)
    assert fort.dsr > 0.9 > faible.dsr, "à N de famille, le DSR sépare le fort du faible"
    assert not fort.saturated and faible.saturated
