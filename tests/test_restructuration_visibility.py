"""La règle de visibilité de l'opérateur, MÉCANISÉE (2026-07-16).

Règle permanente : « tout changement shippé doit être visible sur le dashboard ET
racontable par l'agent dans le chat ». Un verdict qui ne vit que dans un doc ou un
commit n'est pas fini.

Ces tests gardent les deux surfaces pour le verdict le plus coûteux du dossier (la
piste largeur S2 : 3 essais dépensés au registre, close définitivement). Ils existent
parce que deux incidents réels l'ont exigé :
  1. le chat a rendu un 500 sur une clé de jauge manquante (tests/test_chat.py) ;
  2. `_restructuration_chat_block` compose ses lignes par walrus imbriqués + ``else ""``
     — une clé renommée ne CRASHE pas, elle fait DISPARAÎTRE la ligne en silence, ce
     qui est pire : le dashboard resterait juste pendant que l'agent, lui, se tairait.
Le test rend donc réellement les deux blocs et exige que le verdict y soit lisible.
"""

from __future__ import annotations

import pytest

from hermes.api import _restructuration_chat_block, _restructuration_status
from hermes.settings import get_settings


@pytest.fixture
def resto():
    return _restructuration_status(get_settings(), 2)


def test_dashboard_exposes_the_closed_breadth_verdict(resto):
    """Le dashboard doit porter le verdict des DEUX essais et la clôture."""
    bt = resto["breadth_test"]
    assert "CLOSE" in bt["verdict"].upper()
    # Les deux essais, chacun avec son échec chiffré.
    assert bt["essai_1_survivants"]["marginal_events_pct"] == 75.8
    assert bt["essai_2_durci"]["marginal_events_pct"] == 15.0
    assert len(bt["essai_2_durci"]["echecs"]) == 2  # marginaux ET bear-window
    # Le biais de survie MESURÉ — la raison d'être du test durci.
    assert "-12.5%" in bt["biais_de_survie_MESURÉ"]["edge_bps_par_evt"]
    # La réserve structurelle ne doit jamais être enterrée.
    assert "2022" in bt["reserve_non_resolue"]


def test_chat_actually_tells_the_breadth_verdict(resto):
    """Le bloc chat DOIT rendre la ligne largeur — pas la laisser tomber en silence.

    C'est le test qui mord : les walrus imbriqués du rendu font disparaître la ligne
    sans lever si une sous-clé est renommée.
    """
    block = _restructuration_chat_block(get_settings(), 2, 8)
    line = next((ln for ln in block.split("\n") if "LARGEUR" in ln), None)
    assert line is not None, (
        "la ligne LARGEUR a disparu du contexte chat — l'agent ne peut plus raconter "
        "un verdict qui a coûté 3 essais au registre (règle de visibilité opérateur)"
    )
    # Le chat doit dire la clôture, le chiffre qui tue, et le biais mesuré.
    assert "CLOSE" in line.upper()
    assert "+15.0%" in line
    assert "biais de survie" in line
    assert "-12.5%" in line


def test_chat_breadth_line_survives_the_full_render(resto):
    """Le bloc entier se rend sans lever (défense en profondeur, incident chat 500)."""
    block = _restructuration_chat_block(get_settings(), 2, 8)
    assert isinstance(block, str) and len(block) > 200
