"""L'image doit embarquer TOUT ce que le runtime lit (2026-07-16).

Cause racine trouvée ce jour : le Dockerfile copiait `src` et `config` mais **jamais
`governance/`**. Le conteneur n'a donc JAMAIS vu l'état canonique committé —
`governance_registry_dir()` retombait sur un `/app/governance` inexistant, que
`seed_defaults()` recréait à 1869+préregs = 1892 à chaque démarrage. D'où un
dashboard bloqué sur `crypto: 1892` (repo : 1911) pendant des semaines, et une
`reconcile_governance_volume()` réduite à un no-op (elle fusionnait depuis un
dossier absent).

Le registre d'essais fixe la barre de déflation de tout le livre et le ledger des
morts empêche de re-dépenser des essais déjà payés : les livrer dans l'image n'est
pas cosmétique. Ce test est le mécanisme (R3) qui empêche la régression — un
`COPY` oublié est invisible en local, où le chemin repo existe toujours.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"

# Répertoires que le code lit au RUNTIME (pas au build) et qui doivent donc exister
# dans l'image finale. `data/` est délibérément absent : c'est le volume persistant.
_RUNTIME_DIRS = ["src", "config", "governance"]


def _runtime_stage() -> str:
    """Le texte de l'étage `runtime` du Dockerfile multi-étages."""
    text = _DOCKERFILE.read_text()
    marker = "AS runtime"
    idx = text.find(marker)
    assert idx != -1, "l'étage runtime a disparu du Dockerfile"
    return text[idx:]


@pytest.mark.parametrize("directory", _RUNTIME_DIRS)
def test_runtime_image_ships_directory(directory):
    """Chaque dossier lu au runtime doit être COPY'é dans l'étage final."""
    stage = _runtime_stage()
    pattern = rf"^COPY\b[^\n]*\s{re.escape(directory)}\s+\./{re.escape(directory)}\b"
    assert re.search(pattern, stage, re.MULTILINE), (
        f"'{directory}/' n'est pas copié dans l'image runtime — le conteneur lira un "
        f"chemin inexistant et se rabattra silencieusement sur un défaut (le bug exact "
        f"du 2026-07-16 : governance/ absent -> registre re-semé à 1892 au lieu de 1911)"
    )


def test_governance_dir_actually_holds_the_canonical_state():
    """Garde-fou du garde-fou : copier un dossier vide ne servirait à rien."""
    gov_dir = _DOCKERFILE.parent / "governance"
    assert (gov_dir / "trials_by_book.json").exists()
    assert (gov_dir / "dead_ends_ledger.json").exists()


def test_dockerignore_does_not_exclude_governance():
    """Un `.dockerignore` trop large annulerait le COPY sans bruit."""
    ignore = _DOCKERFILE.parent / ".dockerignore"
    if not ignore.exists():
        return
    entries = {
        line.strip().strip("/")
        for line in ignore.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "governance" not in entries
