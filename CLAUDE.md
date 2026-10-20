# CLAUDE.md — Hermès

## CI locale d'abord (minutes GitHub payées par le fondateur)

Directive du fondateur (2026-10-19/20, `~/.claude/CLAUDE.md` R8 point 4). Elle vaut pour **toute** séance, y compris les séances cloud (claude.ai/code), qui ne lisent pas le `CLAUDE.md` du Mac : ce paragraphe et `.claude/settings.json` sont leur seule source. Les minutes d'Actions sont payées par le fondateur. **GitHub ne fait tourner que ce qui serait faussé en local.**

- **Le dépôt n'a qu'un workflow, `.github/workflows/deploy.yml`** (déclenché par une poussée sur `main`, ou à la main). Son job `test` rejoue `uv sync --extra dev`, `uv run ruff check src tests` et `uv run pytest -q` ; son job `build-deploy` construit l'image, la pousse au registre GitHub (secrets du dépôt) et déclenche Dokploy. Le déploiement DÉPEND de GitHub (secrets, registre) : il reste là-bas. Le dépôt n'a pas de miroir Docker (`scripts/ci-local.sh`) : les tests se rejouent directement.
- **AVANT toute poussée sur `main`, lancer en local, à la racine du dépôt** (mesuré le 2026-10-20 : 1082 réussis, 47 ignorés, ≈ 2 min 30 après installation) :
  ```bash
  uv sync --extra dev && uv run ruff check src tests && uv run pytest -q
  ```
  Ne pousser sur `main` que si les trois étapes sortent 0. Une poussée sur `main` déclenche aussi le déploiement : jamais « pousser pour voir ».
- **Ordre** : commiter → la commande ci-dessus verte → pousser UNE fois (branche) → ouvrir la PR (brouillon permis ; `deploy.yml` ne se déclenche pas sur les PR) → fusionner : seule la fusion dans `main` coûte une passe sur GitHub, voulue car elle déploie.
- **Séance cloud** : si `uv` est installé dans la séance, lancer la commande ci-dessus avant de pousser. Sinon (ou si elle échoue pour une raison d'environnement), commiter, pousser UNE fois, ouvrir la PR en brouillon, puis écrire dans la réponse : « vérification à lancer sur le Mac : `uv sync --extra dev && uv run ruff check src tests && uv run pytest -q` sur la branche » suivie du nom réel de la branche (celui que donne `git branch --show-current`). Ne jamais relancer Actions, ne jamais repasser un brouillon en « prêt » pour déclencher la CI.
- **Docker** : la commande ci-dessus n'en a pas besoin. Si un miroir Docker est construit un jour, Docker Desktop devra disposer d'au moins 16 Go de mémoire.
- **Interdit sans demande explicite du fondateur dans la séance** : `gh run rerun`, `gh workflow run`, `gh workflow enable` (un `gh workflow run deploy.yml` redéploie en production). Le garde `.claude/hooks/gh-minutes-guard.py` (déclaré dans `.claude/settings.json`, PreToolUse sur Bash) les bloque. Faute de `scripts/ci-local.sh`, il laisse passer `gh pr merge` en rappelant la règle sur stderr : la fusion n'est donc permise qu'après la commande locale verte. Dérogation, seulement sur instruction explicite du fondateur : préfixer la commande par `GH_ACTIONS_OK=1`. Test du garde : `tests/test_gh_minutes_guard.py`.
