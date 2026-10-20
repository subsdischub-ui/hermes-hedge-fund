#!/usr/bin/env python3
"""
gh-minutes-guard — CI locale d'abord (directive du fondateur, 2026-10-19/20, ~/.claude/CLAUDE.md R8 point 4).

Les minutes GitHub Actions sont payées par le fondateur. Ce garde (PreToolUse, Bash) impose deux choses à TOUTES les séances :

1. Aucune minute dépensée « pour voir » : `gh run rerun`, `gh workflow run`, `gh workflow enable` sont bloqués.
   Exception : le fondateur l'a demandé explicitement dans la séance → préfixer la commande par `GH_ACTIONS_OK=1`.

2. Aucune fusion sans le miroir local : `gh pr merge` est bloqué si le dépôt de la PR possède `scripts/ci-local.sh`
   (sur sa branche par défaut) et que le SHA de tête de la PR n'a pas le statut de commit `ci-local` = success.
   Le statut se pose avec `bash scripts/ci-local.sh --post` (miroir Linux, gratuit). Échec de vérification (réseau, gh) →
   blocage (fail-closed). Exception : instruction explicite du fondateur → `CI_LOCAL_SKIP_OK=1`.

Dépôt sans miroir : la fusion passe, avec un rappel sur stderr (le miroir est à construire, voir R8.4).
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys

SPLIT = re.compile(r"&&|\|\||;|\||\n")
# La commande doit COMMENCER le segment (après d'éventuelles affectations VAR=val ; `cd …` est un segment à part) :
# une commande gh citée dans une chaîne (printf, echo, heredoc, test du garde) n'est pas une commande lancée.
LEAD = r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:command\s+|env\s+)?"
MERGE_FLAGS_WITH_VALUE = {"-R", "--repo", "-t", "--subject", "-b", "--body", "-F", "--body-file",
                          "--match-head-commit", "-A", "--author-email"}


def block(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


def gh(args: list[str], cwd: str | None) -> tuple[int, str]:
    try:
        p = subprocess.run(["gh", *args], cwd=cwd or None, capture_output=True, text=True, timeout=20)
        return p.returncode, p.stdout.strip()
    except Exception as e:  # noqa: BLE001 — tout échec = non vérifiable
        return 1, str(e)


def parse_merge(segment: str) -> tuple[str | None, str | None]:
    """Renvoie (sélecteur de PR, dépôt -R) d'un segment `gh pr merge …`."""
    try:
        toks = shlex.split(segment)
    except ValueError:
        toks = segment.split()
    try:
        i = toks.index("merge")
    except ValueError:
        return None, None
    rest = toks[i + 1:]
    selector, repo, skip = None, None, False
    for j, t in enumerate(rest):
        if skip:
            skip = False
            continue
        if t in ("-R", "--repo"):
            repo = rest[j + 1] if j + 1 < len(rest) else None
            skip = True
            continue
        if t.startswith("--repo="):
            repo = t.split("=", 1)[1]
            continue
        if t in MERGE_FLAGS_WITH_VALUE:
            skip = True
            continue
        if t.startswith("-"):
            continue
        if selector is None:
            selector = t
    return selector, repo


def check_merge(segment: str, cwd: str | None) -> None:
    selector, repo = parse_merge(segment)
    view = ["pr", "view"] + ([selector] if selector else []) + (["-R", repo] if repo else []) + ["--json", "headRefOid,url"]
    rc, out = gh(view, cwd)
    if rc != 0:
        block("gh-minutes-guard : impossible de lire la PR à fusionner (gh pr view a échoué) — fusion bloquée (fail-closed). "
              "Vérifiez la PR, ou instruction explicite du fondateur → préfixer CI_LOCAL_SKIP_OK=1.")
    pr = json.loads(out)
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/\d+", pr.get("url", ""))
    if not m:
        block("gh-minutes-guard : URL de PR illisible — fusion bloquée (fail-closed).")
    slug, sha = m.group(1), pr["headRefOid"]
    rc, _ = gh(["api", f"repos/{slug}/contents/scripts/ci-local.sh", "--silent"], cwd)
    if rc != 0:
        print(f"gh-minutes-guard : {slug} n'a pas encore de miroir local (scripts/ci-local.sh). Fusion permise, mais R8.4 "
              "demande de le construire (modèle : RDP scripts/ci-local.sh) et de vérifier en local avant toute poussée.",
              file=sys.stderr)
        return
    rc, out = gh(["api", f"repos/{slug}/commits/{sha}/status",
                  "--jq", '[.statuses[] | select(.context == "ci-local") | .state] | first // "absent"'], cwd)
    if rc != 0:
        block(f"gh-minutes-guard : statut ci-local illisible pour {slug}@{sha[:8]} — fusion bloquée (fail-closed).")
    if out.strip('"') != "success":
        block(f"gh-minutes-guard : {slug}@{sha[:8]} n'a pas le statut ci-local=success (état : {out}). Lancez d'abord "
              "`bash scripts/ci-local.sh --post` sur la tête de la PR (miroir Linux local, gratuit), puis fusionnez. "
              "Exception : instruction explicite du fondateur → préfixer CI_LOCAL_SKIP_OK=1.")


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return
    if data.get("tool_name") != "Bash":
        return
    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    cwd = data.get("cwd")
    for seg in SPLIT.split(cmd):
        s = seg.strip()
        if re.search(LEAD + r"gh\s+(run\s+rerun|workflow\s+(run|enable))\b", s) and "GH_ACTIONS_OK=1" not in cmd:
            block("gh-minutes-guard : déclencher GitHub Actions à la main coûte des minutes payées par le fondateur (R8.4). "
                  "Vérifiez en local (scripts/ci-local.sh). Seulement sur demande explicite du fondateur → préfixer GH_ACTIONS_OK=1.")
        if re.search(LEAD + r"gh\s+pr\s+merge\b", s) and "CI_LOCAL_SKIP_OK=1" not in cmd:
            try:
                check_merge(s, cwd)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001 — un garde de fusion qui plante bloque (fail-closed)
                block(f"gh-minutes-guard : vérification impossible ({e.__class__.__name__}) — fusion bloquée (fail-closed).")


if __name__ == "__main__":
    main()
