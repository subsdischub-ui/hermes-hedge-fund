"""Garde des minutes GitHub, copie DU DÉPÔT (directive du fondateur, 2026-10-20, ~/.claude/CLAUDE.md R8 point 4).

Les séances cloud (claude.ai/code) ne lisent pas ~/.claude du Mac : seul `.claude/settings.json` du dépôt les protège. Ce test
exécute le VRAI garde (`.claude/hooks/gh-minutes-guard.py`, jamais une copie de sa règle) et vérifie que `.claude/settings.json`
le déclare. Aucune commande ici n'interroge `gh` : pas de réseau.

Lancer : `uv run pytest tests/test_gh_minutes_guard.py -q` depuis la racine du dépôt.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HOOK = REPO_ROOT / ".claude" / "hooks" / "gh-minutes-guard.py"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"


def run_hook(command: str) -> subprocess.CompletedProcess[str]:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )


def test_blocks_gh_run_rerun() -> None:
    r = run_hook("gh run rerun 1")
    assert r.returncode == 2
    assert "minutes" in r.stderr


def test_blocks_gh_workflow_run() -> None:
    assert run_hook("gh workflow run tests.yml").returncode == 2


def test_lets_git_status_through() -> None:
    assert run_hook("git status").returncode == 0


def test_explicit_override_lets_through() -> None:
    assert run_hook("GH_ACTIONS_OK=1 gh run rerun 1").returncode == 0


def test_quoted_gh_command_is_not_a_command() -> None:
    assert run_hook('echo "gh run rerun 1"').returncode == 0


def test_script_stays_runnable_on_python_39() -> None:
    assert "from __future__ import annotations" in HOOK.read_text(encoding="utf-8").splitlines()


def test_settings_declares_the_guard() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    commands = [
        h["command"]
        for entry in settings["hooks"]["PreToolUse"]
        if entry["matcher"] == "Bash"
        for h in entry["hooks"]
    ]
    assert 'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/gh-minutes-guard.py"' in commands
    assert "/usr/bin/python3" not in "\n".join(commands)
