#!/usr/bin/env python3
"""Histoire-du-code LLM bridge for the git post-commit hook.

Reads a prompt on stdin, sends it through Hermès's own LLM router
(hermes.llm.router.get_router -> DeepSeek primary, local llama.cpp fallback),
and writes the model's text to stdout. Designed to be invoked by
.githooks/histoire-worker.sh via the project virtualenv, so it inherits the
project's settings/credentials. Fails closed: any error -> non-zero exit and
empty stdout, so the worker simply skips the doc update (never breaks a commit).
"""
from __future__ import annotations

import asyncio
import os
import sys

# Make the package importable whether or not it's installed editable.
ROOT = os.environ.get("HIST_ROOT") or os.getcwd()
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)
try:
    os.chdir(ROOT)  # so get_settings() finds .env / config relative to repo root
except OSError:
    pass

PROMPT = sys.stdin.read()
if not PROMPT.strip():
    sys.exit(1)

try:
    from hermes.llm.router import get_router
except Exception as e:  # noqa: BLE001 - bridge must never raise uncaught
    sys.stderr.write(f"import error: {e}\n")
    sys.exit(2)


async def _run() -> str:
    router = get_router()
    out = await router.chat(
        agent_role="documentarian",  # unconfigured role -> falls back to primary
        messages=[
            {
                "role": "system",
                "content": "Tu es un documentaliste technique francophone, concis et factuel.",
            },
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    return ((out or {}).get("content") or "").strip()


try:
    text = asyncio.run(asyncio.wait_for(_run(), timeout=120))
except Exception as e:  # noqa: BLE001
    sys.stderr.write(f"llm call error: {e}\n")
    sys.exit(3)

if not text:
    sys.exit(4)
sys.stdout.write(text)
