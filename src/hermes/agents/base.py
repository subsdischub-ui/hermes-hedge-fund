"""Shared utilities for agent nodes."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
