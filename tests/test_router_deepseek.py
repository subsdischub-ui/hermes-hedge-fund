"""Hybrid LLM routing: when DEEPSEEK_API_KEY is set, primary/secondary go to
DeepSeek V4 (Pro/Flash) and the local llama.cpp targets become automatic
fallback; absent the key, behaviour is unchanged (local only, no fallback)."""

from __future__ import annotations

import httpx
import pytest
from openai import APIStatusError

from hermes.llm.router import DEEPSEEK, LLMRouter, LLMTarget, resolve_targets


def _local():
    return {
        "primary": LLMTarget("primary", "http://hermes-mac:11434/v1", "qwen-local", "ollama"),
        "secondary": LLMTarget("secondary", "http://hermes-mac:11434/v1", "qwen-sm", "ollama"),
        "escalation": LLMTarget("escalation", "", "", ""),
    }


def test_no_key_keeps_local_and_no_fallback():
    active, fallback, on = resolve_targets(_local(), "", DEEPSEEK)
    assert on is False
    assert active["primary"].model == "qwen-local"
    assert active["primary"].base_url == "http://hermes-mac:11434/v1"
    assert fallback == {}


def test_key_activates_deepseek_primary_secondary_with_local_fallback():
    loc = _local()
    active, fallback, on = resolve_targets(loc, "sk-deepseek-123", DEEPSEEK)
    assert on is True
    assert active["primary"].base_url == "https://api.deepseek.com"
    # Cerveau principal basculé sur Flash (décision opérateur 2026-08-01, coût) —
    # secondary reste flash aussi (déjà le cas), escalation reste pro (filet de
    # sécurité pour les cas rares à fort enjeu, non demandé au changement).
    assert active["primary"].model == "deepseek-v4-flash"
    assert active["secondary"].model == "deepseek-v4-flash"
    assert active["escalation"].model == "deepseek-v4-pro"
    assert active["primary"].api_key == "sk-deepseek-123"
    # Local llama.cpp preserved as fallback for primary/secondary.
    assert fallback["primary"].model == "qwen-local"
    assert fallback["secondary"].base_url == "http://hermes-mac:11434/v1"


def test_blank_key_is_treated_as_absent():
    _active, fallback, on = resolve_targets(_local(), "   ", DEEPSEEK)
    assert on is False
    assert fallback == {}


@pytest.mark.asyncio
async def test_structured_degrades_json_schema_to_json_object_on_400():
    """DeepSeek V4 rejects json_schema with 400 — structured() must fall back to
    json_object (then plain) so the LLM agents never hard-fail on it."""
    r = LLMRouter.__new__(LLMRouter)  # bypass __init__ (we override chat directly)
    seen: list[str] = []

    async def fake_chat(*, agent_role, messages, response_format=None, temperature=0.1):
        seen.append(response_format.get("type") if response_format else "none")
        if response_format and response_format.get("type") == "json_schema":
            req = httpx.Request("POST", "https://api.deepseek.com/chat")
            raise APIStatusError(
                "This response_format type is unavailable now",
                response=httpx.Response(400, request=req), body=None,
            )
        return {"content": '{"side": "BUY"}', "target": "primary", "model": "deepseek-v4-pro"}

    r.chat = fake_chat  # type: ignore[method-assign]
    out = await r.structured(agent_role="researcher", messages=[{"role": "user", "content": "x"}], schema={})
    assert out == {"side": "BUY"}
    assert seen[0] == "json_schema" and seen[1] == "json_object"  # degraded in order
