"""Hybrid LLM routing: when DEEPSEEK_API_KEY is set, primary/secondary go to
DeepSeek V4 (Pro/Flash) and the local llama.cpp targets become automatic
fallback; absent the key, behaviour is unchanged (local only, no fallback)."""

from __future__ import annotations

from hermes.llm.router import DEEPSEEK, LLMTarget, resolve_targets


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
    assert active["primary"].model == "deepseek-v4-pro"
    assert active["secondary"].model == "deepseek-v4-flash"
    assert active["primary"].api_key == "sk-deepseek-123"
    # Local llama.cpp preserved as fallback for primary/secondary.
    assert fallback["primary"].model == "qwen-local"
    assert fallback["secondary"].base_url == "http://hermes-mac:11434/v1"


def test_blank_key_is_treated_as_absent():
    _active, fallback, on = resolve_targets(_local(), "   ", DEEPSEEK)
    assert on is False
    assert fallback == {}
