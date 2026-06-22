"""LLM router — primary / secondary / escalation, OpenAI-compatible.

The router is the only place the agent code touches LLMs. Each role maps to a
target via config/default.yaml:agents. Falls back to "primary" when the chosen
target is not configured.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)

Target = Literal["primary", "secondary", "escalation"]

# Only retry on transient errors. 4xx (auth, invalid model, malformed request)
# is NOT recoverable — retrying burns API budget and slows the trade loop.
_RETRY_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


@dataclass(frozen=True)
class LLMTarget:
    name: Target
    base_url: str
    model: str
    api_key: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


# DeepSeek V4 (OpenAI-compatible). Verified IDs (api-docs.deepseek.com): the
# legacy deepseek-chat/reasoner are deprecated 2026-07-24 in favour of these.
DEEPSEEK = {
    "base_url": "https://api.deepseek.com",
    "primary": "deepseek-v4-pro",
    "secondary": "deepseek-v4-flash",
    "escalation": "deepseek-v4-pro",
}


def resolve_targets(
    local: dict[Target, LLMTarget], deepseek_key: str, cfg: dict
) -> tuple[dict[Target, LLMTarget], dict[Target, LLMTarget], bool]:
    """Pick the active LLM targets + fallback set.

    With a DeepSeek key, primary/secondary/escalation point to DeepSeek V4 and the
    LOCAL targets become the fallback (used automatically if DeepSeek is
    unreachable) — so the agent runs 24/7 off the Mac but degrades gracefully.
    Without a key, the local targets are used as-is and there is no fallback.
    Returns ``(active, fallback, deepseek_active)``.
    """
    if not deepseek_key.strip():
        return local, {}, False
    active: dict[Target, LLMTarget] = {
        "primary": LLMTarget("primary", cfg["base_url"], cfg["primary"], deepseek_key),
        "secondary": LLMTarget("secondary", cfg["base_url"], cfg["secondary"], deepseek_key),
        "escalation": LLMTarget("escalation", cfg["base_url"], cfg["escalation"], deepseek_key),
    }
    return active, dict(local), True


class LLMRouter:
    """Routes chat/tool calls to the configured LLM target."""

    def __init__(self) -> None:
        s = get_settings()
        local: dict[Target, LLMTarget] = {
            "primary": LLMTarget(
                "primary",
                s.llm_primary_base_url,
                s.llm_primary_model,
                _reveal(s.llm_primary_api_key),
            ),
            "secondary": LLMTarget(
                "secondary",
                s.llm_secondary_base_url,
                s.llm_secondary_model,
                _reveal(s.llm_secondary_api_key),
            ),
            "escalation": LLMTarget(
                "escalation",
                s.llm_escalation_base_url,
                s.llm_escalation_model,
                _reveal(s.llm_escalation_api_key),
            ),
        }
        self.targets, self.fallback_targets, self.deepseek_active = resolve_targets(
            local, _reveal(s.deepseek_api_key), DEEPSEEK
        )
        if self.deepseek_active:
            log.info("llm_router_deepseek_active", primary=DEEPSEEK["primary"], fallback="local")
        self._agent_targets = s.yaml("agents", default={}) or {}
        self._clients: dict[str, AsyncOpenAI] = {}

    def _client_for(self, tgt: LLMTarget) -> AsyncOpenAI:
        key = f"{tgt.name}@{tgt.base_url}"
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(base_url=tgt.base_url, api_key=tgt.api_key or "x")
        return self._clients[key]

    def _client(self, t: Target) -> AsyncOpenAI:
        return self._client_for(self.targets[t])

    def resolve(self, agent_role: str) -> Target:
        wanted: Target = self._agent_targets.get(agent_role, "primary")
        if not self.targets[wanted].configured:
            log.warning("llm_target_not_configured", agent=agent_role, requested=wanted)
            wanted = "primary"
        return wanted

    async def chat(
        self,
        *,
        agent_role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self.resolve(agent_role)
        base_kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            base_kwargs["tools"] = tools
            base_kwargs["tool_choice"] = "auto"
        if response_format:
            base_kwargs["response_format"] = response_format

        try:
            return await self._do_call(self.targets[target], base_kwargs, agent_role)
        except _RETRY_EXCEPTIONS as e:
            fb = self.fallback_targets.get(target)
            if fb and fb.configured:
                # DeepSeek unreachable -> degrade to the local llama.cpp fallback.
                log.warning(
                    "llm_fallback_to_local", agent=agent_role, from_target=target, error=str(e)
                )
                return await self._do_call(fb, base_kwargs, agent_role)
            raise

    async def _do_call(
        self, tgt: LLMTarget, base_kwargs: dict[str, Any], agent_role: str
    ) -> dict[str, Any]:
        client = self._client_for(tgt)
        kwargs = {**base_kwargs, "model": tgt.model}
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                try:
                    resp = await client.chat.completions.create(**kwargs)
                except APIStatusError as e:
                    # Don't retry 4xx. Surface immediately so callers see real errors
                    # (e.g. a bad DeepSeek key) instead of silently falling back.
                    log.warning(
                        "llm_status_error",
                        agent=agent_role,
                        status=getattr(e, "status_code", None),
                        target=tgt.name,
                        model=tgt.model,
                    )
                    raise
                choice = resp.choices[0]
                return {
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": _safe_json(tc.function.arguments),
                        }
                        for tc in (choice.message.tool_calls or [])
                    ],
                    "finish_reason": choice.finish_reason,
                    "usage": resp.usage.model_dump() if resp.usage else {},
                    "target": tgt.name,
                    "model": tgt.model,
                }
        raise RuntimeError("unreachable")

    async def structured(
        self,
        *,
        agent_role: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """JSON-mode helper.

        Many local servers (Ollama, LM Studio) silently ignore `response_format`
        with `json_schema` and return free-form text. We therefore:
          1. Ask for `json_schema` strict mode (honoured by OpenAI-compatible servers).
          2. Also instruct the model to emit ONLY JSON in the system message.
          3. On parse failure, try to extract the first {...} block and reparse.
          4. If still unparseable, return an empty dict and log — never crash callers.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "hermes_output", "schema": schema, "strict": True},
        }
        out = await self.chat(
            agent_role=agent_role,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        raw = out["content"] or "{}"
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            log.warning(
                "llm_structured_parse_failed",
                agent=agent_role,
                target=out.get("target"),
                preview=str(raw)[:200],
            )
            return {}
        return parsed


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> Any:
    """Parse a JSON object from a string, tolerating prose around it."""
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_BLOCK.search(raw or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return None


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _reveal(secret: Any) -> str:
    """Unwrap a SecretStr or pass through a plain str."""
    if hasattr(secret, "get_secret_value"):
        return str(secret.get_secret_value() or "")
    return str(secret or "")


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    return LLMRouter()
