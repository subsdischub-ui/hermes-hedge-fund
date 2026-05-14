"""LLM router — primary / secondary / escalation, OpenAI-compatible.

The router is the only place the agent code touches LLMs. Each role maps to a
target via config/default.yaml:agents. Falls back to "primary" when the chosen
target is not configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from openai import AsyncOpenAI
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


@dataclass(frozen=True)
class LLMTarget:
    name: Target
    base_url: str
    model: str
    api_key: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


class LLMRouter:
    """Routes chat/tool calls to the configured LLM target."""

    def __init__(self) -> None:
        s = get_settings()
        self.targets: dict[Target, LLMTarget] = {
            "primary": LLMTarget(
                "primary", s.llm_primary_base_url, s.llm_primary_model, s.llm_primary_api_key
            ),
            "secondary": LLMTarget(
                "secondary",
                s.llm_secondary_base_url,
                s.llm_secondary_model,
                s.llm_secondary_api_key,
            ),
            "escalation": LLMTarget(
                "escalation",
                s.llm_escalation_base_url,
                s.llm_escalation_model,
                s.llm_escalation_api_key,
            ),
        }
        self._agent_targets = s.yaml("agents", default={}) or {}
        self._clients: dict[Target, AsyncOpenAI] = {}

    def _client(self, t: Target) -> AsyncOpenAI:
        if t not in self._clients:
            tgt = self.targets[t]
            self._clients[t] = AsyncOpenAI(base_url=tgt.base_url, api_key=tgt.api_key or "x")
        return self._clients[t]

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
        client = self._client(target)
        model = self.targets[target].model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                resp = await client.chat.completions.create(**kwargs)
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
                    "target": target,
                    "model": model,
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
        """JSON-mode helper. Returns the parsed dict (raises on parse failure)."""
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
        return _safe_json(out["content"] or "{}")


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    return LLMRouter()
