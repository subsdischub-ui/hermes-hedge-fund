"""OpenAI-style tool/function schemas exposed to the LLM."""

from __future__ import annotations

from typing import Any


def tool(name: str, description: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": [k for k, v in params.items() if not v.get("optional")],
                "additionalProperties": False,
            },
        },
    }


TRADER_TOOLS = [
    tool(
        "get_market_snapshot",
        "Return latest OHLCV, ATR, key indicators for a symbol.",
        {
            "symbol": {"type": "string", "description": "e.g. BTC/USDC"},
            "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]},
            "bars": {"type": "integer", "minimum": 30, "maximum": 500},
        },
    ),
    tool(
        "get_news",
        "Recent crypto news headlines for the symbol (CryptoPanic).",
        {"symbol": {"type": "string"}, "limit": {"type": "integer", "optional": True}},
    ),
    tool(
        "get_open_positions",
        "Current open positions and unrealized PnL.",
        {},
    ),
    tool(
        "search_lessons",
        "Retrieve top-k lessons learned from prior trades, by semantic similarity.",
        {"query": {"type": "string"}, "k": {"type": "integer", "optional": True}},
    ),
    tool(
        "propose_trade",
        "Submit a draft trade proposal to the risk engine + TRM verifier. Returns approved or rejected with reason.",
        {
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "thesis": {"type": "string", "description": "Why this trade — 2–4 sentences."},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "stop_pct": {"type": "number", "description": "Stop distance as % of entry."},
            "take_profit_pct": {"type": "number", "description": "TP distance as % of entry."},
            "skill_id": {"type": "string", "description": "Skill template id, e.g. mean_reversion_atr"},
        },
    ),
]
