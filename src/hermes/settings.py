"""Pydantic-backed settings — env vars + config/default.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class LLMRole(BaseSettings):
    base_url: str
    model: str
    api_key: str = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    # Runtime
    stage: str = Field("stage_0_paper", alias="HERMES_STAGE")
    data_dir: Path = Field(Path("/data"), alias="HERMES_DATA_DIR")
    log_level: str = Field("INFO", alias="HERMES_LOG_LEVEL")
    tz: str = Field("America/Toronto", alias="HERMES_TZ")
    bankroll_usd: float = Field(100.0, alias="HERMES_BANKROLL_USD")
    universe_raw: str = Field("BTC/USDC,ETH/USDC,SOL/USDC", alias="HERMES_UNIVERSE")
    max_open_positions: int = Field(3, alias="HERMES_MAX_OPEN_POSITIONS")
    daily_loss_kill_pct: float = Field(2.0, alias="HERMES_DAILY_LOSS_KILL_PCT")

    # LLM
    llm_primary_base_url: str = Field("http://hermes-mac:11434/v1", alias="LLM_PRIMARY_BASE_URL")
    llm_primary_model: str = Field("qwen3-coder-next", alias="LLM_PRIMARY_MODEL")
    llm_primary_api_key: str = Field("ollama", alias="LLM_PRIMARY_API_KEY")
    llm_secondary_base_url: str = Field("http://hermes-mac:11434/v1", alias="LLM_SECONDARY_BASE_URL")
    llm_secondary_model: str = Field("qwen3.6:27b", alias="LLM_SECONDARY_MODEL")
    llm_secondary_api_key: str = Field("ollama", alias="LLM_SECONDARY_API_KEY")
    llm_escalation_base_url: str = Field("", alias="LLM_ESCALATION_BASE_URL")
    llm_escalation_model: str = Field("", alias="LLM_ESCALATION_MODEL")
    llm_escalation_api_key: str = Field("", alias="LLM_ESCALATION_API_KEY")

    # TRM
    trm_enabled: bool = Field(True, alias="TRM_ENABLED")
    trm_checkpoint: Path = Field(Path("/data/trm/latest.pt"), alias="TRM_CHECKPOINT")
    trm_confidence_floor: float = Field(0.55, alias="TRM_CONFIDENCE_FLOOR")

    # Exchanges
    exchange_primary: str = Field("hyperliquid", alias="EXCHANGE_PRIMARY")
    exchange_backup: str = Field("bybit", alias="EXCHANGE_BACKUP")
    hl_testnet: bool = Field(True, alias="HL_TESTNET")
    hl_account_address: str = Field("", alias="HL_ACCOUNT_ADDRESS")
    hl_secret_key: str = Field("", alias="HL_SECRET_KEY")
    bybit_api_key: str = Field("", alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field("", alias="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(True, alias="BYBIT_TESTNET")

    # Data
    cryptopanic_token: str = Field("", alias="CRYPTOPANIC_TOKEN")
    etherscan_api_key: str = Field("", alias="ETHERSCAN_API_KEY")
    coingecko_api_key: str = Field("", alias="COINGECKO_API_KEY")

    # Infra
    redis_url: str = Field("redis://hermes-redis:6379/0", alias="REDIS_URL")
    database_url: str = Field("sqlite+aiosqlite:////data/hermes.db", alias="DATABASE_URL")
    vector_dir: Path = Field(Path("/data/chroma"), alias="VECTOR_DIR")

    # Loaded from YAML
    yaml_config: dict[str, Any] = Field(default_factory=dict)

    @property
    def universe(self) -> list[str]:
        return [s.strip() for s in self.universe_raw.split(",") if s.strip()]

    @property
    def stage_config(self) -> dict[str, Any]:
        return self.yaml_config.get("stages", {}).get(self.stage, {})

    def yaml(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.yaml_config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return Settings(yaml_config=cfg)  # type: ignore[arg-type]
