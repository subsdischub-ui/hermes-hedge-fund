"""Pydantic-backed settings — env vars + config/default.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # `forbid` would surface typos like HERMES_BANKROL_USD; pydantic-settings
        # also reads OS env which has many unrelated vars, so we ignore extras —
        # but we emit a startup-time warning by checking for HERMES_-prefixed
        # vars that didn't match a field (handled in api.lifespan).
        extra="ignore",
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
    # API key for the FastAPI surface (x-api-key). Empty = open (paper convenience);
    # set HERMES_API_KEY before any live stage to lock /tick + /learn/* down.
    api_key: SecretStr = Field(SecretStr(""), alias="HERMES_API_KEY")

    # LLM
    llm_primary_base_url: str = Field("http://hermes-mac:11434/v1", alias="LLM_PRIMARY_BASE_URL")
    llm_primary_model: str = Field("qwen3-coder-next", alias="LLM_PRIMARY_MODEL")
    llm_primary_api_key: SecretStr = Field(SecretStr("ollama"), alias="LLM_PRIMARY_API_KEY")
    llm_secondary_base_url: str = Field("http://hermes-mac:11434/v1", alias="LLM_SECONDARY_BASE_URL")
    llm_secondary_model: str = Field("qwen3.6:27b", alias="LLM_SECONDARY_MODEL")
    llm_secondary_api_key: SecretStr = Field(SecretStr("ollama"), alias="LLM_SECONDARY_API_KEY")
    llm_escalation_base_url: str = Field("", alias="LLM_ESCALATION_BASE_URL")
    llm_escalation_model: str = Field("", alias="LLM_ESCALATION_MODEL")
    llm_escalation_api_key: SecretStr = Field(SecretStr(""), alias="LLM_ESCALATION_API_KEY")
    # When set, the router uses DeepSeek (V4 Pro primary / Flash secondary) and
    # keeps the local llama.cpp targets above as automatic fallback. 24/7 off-Mac.
    deepseek_api_key: SecretStr = Field(SecretStr(""), alias="DEEPSEEK_API_KEY")

    # TRM
    trm_enabled: bool = Field(True, alias="TRM_ENABLED")
    trm_checkpoint: Path = Field(Path("/data/trm/latest.pt"), alias="TRM_CHECKPOINT")
    trm_confidence_floor: float = Field(0.55, alias="TRM_CONFIDENCE_FLOOR")

    # Exchanges. Primary = the data venue (OHLCV + reference prices). Default is
    # Binance USDT-M futures: the operator's actual account, it carries perps +
    # funding for the whole universe, AND it is the venue the strategies were
    # backtested on (so live matches the validated backtest). yaml `data_venue:`
    # drives it; EXCHANGE_PRIMARY env overrides.
    exchange_primary_raw: str = Field("binanceusdm", alias="EXCHANGE_PRIMARY")
    exchange_backup: str = Field("bybit", alias="EXCHANGE_BACKUP")
    # Venue the paper book prices fills at — see the paper_price_venue property
    # (defaults to the data venue so fills + OHLCV come from the SAME exchange).
    paper_price_venue_raw: str = Field("", alias="PAPER_PRICE_VENUE")
    hl_testnet: bool = Field(True, alias="HL_TESTNET")
    hl_account_address: str = Field("", alias="HL_ACCOUNT_ADDRESS")
    hl_secret_key: SecretStr = Field(SecretStr(""), alias="HL_SECRET_KEY")
    bybit_api_key: SecretStr = Field(SecretStr(""), alias="BYBIT_API_KEY")
    bybit_api_secret: SecretStr = Field(SecretStr(""), alias="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(True, alias="BYBIT_TESTNET")

    # Data
    cryptopanic_token: SecretStr = Field(SecretStr(""), alias="CRYPTOPANIC_TOKEN")
    etherscan_api_key: SecretStr = Field(SecretStr(""), alias="ETHERSCAN_API_KEY")
    coingecko_api_key: SecretStr = Field(SecretStr(""), alias="COINGECKO_API_KEY")

    # Infra
    redis_url: str = Field("redis://hermes-redis:6379/0", alias="REDIS_URL")
    database_url: str = Field("sqlite+aiosqlite:////data/hermes.db", alias="DATABASE_URL")
    vector_dir: Path = Field(Path("/data/chroma"), alias="VECTOR_DIR")

    # Loaded from YAML
    yaml_config: dict[str, Any] = Field(default_factory=dict)

    # --- properties ---------------------------------------------------------
    @property
    def universe(self) -> list[str]:
        # The traded universe is STRATEGY config tightly coupled to the validated
        # code (symbols must match the data feed + skills), so the yaml `universe:`
        # list is AUTHORITATIVE and deploys with the repo. HERMES_UNIVERSE (env) is
        # only a fallback when the yaml list is absent — this deliberately ignores a
        # stale boilerplate env var so the code-controlled universe always wins.
        y = self.yaml("universe")
        if isinstance(y, list) and y:
            return [str(x).strip() for x in y if str(x).strip()]
        return [s.strip() for s in self.universe_raw.split(",") if s.strip()]

    @property
    def exchange_primary(self) -> str:
        # Same rationale as `universe`: the data venue is code-controlled — yaml
        # `data_venue:` is authoritative, EXCHANGE_PRIMARY env is the absent-yaml
        # fallback only (so a stale env var can't pin us to the wrong venue).
        y = self.yaml("data_venue")
        if isinstance(y, str) and y.strip():
            return y.strip()
        return self.exchange_primary_raw

    @property
    def paper_price_venue(self) -> str:
        # Price the paper book at the SAME venue it pulls OHLCV from, so fills and
        # the prices the strategy sees are consistent (a stale PAPER_PRICE_VENUE=kraken
        # env that lacks the Binance perp symbols is deliberately ignored — yaml
        # `paper_price_venue:` overrides if you really need a different venue).
        y = self.yaml("paper_price_venue")
        if isinstance(y, str) and y.strip():
            return y.strip()
        return self.exchange_primary

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

    # --- validators ---------------------------------------------------------
    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, v: str) -> str:
        allowed = {"stage_0_paper", "stage_1_micro", "stage_2_normal", "stage_3_perps"}
        if v not in allowed:
            raise ValueError(f"HERMES_STAGE must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("bankroll_usd")
    @classmethod
    def _validate_bankroll(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("HERMES_BANKROLL_USD must be positive")
        return v

    @field_validator("daily_loss_kill_pct")
    @classmethod
    def _validate_kill_pct(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError("HERMES_DAILY_LOSS_KILL_PCT must be in (0, 100]")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return Settings(yaml_config=cfg)  # type: ignore[arg-type]
