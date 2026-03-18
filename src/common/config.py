"""Configuration loader using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseSettings(BaseSettings):
    url: str = "postgresql+asyncpg://mulagent:mulagent@localhost:5432/mulagent"
    echo: bool = False


class RedisSettings(BaseSettings):
    url: str = "redis://localhost:6379/0"


class QdrantSettings(BaseSettings):
    url: str = "http://localhost:6333"
    collection_name: str = "case_library"


class ModelConfig(BaseSettings):
    """Single model configuration."""
    name: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float | None = None  # None = don't send, let provider decide
    max_tokens: int = 4096


class LLMSettings(BaseSettings):
    """Multi-model LLM configuration.

    models: dict of model_id → ModelConfig
    default: which model_id to use by default
    """
    default: str = ""
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    def get_model(self, model_id: str | None = None) -> ModelConfig | None:
        """Get a model config by ID, or the default."""
        key = model_id or self.default
        return self.models.get(key)

    def list_models(self) -> list[dict[str, str]]:
        """List all available models."""
        return [
            {"id": mid, "name": m.name or mid, "model": m.model}
            for mid, m in self.models.items()
        ]


class OpenClawSettings(BaseSettings):
    enabled: bool = False
    timeout: int = 120


class FeishuSettings(BaseSettings):
    app_id: str = ""
    app_secret: str = ""
    autostart: bool = False  # whether to enable systemd service on boot


class SecuritySettings(BaseSettings):
    max_sandbox_timeout_seconds: int = 30
    max_sandbox_memory_mb: int = 512


class Settings(BaseSettings):
    app_name: str = "mul-agent"
    app_version: str = "0.1.0"
    debug: bool = True

    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    openclaw: OpenClawSettings = Field(default_factory=OpenClawSettings)
    feishu: FeishuSettings = Field(default_factory=FeishuSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)


def _parse_llm_settings(raw_llm: dict[str, Any]) -> LLMSettings:
    """Parse the multi-model LLM config from YAML."""
    default = raw_llm.get("default", "")
    raw_models = raw_llm.get("models", {})

    models = {}
    for mid, mcfg in raw_models.items():
        if isinstance(mcfg, dict):
            models[mid] = ModelConfig(**mcfg)

    return LLMSettings(default=default, models=models)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML config file, with env var overrides."""
    path = config_path or CONFIG_DIR / "settings.yaml"
    raw = _load_yaml(path)

    app_cfg = raw.get("app", {})
    return Settings(
        app_name=app_cfg.get("name", "mul-agent"),
        app_version=app_cfg.get("version", "0.1.0"),
        debug=app_cfg.get("debug", True),
        server=ServerSettings(**raw.get("server", {})),
        database=DatabaseSettings(**raw.get("database", {})),
        redis=RedisSettings(**raw.get("redis", {})),
        qdrant=QdrantSettings(**raw.get("qdrant", {})),
        llm=_parse_llm_settings(raw.get("llm", {})),
        openclaw=OpenClawSettings(**raw.get("openclaw", {})),
        feishu=FeishuSettings(**raw.get("feishu", {})),
        security=SecuritySettings(**raw.get("security", {})),
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings()
