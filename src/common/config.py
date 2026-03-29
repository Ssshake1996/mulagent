"""Configuration loader using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

def _find_project_root() -> Path:
    """Locate the mul-agent project root directory.

    Resolution order:
    1. MULAGENT_ROOT env var (explicit override)
    2. Walk up from CWD looking for config/settings.yaml
    3. Walk up from this source file (works for editable installs & dev)
    4. ~/.mulagent (user-level fallback for global pip install)
    """
    import os

    # 1. Explicit env var
    env_root = os.environ.get("MULAGENT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "config").is_dir():
            return p

    # 2. Walk up from CWD
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "config" / "settings.yaml").exists():
            return parent
        if parent == parent.parent:
            break

    # 3. Walk up from source file (editable install / dev)
    src_root = Path(__file__).resolve().parent.parent.parent
    if (src_root / "config").is_dir():
        return src_root

    # 4. User-level fallback
    user_dir = Path.home() / ".mulagent"
    if (user_dir / "config").is_dir():
        return user_dir

    # Last resort: source tree
    return src_root


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


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


class FeishuSettings(BaseSettings):
    app_id: str = ""
    app_secret: str = ""
    autostart: bool = False  # whether to enable systemd service on boot


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration.

    Supports any OpenAI-compatible embedding API (DashScope, OpenAI, local, etc.).
    When not configured, falls back to LLM-based keyword extraction + hashing.
    """
    model: str = ""           # e.g. "text-embedding-v3"
    api_key: str = ""
    base_url: str = ""        # e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dimensions: int = 1024    # output vector dimensions


class CompressSettings(BaseSettings):
    """Context compression thresholds and parameters."""
    # ── Facts 压缩 ──
    facts_compact_trigger: int = 15        # facts 超过此数触发压缩
    facts_keep_recent: int = 5             # 压缩时保留最近 N 条
    tool_result_max_tokens: int = 1500     # 单个工具结果截断上限 (token)

    # ── 上下文预算 ──
    context_max_chars: int = 8000          # 上下文字符预算 (0=自动: max_tokens*0.5*4)

    # ── 四级压缩阈值 (相关性分数) ──
    level_full: float = 0.7                # ≥0.7 → 完整保留
    level_summary: float = 0.3             # 0.3~0.7 → 摘要
    level_title: float = 0.1              # 0.1~0.3 → 仅标题
    # <0.1 → 隐藏

    # ── 相关性三信号权重 ──
    weight_keyword: float = 0.5            # 关键词重叠 (Jaccard)
    weight_recall: float = 0.3             # 召回意图检测
    weight_decay: float = 0.2              # 时间衰减

    # ── 话题归档 ──
    archive_threshold: int = 30            # 自动归档：超过 N 轮归档冷话题
    archive_manual_threshold: int = 6      # 手动压缩时的归档阈值
    decay_half_life_hours: float = 24.0    # 时间衰减半衰期（小时）


class ReactSettings(BaseSettings):
    """ReAct orchestrator configuration."""
    max_rounds: int = 30           # 最大推理轮数（复杂任务需要更多轮）
    timeout: int = 600             # 整体超时（秒）— 10 分钟
    tool_timeout: int = 120        # 单工具超时（秒）
    max_parallel_tools: int = 5    # 工具并行执行上限，1 = 串行
    max_conversation_pairs: int = 4  # 保留的对话轮数
    compress: CompressSettings = Field(default_factory=CompressSettings)


class SandboxSettings(BaseSettings):
    """Docker sandbox configuration for shell/code execution."""
    enabled: bool = True               # 启用 Docker 沙箱（不可用时自动降级）
    image: str = "python:3.12-slim"    # 沙箱镜像
    memory_limit: str = "512m"         # 内存限制
    cpu_limit: float = 1.0             # CPU 核数限制
    network: bool = True               # 是否允许网络访问
    workdir: str = "/workspace"        # 容器内工作目录


class SecuritySettings(BaseSettings):
    max_sandbox_timeout_seconds: int = 300
    max_sandbox_memory_mb: int = 512


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""
    enable_prometheus: bool = True     # Expose /metrics endpoint
    enable_tracing: bool = True        # Enable distributed tracing
    alert_webhook_url: str = ""        # Optional webhook for alerts
    metrics_retention_minutes: int = 60  # In-memory metrics window


class Settings(BaseSettings):
    app_name: str = "mul-agent"
    app_version: str = "0.1.0"
    debug: bool = True

    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    feishu: FeishuSettings = Field(default_factory=FeishuSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    react: ReactSettings = Field(default_factory=ReactSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    hooks: dict = Field(default_factory=dict)  # pre/post tool hooks


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
        feishu=FeishuSettings(**raw.get("feishu", {})),
        embedding=EmbeddingSettings(**raw.get("embedding", {})),
        react=ReactSettings(**raw.get("react", {})),
        sandbox=SandboxSettings(**raw.get("sandbox", {})),
        security=SecuritySettings(**raw.get("security", {})),
        observability=ObservabilitySettings(**raw.get("observability", {})),
        hooks=raw.get("hooks", {}),
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings()


def reload_settings() -> Settings:
    """Force reload settings from disk (config hot-reload).

    Clears the lru_cache and re-reads settings.yaml.
    Returns the new Settings object.
    """
    get_settings.cache_clear()
    return get_settings()
