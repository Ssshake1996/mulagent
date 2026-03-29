"""LLM client factory — creates LangChain ChatModels from multi-model config."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from common.config import ModelConfig, get_settings


def create_llm(
    model_config: ModelConfig,
    max_tokens_override: int | None = None,
    extra_body: dict | None = None,
) -> ChatOpenAI:
    """Create a LangChain ChatModel from a single ModelConfig."""
    kwargs = {"model": model_config.model, "max_tokens": max_tokens_override or model_config.max_tokens}

    if model_config.api_key:
        kwargs["api_key"] = model_config.api_key
    if model_config.base_url:
        kwargs["base_url"] = model_config.base_url
    if model_config.temperature is not None:
        kwargs["temperature"] = model_config.temperature
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


class LLMManager:
    """Manages multiple LLM instances, lazily created from config."""

    def __init__(self):
        self._settings = get_settings().llm
        self._cache: dict[str, ChatOpenAI] = {}

    def get(self, model_id: str | None = None, max_tokens: int | None = None) -> ChatOpenAI | None:
        """Get an LLM by model_id, or the default. Returns None if not configured."""
        key = model_id or self._settings.default
        if not key:
            return None

        cache_key = f"{key}:{max_tokens}" if max_tokens else key
        if cache_key not in self._cache:
            config = self._settings.get_model(key)
            if config is None:
                return None
            self._cache[cache_key] = create_llm(
                config,
                max_tokens_override=max_tokens,
                extra_body={"enable_thinking": False},
            )

        return self._cache[cache_key]

    @property
    def default(self) -> ChatOpenAI | None:
        return self.get()

    def list_models(self) -> list[dict[str, str]]:
        return self._settings.list_models()

    @property
    def default_id(self) -> str:
        return self._settings.default
