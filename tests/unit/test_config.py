"""Tests for configuration loading."""

from pathlib import Path

from common.config import Settings, LLMSettings, ModelConfig, load_settings, CONFIG_DIR


def test_load_settings_from_yaml():
    """Should load settings from the default YAML config file."""
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    assert settings.app_name == "mul-agent"
    assert settings.server.port == 8000
    assert "postgresql" in settings.database.url
    assert "redis" in settings.redis.url
    assert settings.qdrant.collection_name == "case_library"
    assert settings.llm.default  # should have a default model
    assert len(settings.llm.models) >= 1  # at least one model configured


def test_load_settings_missing_file():
    """Should return defaults when config file doesn't exist."""
    settings = load_settings(Path("/nonexistent/settings.yaml"))
    assert settings.app_name == "mul-agent"
    assert settings.server.host == "0.0.0.0"


def test_settings_nested_structure():
    settings = load_settings()
    assert isinstance(settings.server.port, int)
    assert isinstance(settings.database.echo, bool)


def test_default_settings():
    settings = Settings()
    assert settings.debug is True
    assert settings.server.port == 8000


def test_multi_model_config():
    """LLMSettings should support multiple models."""
    llm = LLMSettings(
        default="qwen",
        models={
            "qwen": ModelConfig(name="Qwen", model="qwen3.5-plus", api_key="sk-xxx"),
            "gpt": ModelConfig(name="GPT-4o", model="gpt-4o", api_key="sk-yyy"),
        },
    )
    assert llm.get_model("qwen").model == "qwen3.5-plus"
    assert llm.get_model("gpt").model == "gpt-4o"
    assert llm.get_model().model == "qwen3.5-plus"  # default
    assert llm.get_model("nonexistent") is None
    assert len(llm.list_models()) == 2


def test_model_config_optional_temperature():
    """Temperature should be optional (None = don't send)."""
    m1 = ModelConfig(model="test")
    assert m1.temperature is None
    m2 = ModelConfig(model="test", temperature=0.5)
    assert m2.temperature == 0.5


def test_load_qwen_from_yaml():
    """Should correctly load the qwen model from settings.yaml."""
    settings = load_settings(CONFIG_DIR / "settings.yaml")
    qwen = settings.llm.get_model("qwen")
    assert qwen is not None
    assert qwen.model == "qwen3.5-plus"
    assert qwen.base_url  # should have base_url set
