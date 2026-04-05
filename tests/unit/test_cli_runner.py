"""Tests for cli.runner — AgentRunner initialization and model switching."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from cli.runner import AgentRunner


@pytest.fixture
def mock_settings():
    """Patch get_settings to return a minimal mock."""
    settings = MagicMock()
    settings.database.url = ""  # No DB
    settings.redis.url = "redis://localhost:6379/0"
    settings.qdrant.url = "http://localhost:6333"
    settings.qdrant.collection_name = "test_collection"
    settings.react.timeout = 60
    settings.embedding.model = ""
    settings.embedding.dimensions = 1024
    return settings


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="test reply"))
    return llm


@pytest.fixture
def mock_llm_manager(mock_llm):
    mgr = MagicMock()
    mgr.default_id = "test_model"
    mgr.default = mock_llm
    mgr.get.return_value = mock_llm
    mgr.list_models.return_value = [{"id": "test_model", "model": "test"}]
    return mgr


@pytest.fixture
def runner(mock_settings, mock_llm_manager, tmp_path):
    """Create AgentRunner with all external deps mocked."""
    with (
        patch("cli.runner.get_settings", return_value=mock_settings),
        patch("cli.runner.reload_settings"),
        patch("cli.runner.LLMManager", return_value=mock_llm_manager),
        patch("cli.runner.get_qdrant_client") as mock_qdrant,
        patch("cli.runner.ensure_collection"),
        patch("cli.runner.set_shared_llm"),
        patch("cli.runner.create_session_factory", side_effect=Exception("no db")),
        patch("cli.runner.ConversationStore") as mock_conv,
    ):
        # Qdrant mock — returns (client, is_remote) tuple
        qdrant_client = MagicMock()
        mock_qdrant.return_value = (qdrant_client, True)

        # ConversationStore mock
        conv_instance = MagicMock()
        conv_instance.load.return_value = None
        conv_instance.create.return_value = {"session_id": "s1", "user_id": "u1", "turns": []}
        conv_instance.get_history_for_prompt.return_value = ""
        conv_instance.get_directives.return_value = []
        conv_instance.list_sessions.return_value = []
        mock_conv.return_value = conv_instance

        r = AgentRunner()
        yield r


# ── Initialization ───────────────────────────────────────────────

def test_runner_init_sets_model(runner):
    assert runner.current_model == "test_model"


def test_runner_has_session_manager(runner):
    assert runner.session_manager is not None


def test_runner_has_llm_manager(runner):
    assert runner.llm_manager is not None


def test_runner_db_unavailable(runner):
    """DB failure should not prevent runner creation."""
    assert runner.db_session_factory is None


# ── Model switching ──────────────────────────────────────────────

def test_switch_model_success(runner, mock_llm_manager, mock_llm):
    mock_llm_manager.get.return_value = mock_llm
    ok = runner.switch_model("other_model")
    assert ok is True
    assert runner.current_model == "other_model"


def test_switch_model_invalid(runner, mock_llm_manager):
    mock_llm_manager.get.return_value = None
    ok = runner.switch_model("nonexistent")
    assert ok is False
    assert runner.current_model == "test_model"


# ── run() ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_calls_react(runner):
    with patch("cli.runner.run_react", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_output": "hello",
            "status": "completed",
            "intent": "react",
            "directives": [],
        }
        result = await runner.run("test input", "session_1")
        assert result["final_output"] == "hello"
        assert result["status"] == "completed"
        mock_react.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_persists_turns(runner):
    conv = runner.session_manager.conv_store
    with patch("cli.runner.run_react", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_output": "response",
            "status": "completed",
            "intent": "react",
            "directives": [],
        }
        await runner.run("user msg", "session_1")

        # Should have called append_turn for user and assistant
        calls = conv.append_turn.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("session_1", "user", "user msg")
        assert calls[1].args == ("session_1", "assistant", "response")


@pytest.mark.asyncio
async def test_run_saves_directives(runner):
    conv = runner.session_manager.conv_store
    with patch("cli.runner.run_react", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_output": "ok",
            "status": "completed",
            "intent": "react",
            "directives": ["rule1", "rule2"],
        }
        await runner.run("test", "session_1")
        conv.save_directives.assert_called_once_with("session_1", ["rule1", "rule2"])
