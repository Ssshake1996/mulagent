"""Tests for Docker sandbox module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tools.sandbox import is_docker_available, execute_sandboxed, run_in_sandbox


@pytest.fixture(autouse=True)
def reset_docker_cache():
    """Reset the Docker availability cache between tests."""
    import tools.sandbox as sb
    sb._docker_available = None
    sb._image_pulled = False
    yield
    sb._docker_available = None
    sb._image_pulled = False


@pytest.mark.asyncio
async def test_docker_not_available_no_binary():
    """If docker binary is missing, should return False."""
    with patch("shutil.which", return_value=None):
        result = await is_docker_available()
        assert result is False


@pytest.mark.asyncio
async def test_docker_available_cached():
    """Second call should use cached value."""
    import tools.sandbox as sb
    sb._docker_available = True
    assert await is_docker_available() is True


@pytest.mark.asyncio
async def test_execute_sandboxed_disabled():
    """When sandbox is disabled, should return used_sandbox=False."""
    with patch("tools.sandbox.get_settings") as mock_settings:
        mock_cfg = MagicMock()
        mock_cfg.sandbox.enabled = False
        mock_settings.return_value = mock_cfg
        used, rc, out, err = await execute_sandboxed("echo hi", mode="shell")
        assert used is False


@pytest.mark.asyncio
async def test_execute_sandboxed_no_docker():
    """When Docker is unavailable, should return used_sandbox=False."""
    with patch("tools.sandbox.is_docker_available", new_callable=AsyncMock, return_value=False), \
         patch("tools.sandbox.get_settings") as mock_settings:
        mock_cfg = MagicMock()
        mock_cfg.sandbox.enabled = True
        mock_settings.return_value = mock_cfg
        used, rc, out, err = await execute_sandboxed("echo hi", mode="shell")
        assert used is False
