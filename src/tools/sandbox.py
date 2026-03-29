"""Docker sandbox for safe command and code execution.

Provides isolated execution of shell commands and Python code inside
Docker containers. Falls back to direct execution when Docker is unavailable.

Key safety features:
- Memory and CPU limits
- Network isolation (configurable)
- Read-only filesystem except /workspace and /tmp
- No privileged access
- Auto-cleanup of containers

Enhanced features:
- Persistent workspace: mount a host directory for cross-execution persistence
- Multi-file project execution: mount project directories
- Pre-installed dependencies: pip install before execution
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from common.config import get_settings

logger = logging.getLogger(__name__)

# Cache Docker availability check
_docker_available: bool | None = None
_image_pulled: bool = False

# Persistent workspace directory (per-session)
from common.config import DATA_DIR
_WORKSPACE_BASE = DATA_DIR / "workspaces"


async def is_docker_available() -> bool:
    """Check if Docker daemon is accessible."""
    global _docker_available
    if _docker_available is not None:
        return _docker_available

    if not shutil.which("docker"):
        _docker_available = False
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info", "--format", "{{.ServerVersion}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        _docker_available = proc.returncode == 0
        if _docker_available:
            logger.info("Docker available: v%s", stdout.decode().strip())
        else:
            logger.info("Docker not accessible (returncode=%d)", proc.returncode)
    except Exception as e:
        logger.info("Docker not available: %s", e)
        _docker_available = False

    return _docker_available


async def _ensure_image() -> bool:
    """Pull sandbox image if not already present."""
    global _image_pulled
    if _image_pulled:
        return True

    cfg = get_settings().sandbox
    image = cfg.image

    # Check if image exists locally
    proc = await asyncio.create_subprocess_exec(
        "docker", "image", "inspect", image,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode == 0:
        _image_pulled = True
        return True

    # Pull image
    logger.info("Pulling sandbox image: %s", image)
    proc = await asyncio.create_subprocess_exec(
        "docker", "pull", image,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    if proc.returncode == 0:
        _image_pulled = True
        logger.info("Sandbox image pulled: %s", image)
        return True

    logger.warning("Failed to pull sandbox image: %s", stderr.decode()[:200])
    return False


def get_workspace(session_id: str = "default") -> Path:
    """Get or create a persistent workspace directory for a session."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    ws = _WORKSPACE_BASE / safe
    ws.mkdir(parents=True, exist_ok=True)
    return ws


async def run_in_sandbox(
    command: str,
    *,
    timeout: int = 30,
    mode: str = "shell",  # "shell" or "python"
    session_id: str = "",
    mount_files: list[str] | None = None,
) -> tuple[int, str, str]:
    """Execute a command inside a Docker sandbox.

    Args:
        command: Shell command or Python code to execute.
        timeout: Execution timeout in seconds.
        mode: "shell" for shell commands, "python" for Python code.
        session_id: Session ID for persistent workspace.
        mount_files: Additional host files/dirs to mount read-only.

    Returns:
        (return_code, stdout, stderr)
    """
    cfg = get_settings().sandbox

    docker_args = [
        "docker", "run", "--rm",
        "--memory", cfg.memory_limit,
        f"--cpus={cfg.cpu_limit}",
        "--pids-limit", "100",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--tmpfs", f"{cfg.workdir}:rw,noexec,nosuid,size=128m",
        "-w", cfg.workdir,
    ]

    # Mount persistent workspace if session_id provided
    if session_id:
        ws = get_workspace(session_id)
        docker_args.extend(["-v", f"{ws}:{cfg.workdir}/persist:rw"])
        # Mount pre-installed packages
        pip_dir = ws / "_pip_packages"
        if pip_dir.exists():
            docker_args.extend([
                "-v", f"{pip_dir}:/usr/local/lib/python3.12/site-packages/extra:ro",
                "-e", "PYTHONPATH=/usr/local/lib/python3.12/site-packages/extra",
            ])

    # Mount additional files read-only
    if mount_files:
        for f in mount_files[:5]:  # Limit to 5 mounts
            p = Path(f)
            if p.exists():
                docker_args.extend(["-v", f"{p}:{cfg.workdir}/{p.name}:ro"])

    if not cfg.network:
        docker_args.append("--network=none")

    docker_args.append(cfg.image)

    if mode == "python":
        docker_args.extend(["python3", "-c", command])
    else:
        docker_args.extend(["sh", "-c", command])

    try:
        proc = await asyncio.create_subprocess_exec(
            *docker_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout + 5,  # +5s buffer for container startup
        )
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )
    except asyncio.TimeoutError:
        # Kill the container if it's still running
        try:
            proc.kill()
        except Exception:
            pass
        return (-1, "", f"Sandbox execution timed out ({timeout}s)")
    except Exception as e:
        return (-1, "", f"Sandbox error: {e}")


async def execute_sandboxed(
    command: str,
    *,
    timeout: int = 30,
    mode: str = "shell",
) -> tuple[bool, int, str, str]:
    """Try Docker sandbox first, return (used_sandbox, returncode, stdout, stderr).

    Returns used_sandbox=False if Docker is unavailable so caller can fall back.
    """
    cfg = get_settings().sandbox
    if not cfg.enabled:
        return (False, 0, "", "")

    if not await is_docker_available():
        return (False, 0, "", "")

    if not await _ensure_image():
        return (False, 0, "", "")

    rc, stdout, stderr = await run_in_sandbox(command, timeout=timeout, mode=mode)
    return (True, rc, stdout, stderr)
