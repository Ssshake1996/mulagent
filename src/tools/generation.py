"""Generation tools: create side-effects in the world.

- execute_shell: Run shell commands (with safety checks)
- code_run: Run code in a sandboxed subprocess
- write_file: Write content to a file
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any


def _get_tool_timeout() -> tuple[int, int]:
    """Derive (default_timeout, max_timeout) from react.timeout.

    Returns:
        default: used when LLM doesn't pass timeout param
        max: hard cap for any tool call
    """
    try:
        from common.config import get_settings
        overall = get_settings().react.timeout
        return max(overall // 20, 60), max(overall // 10, 120)
    except Exception:
        return 60, 120

from tools.base import ToolDef

logger = logging.getLogger(__name__)

# ── Shell safety ──────────────────────────────────────────────────

_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\b:()\s*\{",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0",
    r"\bkill\s+-9\s+1\b",
    r">\s*/dev/sd",
    r"\bchmod\s+-R\s+777\s+/",
    r"\bchown\s+-R\s+.*\s+/\s*$",
]


def _strip_quoted_strings(cmd: str) -> str:
    """Remove content inside quotes to avoid false positives.

    E.g., `curl ... | python3 -c "import json; d = {'k': 'v'}"` should not
    match the fork-bomb pattern `:(){ ... }` against the Python dict literal.
    """
    # Remove double-quoted strings (handling escaped quotes)
    cmd = re.sub(r'"(?:[^"\\]|\\.)*"', '""', cmd)
    # Remove single-quoted strings (no escaping in single quotes per POSIX)
    cmd = re.sub(r"'[^']*'", "''", cmd)
    return cmd


def is_dangerous_command(cmd: str) -> bool:
    """Check if a shell command matches known dangerous patterns.

    Quoted string contents are stripped first to avoid false positives
    on things like `python3 -c "code with dict literals"`.
    """
    stripped = _strip_quoted_strings(cmd)
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, stripped):
            return True
    return False


async def _execute_shell(params: dict[str, Any], **deps: Any) -> str:
    """Execute a shell command with safety checks."""
    command = params.get("command", "")
    if not command:
        return "Error: command is required"

    if is_dangerous_command(command):
        logger.warning("Blocked dangerous command: %s", command)
        return f"BLOCKED: dangerous command detected — `{command}`"

    # Derive default and cap from config react.timeout
    _default_to, _max_to = _get_tool_timeout()
    timeout = min(params.get("timeout", _default_to), _max_to)

    # Try Docker sandbox first
    from tools.sandbox import execute_sandboxed
    used_sandbox, rc, out, err = await execute_sandboxed(
        command, timeout=timeout, mode="shell",
    )
    if used_sandbox:
        return _format_exec_result(rc, out.strip(), err.strip(), sandbox=True)

    # Fallback: direct execution
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        return _format_exec_result(proc.returncode, out, err)
    except asyncio.TimeoutError:
        return f"Command timed out ({timeout}s): `{command}`"
    except Exception as e:
        return f"Execution error: {e}"


def _format_exec_result(
    returncode: int | None, out: str, err: str, *, sandbox: bool = False,
) -> str:
    """Format shell/code execution output with token-based truncation."""
    from common.tokenizer import truncate_middle, truncate_to_tokens

    prefix = "[sandbox] " if sandbox else ""
    result_parts = [f"{prefix}exit_code: {returncode}"]
    if out:
        out = truncate_middle(out, 1200)  # ~1200 tokens for stdout
        result_parts.append(f"stdout:\n{out}")
    if err:
        err = truncate_to_tokens(err, 400)  # ~400 tokens for stderr
        result_parts.append(f"stderr:\n{err}")
    return "\n".join(result_parts)


async def _code_run(params: dict[str, Any], **deps: Any) -> str:
    """Run code in an isolated subprocess. Supports multiple languages.

    Uses temp files instead of -c flag to avoid shell length limits.
    Tries Docker sandbox first for isolation.
    """
    import shutil
    import tempfile

    code = params.get("code", "")
    if not code:
        return "Error: code is required"

    language = params.get("language", "python").lower()
    _default_to, _max_to = _get_tool_timeout()
    timeout = min(params.get("timeout", _default_to), _max_to)

    _tmpdir = tempfile.gettempdir()
    _rust_out = str(Path(_tmpdir) / "_mul_agent_out")

    # Language → (file extension, command builder)
    lang_configs: dict[str, tuple[str, list[str]]] = {
        "python": (".py", ["python3"] if shutil.which("python3") else ["python"]),
        "javascript": (".js", ["node"]),
        "typescript": (".ts", ["npx", "tsx"]),
        "go": (".go", ["go", "run"]),
        "rust": (".rs", ["rustc", "-o", _rust_out]),
        "java": (".java", ["java"]),  # Java 11+ can run single files
    }

    config = lang_configs.get(language)
    if config is None:
        return f"Error: unsupported language '{language}'. Supported: {', '.join(lang_configs.keys())}"

    ext, base_cmd = config

    # Check if runtime is available (for non-python)
    if language != "python" and not shutil.which(base_cmd[0]):
        return f"Error: {base_cmd[0]} not found. Cannot run {language} code."

    # Try Docker sandbox first (Python only for now)
    if language == "python":
        from tools.sandbox import execute_sandboxed
        used_sandbox, rc, out, err = await execute_sandboxed(
            code, timeout=timeout, mode="python",
        )
        if used_sandbox:
            return _format_exec_result(rc, out.strip(), err.strip(), sandbox=True)

    # Write code to temp file (avoids -c flag length limits)
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, prefix="_mul_agent_",
            dir=_tmpdir, delete=False,
        ) as f:
            f.write(code)
            tmp_file = f.name

        # Build command
        if language == "rust":
            # Rust: compile then run
            compile_proc = await asyncio.create_subprocess_exec(
                "rustc", tmp_file, "-o", _rust_out,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            c_stdout, c_stderr = await asyncio.wait_for(
                compile_proc.communicate(), timeout=timeout,
            )
            if compile_proc.returncode != 0:
                return _format_exec_result(
                    compile_proc.returncode,
                    c_stdout.decode(errors="replace").strip(),
                    c_stderr.decode(errors="replace").strip(),
                )
            cmd = [_rust_out]
        else:
            cmd = base_cmd + [tmp_file]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        return _format_exec_result(proc.returncode, out, err)
    except asyncio.TimeoutError:
        return f"Code execution timed out ({timeout}s)."
    except Exception as e:
        return f"Code execution error: {e}"
    finally:
        # Clean up temp files
        if tmp_file:
            Path(tmp_file).unlink(missing_ok=True)
        Path(_rust_out).unlink(missing_ok=True)


async def _write_file(params: dict[str, Any], **deps: Any) -> str:
    """Write content to a file."""
    file_path = params.get("path", "")
    content = params.get("content", "")
    if not file_path:
        return "Error: path is required"

    path = Path(file_path).expanduser()

    # Safety: only allow writing under home or /tmp
    from tools.injection import _is_path_allowed
    if not _is_path_allowed(path):
        return f"Error: write access denied for {path}"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Write error: {e}"


EXECUTE_SHELL = ToolDef(
    name="execute_shell",
    description=(
        "Execute a shell command. Use for system operations like checking status, "
        "installing packages, running scripts. Dangerous commands (rm -rf /, sudo, etc.) "
        "are blocked."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 60, max 120)",
            },
        },
        "required": ["command"],
    },
    fn=_execute_shell,
    category="execution",
)

CODE_RUN = ToolDef(
    name="code_run",
    description=(
        "Run a code snippet and return the output. Supports Python (default), "
        "JavaScript, TypeScript, Go, Rust, Java. For calculations, data processing, testing code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Code to execute",
            },
            "language": {
                "type": "string",
                "description": "Programming language (default: python)",
                "enum": ["python", "javascript", "typescript", "go", "rust", "java"],
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 60, max 120)",
            },
        },
        "required": ["code"],
    },
    fn=_code_run,
    category="execution",
)

async def _edit_file(params: dict[str, Any], **deps: Any) -> str:
    """Apply a surgical find-and-replace edit to a file.

    Safer than write_file for small changes — preserves the rest of the file
    and requires specifying the exact text to replace.
    """
    file_path = params.get("path", "")
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")

    if not file_path:
        return "Error: path is required"
    if not old_text:
        return "Error: old_text is required (the text to find and replace)"

    path = Path(file_path).expanduser()

    from tools.injection import _is_path_allowed
    if not _is_path_allowed(path):
        return f"Error: access denied for {path}"
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"

    replace_all = params.get("replace_all", False)

    try:
        content = path.read_text(errors="replace")

        # ── mtime conflict detection ──
        current_mtime = path.stat().st_mtime
        expected_mtime = params.get("_expected_mtime")
        if expected_mtime is not None and abs(current_mtime - expected_mtime) > 0.5:
            return (
                f"Error: file {path} was modified externally since last read. "
                "Please read_file again before editing."
            )

        count = content.count(old_text)

        if count == 0:
            # Show context to help debug
            lines = content.split("\n")
            preview = "\n".join(lines[:20])
            return (
                f"Error: old_text not found in {path}. "
                f"File has {len(lines)} lines. First 20 lines:\n{preview}"
            )

        if count > 1 and not replace_all:
            # Show locations to help make match unique
            positions = []
            start = 0
            for i in range(min(count, 5)):
                idx = content.index(old_text, start)
                line_num = content[:idx].count("\n") + 1
                positions.append(f"line {line_num}")
                start = idx + 1
            return (
                f"Error: old_text found {count} times in {path} "
                f"(at {', '.join(positions)}{'...' if count > 5 else ''}). "
                "Provide more surrounding context to make the match unique, "
                "or set replace_all=true to replace all occurrences."
            )

        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
        path.write_text(new_content)

        # ── Generate unified diff for visibility ──
        import difflib
        old_lines_list = old_text.splitlines(keepends=True)
        new_lines_list = new_text.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines_list, new_lines_list,
            fromfile=f"a/{path.name}", tofile=f"b/{path.name}", lineterm="",
        ))
        diff_text = "\n".join(diff[:30])  # cap at 30 lines
        if len(diff) > 30:
            diff_text += f"\n... ({len(diff) - 30} more diff lines)"

        replaced = count if replace_all else 1
        old_line_count = old_text.count("\n") + 1
        new_line_count = new_text.count("\n") + 1
        summary = (
            f"Edited {path}: replaced {replaced} occurrence{'s' if replaced > 1 else ''} "
            f"({old_line_count} lines → {new_line_count} lines)"
        )
        if diff_text:
            return f"{summary}\n\n```diff\n{diff_text}\n```"
        return summary
    except Exception as e:
        return f"Edit error: {e}"


EDIT_FILE = ToolDef(
    name="edit_file",
    description=(
        "Apply a surgical find-and-replace edit to a file. Safer than write_file — "
        "only changes the specified text, preserving the rest. Use read_file first "
        "to see the current content, then specify the exact text to replace."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path to edit",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find (must match uniquely)",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace ALL occurrences (default: false, requires unique match)",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    fn=_edit_file,
    category="file",
)


WRITE_FILE = ToolDef(
    name="write_file",
    description="Write content to a local file. Creates parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path to write to",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    },
    fn=_write_file,
    category="file",
)
