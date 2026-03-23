"""CLI entry point for mul-agent.

Usage:
    mulagent                         # Launch TUI (Textual)
    mulagent --headless              # Plain REPL (no TUI)
    mulagent -c "帮我分析这段代码"     # Single-shot execution
    mulagent --model deepseek        # Override default model
    mulagent --config path.yaml      # Custom config file
    mulagent --session <id>          # Resume a specific session
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _ensure_src_path() -> None:
    """Add src/ to sys.path so internal imports resolve correctly."""
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _run_single(args) -> None:
    """Execute a single command and exit."""
    _ensure_src_path()
    from cli.runner import AgentRunner

    runner = AgentRunner(
        config_path=Path(args.config) if args.config else None,
        model_override=args.model,
    )
    session_id = args.session or runner.session_manager.get_or_create("cli_user", "cli")
    runner.session_manager.ensure_conversation(session_id, "cli_user")

    async def _exec():
        result = await runner.run(args.command, session_id)
        print(result.get("final_output", "(no output)"))

    asyncio.run(_exec())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mulagent",
        description="mul-agent CLI — interactive multi-agent assistant",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless REPL mode (no TUI, plain stdin/stdout)",
    )
    parser.add_argument(
        "-c", "--command",
        type=str,
        default=None,
        help="Execute a single command and exit",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override default LLM model ID",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Resume an existing session by ID",
    )
    args = parser.parse_args()

    _ensure_src_path()

    # Single-shot mode
    if args.command:
        _run_single(args)
        return

    # Headless REPL
    if args.headless:
        from cli.headless import run_headless
        run_headless(args)
        return

    # TUI mode (default)
    try:
        import textual  # noqa: F401
    except ImportError:
        print(
            "TUI mode requires the 'textual' package.\n"
            "Install it with:  pip install mulagent[cli]\n"
            "Or use headless mode:  mulagent --headless"
        )
        sys.exit(1)

    from cli.tui import run_tui
    run_tui(args)


if __name__ == "__main__":
    main()
