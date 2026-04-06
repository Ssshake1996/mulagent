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
import os
import sys
from pathlib import Path

# Ensure truecolor support before any Textual import caches terminal capabilities.
# Many terminals (SSH, tmux, VS Code) report TERM=xterm (8-color) but actually
# support 256/truecolor. Without this, colors are mapped to nearest 8-color value.
if os.environ.get("TERM") in ("xterm", "screen", "vt100", ""):
    os.environ["TERM"] = "xterm-256color"
if not os.environ.get("COLORTERM"):
    os.environ["COLORTERM"] = "truecolor"

from cli import ensure_src_path


def _run_evolve(args) -> None:
    """Run self-evolution and exit."""
    ensure_src_path()
    from cli.runner import AgentRunner

    runner = AgentRunner(
        config_path=Path(args.config) if args.config else None,
        model_override=args.model,
    )
    llm = runner._react_params.get("llm")
    mode = args.evolve

    async def _exec():
        from evolution.controller import EvolutionController
        ctrl = EvolutionController()

        if mode == "diagnose":
            print("Diagnosing system...\n")
            summary = await ctrl.diagnose_only()
            print(summary)
        elif mode == "history":
            logs = ctrl.list_evolution_logs()
            if not logs:
                print("(no evolution history)")
            else:
                for log in logs:
                    print(f"  {log['timestamp']}  {log['mode']}  "
                          f"proposed={log['proposed']} applied={log['applied']}")
        else:
            print(f"Running evolution ({mode} mode)...\n")
            report = await ctrl.evolve(mode=mode, llm=llm)
            print(report.summary())

    asyncio.run(_exec())


def _run_absorb(args) -> None:
    """Absorb an external project and exit."""
    ensure_src_path()
    from cli.runner import AgentRunner

    runner = AgentRunner(
        config_path=Path(args.config) if args.config else None,
        model_override=args.model,
    )
    llm = runner._react_params.get("llm")
    git_url = args.absorb

    async def _exec():
        from evolution.controller import EvolutionController
        ctrl = EvolutionController()
        print(f"Cloning and analyzing: {git_url}\n")
        report = await ctrl.absorb_project(git_url, auto_apply=False, llm=llm)
        print(report.summary())
        if report.proposed:
            print("\nTo apply: mulagent --evolve full")

    asyncio.run(_exec())


def _run_single(args) -> None:
    """Execute a single command and exit."""
    ensure_src_path()
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
    parser.add_argument(
        "--evolve",
        nargs="?",
        const="propose",
        default=None,
        metavar="MODE",
        help="Self-evolution: propose (default), diagnose, auto, full",
    )
    parser.add_argument(
        "--absorb",
        type=str,
        default=None,
        metavar="GIT_URL",
        help="Absorb external project capabilities from a Git URL",
    )
    # Handle "mulagent init" before argparse
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        ensure_src_path()
        from cli.init_wizard import run_init
        run_init()
        return

    args = parser.parse_args()

    ensure_src_path()

    # Single-shot mode
    if args.command:
        _run_single(args)
        return

    # Self-evolution mode
    if args.evolve is not None:
        _run_evolve(args)
        return

    # Absorb external project
    if args.absorb:
        _run_absorb(args)
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
