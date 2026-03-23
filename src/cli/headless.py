"""Headless REPL — plain stdin/stdout interface, no TUI dependency.

Usage:
    mulagent --headless
    echo "帮我搜索天气" | mulagent --headless   (pipe mode)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path


def _print_banner(session_id: str, model: str) -> None:
    print("=" * 60)
    print("  mul-agent  ·  headless REPL")
    print(f"  model: {model}  ·  session: {session_id[-16:]}")
    print("=" * 60)
    print("  /new   新会话  |  /model <id>  切换模型")
    print("  /sessions      |  /resume <id> 恢复会话")
    print("  /help          |  /quit        退出")
    print("-" * 60)
    print()


async def _on_progress(round_num: int, action: str, detail: str) -> None:
    """Print progress inline."""
    if action == "tool_call" and detail:
        print(f"  [{round_num}] tool: {detail}")
    elif action == "thinking":
        print(f"  [{round_num}] thinking...")


async def _repl(runner, session_id: str) -> None:
    """Core async REPL loop."""
    from cli.runner import AgentRunner

    runner: AgentRunner  # type hint for IDE

    _print_banner(session_id, runner.current_model)

    is_tty = sys.stdin.isatty()
    loop = asyncio.get_event_loop()

    while True:
        # Read input
        try:
            if is_tty:
                line = await loop.run_in_executor(None, input, ">>> ")
            else:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:  # EOF
                    break
                line = line.rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        line = line.strip()
        if not line:
            continue

        # ── Commands ──────────────────────────────────────────
        cmd = line.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("Bye.")
            break

        if cmd in ("/help", "/h"):
            _print_banner(session_id, runner.current_model)
            continue

        if cmd == "/new":
            session_id = runner.session_manager.new_session("cli_user", "cli")
            print(f"[new session: {session_id}]\n")
            continue

        if cmd == "/sessions":
            sessions = runner.session_manager.list_sessions("cli_user", limit=10)
            if not sessions:
                print("  (no sessions found)")
            for s in sessions:
                sid = s["session_id"][-16:]
                print(f"  {sid}  {s['turns']}轮  {s['preview'][:40]}...")
            print()
            continue

        if cmd.startswith("/resume"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print("  usage: /resume <session_id_fragment>")
                continue
            target = parts[1].strip()
            sessions = runner.session_manager.list_sessions("cli_user", limit=50)
            matched = next((s for s in sessions if target in s["session_id"]), None)
            if matched:
                session_id = matched["session_id"]
                runner.session_manager.resume_session("cli_user", "cli", session_id)
                print(f"[resumed: {session_id[-16:]}  {matched['turns']}轮]\n")
            else:
                print(f"  session matching '{target}' not found")
            continue

        if cmd.startswith("/model"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                models = runner.llm_manager.list_models()
                cur = runner.current_model
                for m in models:
                    marker = " *" if m["id"] == cur else ""
                    print(f"  {m['id']}: {m['model']}{marker}")
                print()
                continue
            model_id = parts[1].strip()
            if runner.switch_model(model_id):
                print(f"[model → {model_id}]\n")
            else:
                print(f"  unknown model: {model_id}")
            continue

        if cmd.startswith("/"):
            print(f"  unknown command: {cmd}")
            continue

        # ── Execute task ──────────────────────────────────────
        t0 = time.monotonic()
        try:
            result = await runner.run(line, session_id, on_progress=_on_progress)
            output = result.get("final_output", "(no output)")
            elapsed = time.monotonic() - t0
            print(f"\n{output}")
            print(f"\n  [{result.get('status', '?')} · {elapsed:.1f}s]\n")
        except KeyboardInterrupt:
            print("\n  [cancelled]\n")
        except Exception as e:
            print(f"\n  [error: {e}]\n")


def run_headless(args) -> None:
    """Entry point called from cli.main."""
    # Add src/ to path so internal imports work
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from cli.runner import AgentRunner

    runner = AgentRunner(
        config_path=Path(args.config) if args.config else None,
        model_override=args.model,
    )

    user_id = "cli_user"
    session_id = (
        args.session
        if args.session
        else runner.session_manager.get_or_create(user_id, "cli")
    )
    runner.session_manager.ensure_conversation(session_id, user_id)

    asyncio.run(_repl(runner, session_id))
