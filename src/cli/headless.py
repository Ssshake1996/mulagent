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
    print("  /modify        |  /quit        退出")
    print("  /modify help   -- context CRUD (list/view/del/edit/clear)")
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

        if cmd.startswith("/modify"):
            _handle_modify(runner, session_id, line)
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


def _handle_modify(runner, session_id: str, text: str) -> None:
    """Handle /modify subcommands for context CRUD."""
    parts = text.split(maxsplit=2)
    sub = parts[1].lower() if len(parts) > 1 else "list"
    conv_store = runner.session_manager.conv_store

    if sub == "list":
        turns = conv_store.list_turns(session_id)
        if not turns:
            print("  (no turns in context)")
            return
        print(f"  Context: {len(turns)} turns")
        for i, t in enumerate(turns):
            role = "U" if t["role"] == "user" else "A"
            preview = t["content"][:60].replace("\n", " ")
            ts = t.get("ts", "")[:16]
            print(f"  [{i}] {role}: {preview}...  ({ts})")
        summary = conv_store.get_summary(session_id)
        if summary:
            print(f"\n  Summary: {summary[:100]}...")
        print()

    elif sub == "view":
        if len(parts) < 3:
            print("  usage: /modify view <index>")
            return
        try:
            idx = int(parts[2])
        except ValueError:
            print("  index must be a number")
            return
        turns = conv_store.list_turns(session_id)
        if idx < 0 or idx >= len(turns):
            print(f"  index out of range (0-{len(turns)-1})")
            return
        t = turns[idx]
        role = "User" if t["role"] == "user" else "Assistant"
        print(f"  [{idx}] {role} ({t.get('ts', '')[:19]}):")
        print(f"  {t['content']}")
        print()

    elif sub in ("del", "delete", "rm"):
        if len(parts) < 3:
            print("  usage: /modify del <index|start-end>")
            return
        arg = parts[2].strip()
        if "-" in arg and not arg.startswith("-"):
            try:
                start, end = arg.split("-", 1)
                start_i, end_i = int(start), int(end)
                count = conv_store.delete_turns_range(session_id, start_i, end_i + 1)
                print(f"  Deleted {count} turns [{start_i}-{end_i}]")
            except ValueError:
                print("  invalid range, use: /modify del 2-5")
        else:
            try:
                idx = int(arg)
            except ValueError:
                print("  index must be a number")
                return
            if conv_store.delete_turn(session_id, idx):
                print(f"  Deleted turn [{idx}]")
            else:
                print(f"  Failed to delete turn [{idx}]")
        print()

    elif sub == "edit":
        if len(parts) < 3:
            print("  usage: /modify edit <index>")
            return
        try:
            idx = int(parts[2])
        except ValueError:
            print("  index must be a number")
            return
        turns = conv_store.list_turns(session_id)
        if idx < 0 or idx >= len(turns):
            print(f"  index out of range (0-{len(turns)-1})")
            return
        t = turns[idx]
        role = "User" if t["role"] == "user" else "Assistant"
        # Open in $EDITOR if available, otherwise multi-line input
        import os
        import subprocess
        import tempfile
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", prefix=f"turn{idx}_", delete=False
            ) as f:
                f.write(t["content"])
                tmp_path = f.name
            print(f"  Opening [{idx}] {role} in {editor}...")
            subprocess.run([editor, tmp_path], check=True)
            new_content = open(tmp_path).read()
            os.unlink(tmp_path)
            if new_content != t["content"]:
                if conv_store.edit_turn(session_id, idx, new_content):
                    print(f"  Updated turn [{idx}] ({len(new_content)} chars)")
                else:
                    print(f"  Failed to save turn [{idx}]")
            else:
                print("  No changes")
        except FileNotFoundError:
            os.unlink(tmp_path) if os.path.exists(tmp_path) else None
            print(f"  Editor '{editor}' not found. Set $EDITOR env var.")
        except subprocess.CalledProcessError:
            os.unlink(tmp_path) if os.path.exists(tmp_path) else None
            print("  Editor exited with error, changes discarded")
        print()

    elif sub == "clear":
        if conv_store.clear_turns(session_id):
            print("  Context cleared (all turns removed)")
        else:
            print("  Failed to clear context")
        print()

    elif sub == "summary":
        summary = conv_store.get_summary(session_id)
        if summary:
            print(f"  Summary: {summary}")
        else:
            print("  (no summary yet, auto-generated after 20+ turns)")
        print()

    elif sub == "compress":
        llm = runner._react_params.get("llm")
        try:
            import asyncio as _aio
            _aio.get_event_loop().run_until_complete(
                conv_store.maybe_summarize(session_id, llm=llm)
            )
            print("  Context compressed")
        except Exception as e:
            print(f"  Compress failed: {e}")
        print()

    else:
        print("  /modify subcommands:")
        print("    list           -- show all turns with indices")
        print("    view <n>       -- view full content of turn n")
        print("    del <n>        -- delete turn n")
        print("    del <n-m>      -- delete turns n through m")
        print("    edit <n>       -- open $EDITOR to edit turn n")
        print("    clear          -- remove all turns")
        print("    summary        -- show conversation summary")
        print("    compress       -- force summarize old turns")
        print()


def run_headless(args) -> None:
    """Entry point called from cli.main."""
    from cli import ensure_src_path
    ensure_src_path()

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
