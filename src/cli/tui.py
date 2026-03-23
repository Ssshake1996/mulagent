"""Textual-based TUI for mul-agent.

Rich terminal interface with:
- Chat panel with Markdown rendering
- Real-time tool progress display
- Session sidebar
- Model selector
- Keyboard shortcuts
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)


# ── TUI App ──────────────────────────────────────────────────────

class MulAgentApp(App):
    """mul-agent Terminal UI."""

    TITLE = "mul-agent"
    CSS = """
    #main {
        height: 1fr;
    }
    #sidebar {
        width: 28;
        border-right: solid $primary-background;
        padding: 0 1;
    }
    #sidebar-title {
        text-style: bold;
        padding: 1 0;
        color: $text;
    }
    #session-list {
        height: 1fr;
    }
    #model-label {
        padding: 1 0 0 0;
        color: $text-muted;
    }
    #chat-area {
        width: 1fr;
    }
    #chat-log {
        height: 1fr;
        padding: 0 1;
    }
    #progress-bar {
        height: auto;
        max-height: 8;
        padding: 0 1;
        color: $warning;
    }
    #input-bar {
        dock: bottom;
        padding: 0 1;
    }
    #status {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(self, runner: Any, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner
        self.session_id = session_id
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Sessions", id="sidebar-title")
                yield ListView(id="session-list")
                yield Label(f"Model: {self.runner.current_model}", id="model-label")
            with Vertical(id="chat-area"):
                yield RichLog(
                    id="chat-log",
                    wrap=True,
                    highlight=True,
                    markup=True,
                    max_lines=2000,
                )
                yield Static("", id="progress-bar")
        yield Input(
            placeholder="Type a message... (Ctrl+N: new session, Ctrl+Q: quit)",
            id="input-bar",
        )
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_sessions()
        self._update_status()
        self.query_one("#input-bar", Input).focus()

    # ── Input handling ────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        # Commands
        cmd = text.lower()
        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if cmd == "/new":
            self.action_new_session()
            return
        if cmd in ("/help", "/h"):
            self._write_system(
                "/new — 新会话 | /model <id> — 切换模型 | "
                "/sessions — 会话列表 | /resume <id> — 恢复"
            )
            return
        if cmd == "/sessions":
            self._show_sessions()
            return
        if cmd.startswith("/resume"):
            self._handle_resume(text)
            return
        if cmd.startswith("/model"):
            self._handle_model(text)
            return
        if cmd.startswith("/"):
            self._write_system(f"Unknown command: {cmd}")
            return

        if self._busy:
            self._write_system("Please wait for the current task to finish.")
            return

        # Execute task
        self._write_chat("You", text, style="bold cyan")
        self._busy = True
        self._run_task(text)

    @work(exclusive=True)
    async def _run_task(self, text: str) -> None:
        """Run agent task in a Textual worker."""
        progress = self.query_one("#progress-bar", Static)
        t0 = time.monotonic()
        actions: list[str] = []

        async def on_progress(round_num: int, action: str, detail: str) -> None:
            if action == "tool_call" and detail:
                actions.append(f"[{round_num}] {detail}")
                display = "\n".join(actions[-6:])
                progress.update(f"[yellow]{display}[/]")
            elif action == "thinking":
                progress.update(f"[yellow]Round {round_num} thinking...[/]")

        try:
            progress.update("[yellow]Thinking...[/]")
            result = await self.runner.run(
                text, self.session_id, on_progress=on_progress
            )
            output = result.get("final_output", "(no output)")
            elapsed = time.monotonic() - t0
            status = result.get("status", "?")
            self._write_chat(
                "Assistant", output, style="bold green"
            )
            self._write_system(f"{status} · {elapsed:.1f}s")
        except asyncio.CancelledError:
            self._write_system("Task cancelled.")
        except Exception as e:
            self._write_chat("Error", str(e), style="bold red")
        finally:
            progress.update("")
            self._busy = False
            self._refresh_sessions()
            self._update_status()

    # ── Actions ───────────────────────────────────────────────

    def action_new_session(self) -> None:
        self.session_id = self.runner.session_manager.new_session("cli_user", "cli")
        chat = self.query_one("#chat-log", RichLog)
        chat.clear()
        self._write_system(f"New session: {self.session_id[-16:]}")
        self._refresh_sessions()
        self._update_status()

    def action_focus_input(self) -> None:
        self.query_one("#input-bar", Input).focus()

    # ── Helpers ───────────────────────────────────────────────

    def _write_chat(self, role: str, content: str, style: str = "") -> None:
        chat = self.query_one("#chat-log", RichLog)
        if style:
            chat.write(f"[{style}]{role}:[/] {content}")
        else:
            chat.write(f"{role}: {content}")

    def _write_system(self, msg: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"[dim]{msg}[/]")

    def _update_status(self) -> None:
        try:
            status = self.query_one("#status", Static)
            sid_short = self.session_id[-16:] if self.session_id else "none"
            status.update(
                f"  session: {sid_short}  |  model: {self.runner.current_model}"
            )
        except NoMatches:
            pass

    def _refresh_sessions(self) -> None:
        try:
            lv = self.query_one("#session-list", ListView)
            lv.clear()
            sessions = self.runner.session_manager.list_sessions("cli_user", limit=15)
            for s in sessions:
                sid = s["session_id"][-12:]
                preview = s["preview"][:20] if s.get("preview") else ""
                label = f"{sid} {preview}"
                lv.append(ListItem(Label(label)))
        except NoMatches:
            pass

    def _show_sessions(self) -> None:
        sessions = self.runner.session_manager.list_sessions("cli_user", limit=10)
        if not sessions:
            self._write_system("(no sessions)")
            return
        for s in sessions:
            sid = s["session_id"][-16:]
            self._write_system(
                f"  {sid}  {s['turns']}轮  {s.get('preview', '')[:40]}"
            )

    def _handle_resume(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self._write_system("usage: /resume <session_id_fragment>")
            return
        target = parts[1].strip()
        sessions = self.runner.session_manager.list_sessions("cli_user", limit=50)
        matched = next((s for s in sessions if target in s["session_id"]), None)
        if matched:
            self.session_id = matched["session_id"]
            self.runner.session_manager.resume_session(
                "cli_user", "cli", self.session_id
            )
            chat = self.query_one("#chat-log", RichLog)
            chat.clear()
            self._write_system(
                f"Resumed: {self.session_id[-16:]}  {matched['turns']}轮"
            )
            self._refresh_sessions()
            self._update_status()
        else:
            self._write_system(f"No session matching '{target}'")

    def _handle_model(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            models = self.runner.llm_manager.list_models()
            cur = self.runner.current_model
            for m in models:
                marker = " *" if m["id"] == cur else ""
                self._write_system(f"  {m['id']}: {m['model']}{marker}")
            return
        model_id = parts[1].strip()
        if self.runner.switch_model(model_id):
            self._write_system(f"Model → {model_id}")
            self._update_status()
            try:
                self.query_one("#model-label", Label).update(
                    f"Model: {model_id}"
                )
            except NoMatches:
                pass
        else:
            self._write_system(f"Unknown model: {model_id}")


# ── Entry point ──────────────────────────────────────────────────

def run_tui(args) -> None:
    """Launch the Textual TUI. Called from cli.main."""
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

    app = MulAgentApp(runner, session_id)
    app.run()
