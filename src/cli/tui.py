"""Textual-based TUI for mul-agent.

Rich terminal interface with:
- Chat panel with text selection and copy (Ctrl+C)
- Real-time tool progress display
- Session sidebar
- Model selector
- Context management (/modify)
- Command auto-completion (type / to see all commands)
- Keyboard shortcuts
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option


# ── Command definitions for auto-completion ────────────────────
COMMANDS: list[tuple[str, str]] = [
    ("/new", "Start a new session"),
    ("/help", "Show available commands"),
    ("/sessions", "List recent sessions"),
    ("/resume", "Resume a session by ID fragment"),
    ("/model", "Show or switch LLM model"),
    ("/modify list", "Show all context turns"),
    ("/modify view", "View a turn's full content"),
    ("/modify edit", "Edit a turn (Ctrl+S save, Esc cancel)"),
    ("/modify del", "Delete a turn or range"),
    ("/modify clear", "Remove all turns"),
    ("/modify summary", "Show conversation summary"),
    ("/modify compress", "Smart compress (archive old topics)"),
    ("/modify topics", "List all topics (hot + archived)"),
    ("/modify expand", "Recall an archived topic by ID"),
    ("/modify collapse", "Collapse a recalled topic"),
    ("/recall", "Recall archived topic by keyword"),
    ("/directives list", "Show persistent directives"),
    ("/directives add", "Add a persistent directive"),
    ("/directives del", "Remove a directive by index"),
    ("/directives clear", "Remove all directives"),
    ("/evolve", "Self-evolution: diagnose and propose improvements"),
    ("/evolve diagnose", "Show system diagnostic report"),
    ("/evolve auto", "Apply safe improvements automatically"),
    ("/evolve full", "Apply all proposed improvements"),
    ("/evolve history", "Show past evolution records"),
    ("/absorb", "Absorb external Git project capabilities"),
    ("/quit", "Exit the application"),
]


# ── Edit overlay screen ──────────────────────────────────────────

class EditScreen(ModalScreen[str | None]):
    """Modal editor for a conversation turn. Ctrl+S saves, Esc cancels."""

    CSS = """
    EditScreen {
        align: center middle;
    }
    #edit-container {
        width: 90%;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #edit-title {
        text-style: bold;
        padding: 0 0 1 0;
    }
    #edit-area {
        height: 1fr;
    }
    #edit-toolbar {
        height: 3;
        padding: 1 0 0 0;
    }
    .edit-btn {
        margin: 0 2 0 0;
    }
    #edit-hint {
        height: 1;
        color: $text;
        background: $primary;
        padding: 0 2;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, index: int, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self._index = index
        self._role = role
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.widgets import Button
        with Vertical(id="edit-container"):
            yield Label(
                f"Editing turn [{self._index}] ({self._role})",
                id="edit-title",
            )
            yield TextArea(
                self._content,
                id="edit-area",
                soft_wrap=True,
                show_line_numbers=True,
                language=None,
                theme="css",
            )
            yield Label(
                " Ctrl+S = Save and close  |  Esc = Cancel without saving ",
                id="edit-hint",
            )
            with Horizontal(id="edit-toolbar"):
                yield Button("💾 Save (Ctrl+S)", variant="success", id="btn-save", classes="edit-btn")
                yield Button("Cancel (Esc)", variant="default", id="btn-cancel", classes="edit-btn")

    def on_mount(self) -> None:
        self.query_one("#edit-area", TextArea).focus()

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_save(self) -> None:
        text = self.query_one("#edit-area", TextArea).text
        self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss(None)


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
    #input-wrapper {
        dock: bottom;
        height: auto;
    }
    #cmd-popup {
        display: none;
        height: auto;
        max-height: 14;
        padding: 0 1;
        background: $surface;
        border: solid $accent;
        margin: 0 1;
    }
    #cmd-popup.visible {
        display: block;
    }
    #input-bar {
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
        self._popup_visible = False
        self._filtered_cmds: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Sessions", id="sidebar-title")
                yield ListView(id="session-list")
                yield Label(f"Model: {self.runner.current_model}", id="model-label")
            with Vertical(id="chat-area"):
                yield TextArea(
                    "",
                    id="chat-log",
                    read_only=True,
                    show_line_numbers=False,
                    soft_wrap=True,
                    language=None,
                    theme="css",
                )
                yield Static("", id="progress-bar")
        with Vertical(id="input-wrapper"):
            yield OptionList(id="cmd-popup")
            yield Input(
                placeholder="Type / for commands, or enter a message... (Ctrl+N: new, Ctrl+Q: quit)",
                id="input-bar",
            )
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_sessions()
        self._update_status()
        self.query_one("#input-bar", Input).focus()

    # ── Command auto-completion ────────────────────────────────

    @on(Input.Changed, "#input-bar")
    def _on_input_changed(self, event: Input.Changed) -> None:
        """Show/hide command popup as user types."""
        text = event.value
        if text.startswith("/"):
            query = text.lower()
            self._filtered_cmds = [
                (cmd, desc) for cmd, desc in COMMANDS
                if cmd.startswith(query) or query in cmd
            ]
            if self._filtered_cmds:
                self._show_popup(self._filtered_cmds)
            else:
                self._hide_popup()
        else:
            self._hide_popup()

    def _show_popup(self, items: list[tuple[str, str]]) -> None:
        """Display the command completion popup."""
        popup = self.query_one("#cmd-popup", OptionList)
        popup.clear_options()
        for cmd, desc in items:
            popup.add_option(Option(f"{cmd}  — {desc}", id=cmd))
        popup.add_class("visible")
        self._popup_visible = True
        # Highlight first item
        if len(items) > 0:
            popup.highlighted = 0

    def _hide_popup(self) -> None:
        """Hide the command completion popup."""
        popup = self.query_one("#cmd-popup", OptionList)
        popup.remove_class("visible")
        self._popup_visible = False

    def _accept_completion(self) -> None:
        """Accept the currently highlighted completion."""
        popup = self.query_one("#cmd-popup", OptionList)
        idx = popup.highlighted
        if idx is not None and 0 <= idx < len(self._filtered_cmds):
            cmd, _desc = self._filtered_cmds[idx]
            inp = self.query_one("#input-bar", Input)
            # Set the input to the completed command
            # Add trailing space for commands that take arguments
            needs_arg = cmd in (
                "/resume", "/model",
                "/modify view", "/modify edit", "/modify del",
                "/directives add", "/directives del",
            )
            inp.value = cmd + (" " if needs_arg else "")
            inp.cursor_position = len(inp.value)
            self._hide_popup()
            inp.focus()

    @on(OptionList.OptionSelected, "#cmd-popup")
    def _on_popup_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle click/enter selection in the popup."""
        self._accept_completion()

    def on_key(self, event) -> None:
        """Intercept arrow keys and Tab/Enter for popup navigation."""
        if not self._popup_visible:
            return

        popup = self.query_one("#cmd-popup", OptionList)

        if event.key == "down":
            event.prevent_default()
            event.stop()
            h = popup.highlighted
            if h is None:
                popup.highlighted = 0
            elif h < len(self._filtered_cmds) - 1:
                popup.highlighted = h + 1
        elif event.key == "up":
            event.prevent_default()
            event.stop()
            h = popup.highlighted
            if h is not None and h > 0:
                popup.highlighted = h - 1
        elif event.key == "tab":
            event.prevent_default()
            event.stop()
            self._accept_completion()
        elif event.key == "escape":
            event.prevent_default()
            event.stop()
            self._hide_popup()
            self.query_one("#input-bar", Input).focus()

    # ── Input handling ────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        # If popup is visible and Enter is pressed, accept the completion instead
        if self._popup_visible and text.startswith("/"):
            popup = self.query_one("#cmd-popup", OptionList)
            idx = popup.highlighted
            if idx is not None and 0 <= idx < len(self._filtered_cmds):
                cmd, _desc = self._filtered_cmds[idx]
                # Only intercept if the input exactly matches a partial prefix
                # (i.e., user hasn't typed a full command + args yet)
                if text.lower() != cmd and cmd.startswith(text.lower()):
                    self._accept_completion()
                    return
        self._hide_popup()
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
                "/new -- new session  |  /model <id> -- switch model\n"
                "/sessions -- list    |  /resume <id> -- resume\n"
                "/modify -- context management (list/view/del/edit/clear/summary/compress)"
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
        if cmd.startswith("/modify"):
            self._handle_modify(text)
            return
        if cmd.startswith("/directives"):
            self._handle_directives(text)
            return
        if cmd.startswith("/evolve"):
            self._run_evolve(text)
            return
        if cmd.startswith("/absorb"):
            self._run_absorb(text)
            return
        if cmd.startswith("/"):
            self._write_system(f"Unknown command: {cmd}")
            return

        if self._busy:
            self._write_system("Please wait for the current task to finish.")
            return

        # Execute task
        self._write_chat("You", text)
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
            self._write_chat("Assistant", output)
            self._write_system(f"{status} | {elapsed:.1f}s")
        except asyncio.CancelledError:
            self._write_system("Task cancelled.")
        except Exception as e:
            self._write_chat("Error", str(e))
        finally:
            progress.update("")
            self._busy = False
            self._refresh_sessions()
            self._update_status()

    # ── Actions ───────────────────────────────────────────────

    def action_new_session(self) -> None:
        self.session_id = self.runner.session_manager.new_session("cli_user", "cli")
        self._clear_chat()
        self._write_system(f"New session: {self.session_id[-16:]}")
        self._refresh_sessions()
        self._update_status()

    def action_focus_input(self) -> None:
        self.query_one("#input-bar", Input).focus()

    # ── /modify handler ───────────────────────────────────────

    def _handle_modify(self, text: str) -> None:
        """Handle /modify subcommands for context CRUD."""
        parts = text.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else "list"
        conv_store = self.runner.session_manager.conv_store

        if sub == "list":
            turns = conv_store.list_turns(self.session_id)
            if not turns:
                self._write_system("(no turns in context)")
                return
            lines = [f"Context: {len(turns)} turns"]
            for i, t in enumerate(turns):
                role = "U" if t["role"] == "user" else "A"
                preview = t["content"][:60].replace("\n", " ")
                ts = t.get("ts", "")[:16]
                lines.append(f"  [{i}] {role}: {preview}...  ({ts})")
            summary = conv_store.get_summary(self.session_id)
            if summary:
                lines.append(f"\n  Summary: {summary[:100]}...")
            self._write_system("\n".join(lines))

        elif sub == "view":
            if len(parts) < 3:
                self._write_system("usage: /modify view <index>")
                return
            try:
                idx = int(parts[2])
            except ValueError:
                self._write_system("index must be a number")
                return
            turns = conv_store.list_turns(self.session_id)
            if idx < 0 or idx >= len(turns):
                self._write_system(f"index out of range (0-{len(turns)-1})")
                return
            t = turns[idx]
            role = "User" if t["role"] == "user" else "Assistant"
            self._write_system(f"[{idx}] {role} ({t.get('ts', '')[:19]}):\n{t['content']}")

        elif sub in ("del", "delete", "rm"):
            if len(parts) < 3:
                self._write_system("usage: /modify del <index|start-end>")
                return
            arg = parts[2].strip()
            if "-" in arg and not arg.startswith("-"):
                # Range delete: /modify del 2-5
                try:
                    start, end = arg.split("-", 1)
                    start_i, end_i = int(start), int(end)
                    count = conv_store.delete_turns_range(self.session_id, start_i, end_i + 1)
                    self._write_system(f"Deleted {count} turns [{start_i}-{end_i}]")
                except ValueError:
                    self._write_system("invalid range, use: /modify del 2-5")
            else:
                try:
                    idx = int(arg)
                except ValueError:
                    self._write_system("index must be a number")
                    return
                if conv_store.delete_turn(self.session_id, idx):
                    self._write_system(f"Deleted turn [{idx}]")
                else:
                    self._write_system(f"Failed to delete turn [{idx}]")

        elif sub == "edit":
            # /modify edit <n> — open interactive editor overlay
            if len(parts) < 3:
                self._write_system("usage: /modify edit <index>")
                return
            try:
                idx = int(parts[2])
            except ValueError:
                self._write_system("index must be a number")
                return
            turns = conv_store.list_turns(self.session_id)
            if idx < 0 or idx >= len(turns):
                self._write_system(f"index out of range (0-{len(turns)-1})")
                return
            t = turns[idx]
            role = "User" if t["role"] == "user" else "Assistant"

            def _on_edit_done(result: str | None) -> None:
                if result is None:
                    self._write_system("Edit cancelled")
                    return
                if conv_store.edit_turn(self.session_id, idx, result):
                    self._write_system(f"Updated turn [{idx}] ({len(result)} chars)")
                else:
                    self._write_system(f"Failed to save turn [{idx}]")

            self.push_screen(
                EditScreen(idx, role, t["content"]),
                callback=_on_edit_done,
            )

        elif sub == "clear":
            if conv_store.clear_turns(self.session_id):
                self._write_system("Context cleared (all turns removed)")
            else:
                self._write_system("Failed to clear context")

        elif sub == "summary":
            summary = conv_store.get_summary(self.session_id)
            if summary:
                self._write_system(f"Summary: {summary}")
            else:
                self._write_system("(no summary yet, auto-generated after 20+ turns)")

        elif sub == "compress":
            self._write_system("Compressing context...")
            self._run_compress()

        else:
            self._write_system(
                "/modify subcommands:\n"
                "  list           -- show all turns with indices\n"
                "  view <n>       -- view full content of turn n\n"
                "  del <n>        -- delete turn n\n"
                "  del <n-m>      -- delete turns n through m\n"
                "  edit <n>       -- open editor for turn n (Ctrl+S save, Esc cancel)\n"
                "  clear          -- remove all turns\n"
                "  summary        -- show conversation summary\n"
                "  compress       -- force summarize old turns"
            )

    @work(exclusive=True, thread=True)
    def _run_compress(self) -> None:
        """Run async summarization in a worker."""
        import asyncio as _aio
        conv_store = self.runner.session_manager.conv_store
        llm = self.runner._react_params.get("llm")
        try:
            loop = _aio.new_event_loop()
            loop.run_until_complete(
                conv_store.maybe_summarize(self.session_id, llm=llm)
            )
            loop.close()
            self.call_from_thread(self._write_system, "Context compressed")
        except Exception as e:
            self.call_from_thread(self._write_system, f"Compress failed: {e}")

    # ── /evolve and /absorb handlers ────────────────────────────

    @work(exclusive=True, thread=True)
    def _run_evolve(self, text: str) -> None:
        """Run evolution in a worker thread."""
        import asyncio as _aio
        parts = text.split(maxsplit=1)
        sub = parts[1].strip().lower() if len(parts) > 1 else "propose"

        from evolution.controller import EvolutionController
        ctrl = EvolutionController()
        llm = self.runner._react_params.get("llm")

        loop = _aio.new_event_loop()
        try:
            if sub == "diagnose":
                self.call_from_thread(self._write_system, "Running diagnosis...")
                summary = loop.run_until_complete(ctrl.diagnose_only())
                self.call_from_thread(self._write_system, summary)
            elif sub == "history":
                logs = ctrl.list_evolution_logs(limit=10)
                if not logs:
                    self.call_from_thread(self._write_system, "(no evolution history)")
                else:
                    lines = [f"Evolution History ({len(logs)} records):"]
                    for log in logs:
                        lines.append(
                            f"  {log['timestamp']}  {log['mode']}  "
                            f"proposed={log['proposed']} applied={log['applied']}"
                        )
                    self.call_from_thread(self._write_system, "\n".join(lines))
            elif sub in ("propose", "auto", "full"):
                self.call_from_thread(
                    self._write_system,
                    f"Running evolution ({sub} mode)..."
                )
                report = loop.run_until_complete(
                    ctrl.evolve(mode=sub, llm=llm)
                )
                self.call_from_thread(self._write_system, report.summary())
            else:
                self.call_from_thread(self._write_system,
                    "/evolve subcommands:\n"
                    "  propose   -- diagnose + proposals (default)\n"
                    "  diagnose  -- diagnostic report only\n"
                    "  auto      -- apply safe changes\n"
                    "  full      -- apply all changes\n"
                    "  history   -- past evolution records"
                )
        except Exception as e:
            self.call_from_thread(self._write_system, f"Evolution error: {e}")
        finally:
            loop.close()

    @work(exclusive=True, thread=True)
    def _run_absorb(self, text: str) -> None:
        """Run project absorption in a worker thread."""
        import asyncio as _aio
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self.call_from_thread(self._write_system,
                "usage: /absorb <git_url>\n"
                "example: /absorb https://github.com/user/project.git"
            )
            return

        git_url = parts[1].strip()
        self.call_from_thread(
            self._write_system,
            f"Cloning and analyzing: {git_url}..."
        )

        from evolution.controller import EvolutionController
        ctrl = EvolutionController()
        llm = self.runner._react_params.get("llm")

        loop = _aio.new_event_loop()
        try:
            report = loop.run_until_complete(
                ctrl.absorb_project(git_url, auto_apply=False, llm=llm)
            )
            self.call_from_thread(self._write_system, report.summary())
            if report.proposed:
                self.call_from_thread(
                    self._write_system,
                    "To apply: /evolve full"
                )
        except Exception as e:
            self.call_from_thread(self._write_system, f"Absorb error: {e}")
        finally:
            loop.close()

    # ── /directives handler ─────────────────────────────────────

    def _handle_directives(self, text: str) -> None:
        """Handle /directives for persistent cross-session rules."""
        parts = text.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else "list"
        conv_store = self.runner.session_manager.conv_store
        user_id = "cli_user"

        if sub == "list":
            persistent = conv_store.load_persistent_directives(user_id)
            if not persistent:
                self._write_system("(no persistent directives)")
            else:
                lines = [f"Persistent directives ({len(persistent)}):"]
                for i, d in enumerate(persistent):
                    lines.append(f"  [{i}] {d}")
                self._write_system("\n".join(lines))

        elif sub == "add":
            if len(parts) < 3:
                self._write_system("usage: /directives add <rule>")
                return
            directive = parts[2].strip()
            if conv_store.add_persistent_directive(user_id, directive):
                self._write_system(f"Added: {directive}")
            else:
                self._write_system("Already exists")

        elif sub in ("del", "rm"):
            if len(parts) < 3:
                self._write_system("usage: /directives del <index>")
                return
            try:
                idx = int(parts[2])
            except ValueError:
                self._write_system("index must be a number")
                return
            if conv_store.remove_persistent_directive(user_id, idx):
                self._write_system(f"Removed directive [{idx}]")
            else:
                self._write_system(f"Invalid index [{idx}]")

        elif sub == "clear":
            conv_store.save_persistent_directives(user_id, [])
            self._write_system("All persistent directives cleared")

        else:
            self._write_system(
                "/directives subcommands:\n"
                "  list          -- show persistent directives\n"
                "  add <rule>    -- add (persists across sessions)\n"
                "  del <n>       -- remove by index\n"
                "  clear         -- remove all"
            )

    # ── Helpers ───────────────────────────────────────────────

    def _write_chat(self, role: str, content: str) -> None:
        chat = self.query_one("#chat-log", TextArea)
        line = f"\n{role}:\n{content}\n"
        chat.insert(line, chat.document.end)
        chat.scroll_end(animate=False)

    def _write_system(self, msg: str) -> None:
        chat = self.query_one("#chat-log", TextArea)
        chat.insert(f"\n--- {msg}\n", chat.document.end)
        chat.scroll_end(animate=False)

    def _clear_chat(self) -> None:
        chat = self.query_one("#chat-log", TextArea)
        chat.clear()

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
        lines = []
        for s in sessions:
            sid = s["session_id"][-16:]
            lines.append(f"  {sid}  {s['turns']} turns  {s.get('preview', '')[:40]}")
        self._write_system("\n".join(lines))

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
            self._clear_chat()
            self._write_system(
                f"Resumed: {self.session_id[-16:]}  {matched['turns']} turns"
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
            lines = []
            for m in models:
                marker = " *" if m["id"] == cur else ""
                lines.append(f"  {m['id']}: {m['model']}{marker}")
            self._write_system("\n".join(lines))
            return
        model_id = parts[1].strip()
        if self.runner.switch_model(model_id):
            self._write_system(f"Model -> {model_id}")
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

    runner.print_status()
    print()
    app = MulAgentApp(runner, session_id)
    app.run()
