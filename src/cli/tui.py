"""Textual-based TUI for mul-agent.

Three-panel layout:
- Left: Sessions + Favorite prompts
- Center: Main chat (Rendered / Raw tabs)
- Right: Activity panel (task progress, tool calls, warnings)

Top status bar, bottom input with shortcuts.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text as RichText


# ── Favorites persistence ────────────────────────────────────
_FAV_FILE = Path.home() / ".mulagent" / "favorites.json"


def _load_favorites() -> list[str]:
    try:
        return json.loads(_FAV_FILE.read_text()) if _FAV_FILE.exists() else []
    except Exception:
        return []


def _save_favorites(favs: list[str]) -> None:
    _FAV_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FAV_FILE.write_text(json.dumps(favs, ensure_ascii=False, indent=2))


# ── Command definitions for auto-completion ────────────────────
COMMANDS: list[tuple[str, str]] = [
    ("/new", "Start a new session"),
    ("/help", "Show available commands"),
    ("/sessions", "List recent sessions"),
    ("/resume", "Resume a session by ID fragment"),
    ("/model", "Show or switch LLM model"),
    ("/fav", "List favorite prompts"),
    ("/fav add", "Save current input as favorite"),
    ("/fav del", "Remove a favorite by index"),
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
    EditScreen { align: center middle; }
    #edit-container {
        width: 90%; height: 80%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #edit-title { text-style: bold; padding: 0 0 1 0; }
    #edit-area { height: 1fr; }
    #edit-toolbar { height: 3; padding: 1 0 0 0; }
    .edit-btn { margin: 0 2 0 0; }
    #edit-hint {
        height: 1; color: $text; background: $primary;
        padding: 0 2; text-style: bold;
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
            yield Label(f"Editing turn [{self._index}] ({self._role})", id="edit-title")
            yield TextArea(
                self._content, id="edit-area",
                soft_wrap=True, show_line_numbers=True, language=None, theme="css",
            )
            yield Label(" Ctrl+S = Save  |  Esc = Cancel ", id="edit-hint")
            with Horizontal(id="edit-toolbar"):
                yield Button("Save (Ctrl+S)", variant="success", id="btn-save", classes="edit-btn")
                yield Button("Cancel (Esc)", variant="default", id="btn-cancel", classes="edit-btn")

    def on_mount(self) -> None:
        self.query_one("#edit-area", TextArea).focus()

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_save(self) -> None:
        self.dismiss(self.query_one("#edit-area", TextArea).text)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Command Palette ModalScreen ──────────────────────────────────

class CommandPalette(ModalScreen[str | None]):
    """VS Code-style command palette: search and select commands."""

    CSS = """
    CommandPalette { align: center middle; }
    #palette-container {
        width: 60; height: 24;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #palette-input { margin: 0 0 1 0; }
    #palette-list { height: 1fr; }
    """

    BINDINGS = [Binding("escape", "cancel", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-container"):
            yield Input(placeholder="Search commands...", id="palette-input")
            yield OptionList(id="palette-list")

    def on_mount(self) -> None:
        self._refresh_list("")
        self.query_one("#palette-input", Input).focus()

    @on(Input.Changed, "#palette-input")
    def _on_filter(self, event: Input.Changed) -> None:
        self._refresh_list(event.value)

    def _refresh_list(self, query: str) -> None:
        ol = self.query_one("#palette-list", OptionList)
        ol.clear_options()
        q = query.lower()
        for cmd, desc in COMMANDS:
            if not q or q in cmd or q in desc.lower():
                ol.add_option(Option(f"{cmd}  — {desc}", id=cmd))

    @on(OptionList.OptionSelected, "#palette-list")
    def _on_select(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── TUI App ──────────────────────────────────────────────────────

class MulAgentApp(App):
    """mul-agent Terminal UI — three-panel layout."""

    TITLE = "mul-agent"
    theme = "mulagent-dark"

    CSS = """
    /* ── Force pure black everywhere (need !important to override DEFAULT_CSS) ── */
    Screen { background: #000000 !important; }
    RichLog, TextArea, ListView, OptionList, Input, Static, Label,
    Vertical, Horizontal, VerticalScroll,
    TabbedContent, ContentSwitcher, TabPane, ListItem {
        background: #000000 !important;
        background-tint: transparent !important;
    }
    * {
        scrollbar-background: #111111;
        scrollbar-color: #333333;
    }

    /* ── Top bar ── */
    #top-bar {
        height: 1;
        color: #44aaff;
        padding: 0 2;
        text-style: bold;
    }

    /* ── Main three-column layout ── */
    #main { height: 1fr; }

    #left-panel {
        width: 24;
        border-right: solid #333333;
        padding: 0;
    }
    #left-panel-title {
        text-style: bold;
        padding: 0 1;
        color: #44aaff;
    }
    #session-list { height: 1fr; overflow-y: auto; }
    #session-list > ListItem {
        padding: 0 1;
        color: #66aaff;
        text-style: underline;
    }
    #session-list > ListItem:hover {
        color: #99ccff;
    }
    #fav-title {
        text-style: bold;
        padding: 0 1;
        color: #44aaff;
        border-top: solid #333333;
    }
    #fav-list { height: auto; max-height: 10; overflow-y: auto; }
    #fav-list > ListItem { padding: 0 1; }

    #center-panel { width: 1fr; }
    TabbedContent { height: 1fr; }
    TabbedContent ContentSwitcher { height: 1fr; }
    TabPane { height: 1fr; padding: 0; }

    /* ── Tab buttons (text-only, no bg change) ── */
    ContentTab {
        color: #666666;
        padding: 0 2;
        margin: 0 1 0 0;
    }
    ContentTab:hover {
        color: #aaaaaa;
    }
    ContentTab.-active {
        color: #44aaff;
        text-style: bold;
    }
    Underline {
        height: 0;
    }
    #chat-log-rich { height: 1fr; padding: 0 1; }
    #chat-log-raw { height: 1fr; padding: 0 1; }

    #right-panel {
        width: 28;
        border-left: solid #333333;
        padding: 0;
    }
    .panel-section-title {
        text-style: bold;
        padding: 0 1;
        color: #44aaff;
    }
    #activity-log { height: 1fr; padding: 0 1; }
    #progress-bar { height: 1; padding: 0 1; color: $warning; }

    /* ── Bottom bar ── */
    #input-wrapper { dock: bottom; height: auto; }
    #cmd-popup {
        display: none;
        height: auto; max-height: 14;
        padding: 0 1;
        border: solid #333333; margin: 0 1;
    }
    #cmd-popup.visible { display: block; }
    #input-bar { padding: 0 1; }
    #shortcut-bar {
        height: 1;
        padding: 0 1;
        color: #555555;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+l", "clear_display", "Clear"),
        Binding("ctrl+f", "add_favorite", "Fav"),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(self, runner: Any, session_id: str, **kwargs):
        super().__init__(**kwargs)
        from textual.theme import Theme
        self.register_theme(Theme(
            name="mulagent-dark",
            primary="#4488cc",
            secondary="#666666",
            accent="#44aaff",
            warning="#ffaa00",
            error="#ff4444",
            success="#44cc44",
            background="#000000",
            surface="#000000",
            panel="#000000",
            boost="#000000",
            dark=True,
        ))
        self.theme = "mulagent-dark"
        self.runner = runner
        self.session_id = session_id
        self._busy = False
        self._popup_visible = False
        self._filtered_cmds: list[tuple[str, str]] = []
        self._messages: list[dict[str, str]] = []
        self._input_history: list[str] = []
        self._history_index: int = -1
        self._history_draft: str = ""
        self._sessions_cache: list[dict] = []
        self._favorites: list[str] = _load_favorites()
        self._task_start: float = 0.0
        self._last_latency: float = 0.0
        self._last_tokens: str = ""

    def compose(self) -> ComposeResult:
        # ── Top status bar ──
        yield Static(self._render_top_bar(), id="top-bar")

        # ── Three-column main area ──
        with Horizontal(id="main"):
            # Left: sessions + favorites
            with Vertical(id="left-panel"):
                yield Static("[bold cyan]SESSIONS[/]", id="left-panel-title")
                yield ListView(id="session-list")
                yield Static("[bold cyan]FAVORITES[/]", id="fav-title")
                yield ListView(id="fav-list")

            # Center: chat with tabs
            with Vertical(id="center-panel"):
                with TabbedContent("渲染", "原始", id="chat-tabs"):
                    with TabPane("渲染", id="tab-rich"):
                        yield RichLog(highlight=True, markup=True, wrap=True, id="chat-log-rich")
                    with TabPane("原始", id="tab-raw"):
                        yield TextArea(
                            "", id="chat-log-raw", read_only=True,
                            show_line_numbers=False, soft_wrap=True, language=None, theme="css",
                        )

            # Right: activity panel
            with Vertical(id="right-panel"):
                yield Static("[bold cyan]ACTIVITY[/]", classes="panel-section-title")
                yield RichLog(highlight=True, markup=True, wrap=True, id="activity-log")
                yield Static("", id="progress-bar")

        # ── Bottom: input + shortcuts ──
        with Vertical(id="input-wrapper"):
            yield OptionList(id="cmd-popup")
            yield Input(
                placeholder="Enter message or / for commands...",
                id="input-bar",
            )
        yield Static(
            " Ctrl+N New  Ctrl+P Commands  Ctrl+L Clear  Ctrl+F Fav  ↑↓ History  Tab Complete  Ctrl+Q Quit",
            id="shortcut-bar",
        )

    def on_mount(self) -> None:
        self._refresh_sessions()
        self._refresh_favorites()
        self._update_top_bar()
        # Write color test to activity log for debugging
        activity = self.query_one("#activity-log", RichLog)
        activity.write(RichText("Color test:", style="bold"))
        activity.write(RichText("  CYAN", style="bold cyan"))
        activity.write(RichText("  GREEN", style="bold green"))
        activity.write(RichText("  RED", style="bold red"))
        activity.write(RichText("  YELLOW", style="bold yellow"))
        self.query_one("#input-bar", Input).focus()

    # ── Top bar ──────────────────────────────────────────────

    def _render_top_bar(self) -> str:
        sid = self.session_id[-8:] if self.session_id else "none"
        model = getattr(self.runner, "current_model", "?")
        parts = [
            "[bold cyan]mul-agent[/]",
            f"[dim]session:[/] {sid}",
            f"[dim]model:[/] {model}",
        ]
        if self._last_latency:
            parts.append(f"[dim]⏱[/] {self._last_latency:.1f}s")
        if self._last_tokens:
            parts.append(self._last_tokens)
        if self._busy:
            parts.append("[bold yellow]● running[/]")
        return " │ ".join(parts)

    def _update_top_bar(self) -> None:
        try:
            self.query_one("#top-bar", Static).update(self._render_top_bar())
        except NoMatches:
            pass

    # ── Session list: click or Enter to resume ────────────────

    @on(ListView.Selected, "#session-list")
    def _on_session_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self._sessions_cache):
            session = self._sessions_cache[idx]
            self.session_id = session["session_id"]
            self.runner.session_manager.resume_session("cli_user", "cli", self.session_id)
            self._clear_chat()
            self._load_session_history()
            self._refresh_sessions()
            self._update_top_bar()
            self.query_one("#input-bar", Input).focus()

    def _load_session_history(self) -> None:
        conv_store = self.runner.session_manager.conv_store
        turns = conv_store.list_turns(self.session_id)
        if not turns:
            self._write_system(f"Resumed: {self.session_id[-16:]} (empty)")
            return
        for t in turns:
            role = "You" if t["role"] == "user" else "Assistant"
            self._write_chat(role, t["content"])
        self._write_system(f"Resumed: {self.session_id[-16:]} ({len(turns)} turns)")

    # ── Favorites: click to send ──────────────────────────────

    @on(ListView.Selected, "#fav-list")
    def _on_fav_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self._favorites):
            inp = self.query_one("#input-bar", Input)
            inp.value = self._favorites[idx]
            inp.cursor_position = len(inp.value)
            inp.focus()

    def _refresh_favorites(self) -> None:
        try:
            lv = self.query_one("#fav-list", ListView)
            lv.clear()
            for i, fav in enumerate(self._favorites):
                lv.append(ListItem(Label(f"📌 {fav[:22]}")))
        except NoMatches:
            pass

    # ── Command auto-completion ────────────────────────────────

    @on(Input.Changed, "#input-bar")
    def _on_input_changed(self, event: Input.Changed) -> None:
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
        popup = self.query_one("#cmd-popup", OptionList)
        popup.clear_options()
        for cmd, desc in items:
            popup.add_option(Option(f"{cmd}  — {desc}", id=cmd))
        popup.add_class("visible")
        self._popup_visible = True
        if len(items) > 0:
            popup.highlighted = 0

    def _hide_popup(self) -> None:
        popup = self.query_one("#cmd-popup", OptionList)
        popup.remove_class("visible")
        self._popup_visible = False

    def _accept_completion(self) -> None:
        popup = self.query_one("#cmd-popup", OptionList)
        idx = popup.highlighted
        if idx is not None and 0 <= idx < len(self._filtered_cmds):
            cmd, _desc = self._filtered_cmds[idx]
            inp = self.query_one("#input-bar", Input)
            needs_arg = cmd in (
                "/resume", "/model", "/fav add", "/fav del",
                "/modify view", "/modify edit", "/modify del",
                "/directives add", "/directives del",
            )
            inp.value = cmd + (" " if needs_arg else "")
            inp.cursor_position = len(inp.value)
            self._hide_popup()
            inp.focus()

    @on(OptionList.OptionSelected, "#cmd-popup")
    def _on_popup_selected(self, event: OptionList.OptionSelected) -> None:
        self._accept_completion()

    # ── Key handling (popup + history) ────────────────────────

    def on_key(self, event) -> None:
        # ── Popup navigation ──
        if self._popup_visible:
            popup = self.query_one("#cmd-popup", OptionList)
            if event.key == "down":
                event.prevent_default(); event.stop()
                h = popup.highlighted
                if h is None:
                    popup.highlighted = 0
                elif h < len(self._filtered_cmds) - 1:
                    popup.highlighted = h + 1
                return
            elif event.key == "up":
                event.prevent_default(); event.stop()
                h = popup.highlighted
                if h is not None and h > 0:
                    popup.highlighted = h - 1
                return
            elif event.key == "tab":
                event.prevent_default(); event.stop()
                self._accept_completion()
                return
            elif event.key == "escape":
                event.prevent_default(); event.stop()
                self._hide_popup()
                self.query_one("#input-bar", Input).focus()
                return

        # ── Input history ──
        try:
            inp = self.query_one("#input-bar", Input)
        except NoMatches:
            return
        if not inp.has_focus:
            return

        if event.key == "up" and self._input_history:
            event.prevent_default(); event.stop()
            if self._history_index == -1:
                self._history_draft = inp.value
                self._history_index = len(self._input_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            inp.value = self._input_history[self._history_index]
            inp.cursor_position = len(inp.value)
        elif event.key == "down" and self._history_index >= 0:
            event.prevent_default(); event.stop()
            if self._history_index < len(self._input_history) - 1:
                self._history_index += 1
                inp.value = self._input_history[self._history_index]
            else:
                self._history_index = -1
                inp.value = self._history_draft
            inp.cursor_position = len(inp.value)

    # ── Input handling ────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self._popup_visible and text.startswith("/"):
            popup = self.query_one("#cmd-popup", OptionList)
            idx = popup.highlighted
            if idx is not None and 0 <= idx < len(self._filtered_cmds):
                cmd, _desc = self._filtered_cmds[idx]
                if text.lower() != cmd and cmd.startswith(text.lower()):
                    self._accept_completion()
                    return
        self._hide_popup()
        event.input.clear()

        # Save to history
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
            if len(self._input_history) > 100:
                self._input_history.pop(0)
        self._history_index = -1

        # ── Command dispatch ──
        cmd = text.lower()
        if cmd in ("/quit", "/exit", "/q"):
            self.exit(); return
        if cmd == "/new":
            self.action_new_session(); return
        if cmd in ("/help", "/h"):
            self._write_system(
                "/new — new session  |  /model <id> — switch model\n"
                "/sessions — list    |  /resume <id> — resume\n"
                "/fav — favorites    |  Ctrl+P — command palette\n"
                "/modify — context management  |  /directives — rules"
            ); return
        if cmd == "/sessions":
            self._show_sessions(); return
        if cmd.startswith("/resume"):
            self._handle_resume(text); return
        if cmd.startswith("/model"):
            self._handle_model(text); return
        if cmd.startswith("/fav"):
            self._handle_fav(text); return
        if cmd.startswith("/modify"):
            self._handle_modify(text); return
        if cmd.startswith("/directives"):
            self._handle_directives(text); return
        if cmd.startswith("/evolve"):
            self._run_evolve(text); return
        if cmd.startswith("/absorb"):
            self._run_absorb(text); return
        if cmd.startswith("/"):
            self._write_system(f"Unknown command: {cmd}"); return

        if self._busy:
            self._write_system("Please wait for the current task to finish.")
            return

        self._write_chat("You", text)
        self._busy = True
        self._task_start = time.monotonic()
        self._update_top_bar()
        self._clear_activity()
        self._run_task(text)

    @work(exclusive=True)
    async def _run_task(self, text: str) -> None:
        """Run agent task in a Textual worker."""
        t0 = time.monotonic()

        async def on_progress(round_num: int, action: str, detail: str) -> None:
            activity = self.query_one("#activity-log", RichLog)
            pbar = self.query_one("#progress-bar", Static)
            if action == "todo_update" and detail:
                try:
                    tasks = json.loads(detail)
                    done = sum(1 for t in tasks if t.get("status") == "done")
                    total = len(tasks)
                    activity.write(RichText(f"📋 Tasks [{done}/{total}]", style="bold"))
                    for t in tasks:
                        st = t.get("status", "pending")
                        icon = "✅" if st == "done" else "🔄" if st == "running" else "⬜"
                        activity.write(RichText(f"  {icon} #{t.get('id','')} {t.get('text','')[:20]}"))
                    pct = int(done / total * 100) if total else 0
                    filled = pct // 5
                    bar = "█" * filled + "░" * (20 - filled)
                    pbar.update(f"{bar} {pct}%")
                except Exception:
                    pass
            elif action == "tool_call" and detail:
                activity.write(RichText(f"🔧 [{round_num}] {detail}", style="cyan"))
            elif action == "thinking":
                pbar.update(f"🔄 Round {round_num} thinking...")
            elif action == "step_text" and detail:
                line = detail.strip().split("\n")[0][:40]
                if line and not line.startswith("{"):
                    activity.write(RichText(f"📝 {line}", style="dim"))
            return None

        try:
            result = await self.runner.run(text, self.session_id, on_progress=on_progress)
            output = result.get("final_output", "(no output)")
            elapsed = time.monotonic() - t0
            status = result.get("status", "?")
            tools_used = result.get("tools_used", [])

            self._last_latency = elapsed
            self._last_tokens = f"{len(output)}c"
            self._write_chat("Assistant", output)

            # Final activity entry
            activity = self.query_one("#activity-log", RichLog)
            activity.write(RichText(
                f"\n{'✅' if status == 'completed' else '⚠️'} {status} │ {elapsed:.1f}s │ {len(tools_used)} tools",
                style="bold green" if status == "completed" else "bold yellow",
            ))
            self.query_one("#progress-bar", Static).update(
                f"{'█' * 20} 100%" if status == "completed" else f"⚠️ {status}"
            )

        except asyncio.CancelledError:
            self._write_system("Task cancelled.")
        except Exception as e:
            self._write_chat("Error", str(e))
            activity = self.query_one("#activity-log", RichLog)
            activity.write(RichText(f"❌ {e}", style="bold red"))
        finally:
            self._busy = False
            self._refresh_sessions()
            self._update_top_bar()

    # ── Actions ───────────────────────────────────────────────

    def action_new_session(self) -> None:
        self.session_id = self.runner.session_manager.new_session("cli_user", "cli")
        self._clear_chat()
        self._clear_activity()
        self._write_system(f"New session: {self.session_id[-16:]}")
        self._refresh_sessions()
        self._update_top_bar()

    def action_focus_input(self) -> None:
        self.query_one("#input-bar", Input).focus()

    def action_command_palette(self) -> None:
        def _on_result(cmd: str | None) -> None:
            if cmd:
                inp = self.query_one("#input-bar", Input)
                inp.value = cmd + " "
                inp.cursor_position = len(inp.value)
                inp.focus()
        self.push_screen(CommandPalette(), callback=_on_result)

    def action_clear_display(self) -> None:
        self._messages.clear()
        self.query_one("#chat-log-raw", TextArea).clear()
        self.query_one("#chat-log-rich", RichLog).clear()
        self._clear_activity()

    def action_add_favorite(self) -> None:
        inp = self.query_one("#input-bar", Input)
        text = inp.value.strip()
        if text and text not in self._favorites:
            self._favorites.append(text)
            _save_favorites(self._favorites)
            self._refresh_favorites()
            self._write_system(f"Saved favorite: {text[:40]}")
        elif not text:
            self._write_system("Type something first, then Ctrl+F to save")

    # ── /fav handler ─────────────────────────────────────────

    def _handle_fav(self, text: str) -> None:
        parts = text.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list" or len(parts) == 1:
            if not self._favorites:
                self._write_system("(no favorites) — use Ctrl+F or /fav add <text>")
            else:
                lines = [f"Favorites ({len(self._favorites)}):"]
                for i, f in enumerate(self._favorites):
                    lines.append(f"  [{i}] 📌 {f[:50]}")
                self._write_system("\n".join(lines))

        elif sub == "add":
            if len(parts) < 3:
                self._write_system("usage: /fav add <prompt text>")
                return
            prompt = parts[2].strip()
            if prompt not in self._favorites:
                self._favorites.append(prompt)
                _save_favorites(self._favorites)
                self._refresh_favorites()
                self._write_system(f"Saved: {prompt[:40]}")
            else:
                self._write_system("Already in favorites")

        elif sub in ("del", "rm"):
            if len(parts) < 3:
                self._write_system("usage: /fav del <index>")
                return
            try:
                idx = int(parts[2])
                if 0 <= idx < len(self._favorites):
                    removed = self._favorites.pop(idx)
                    _save_favorites(self._favorites)
                    self._refresh_favorites()
                    self._write_system(f"Removed: {removed[:40]}")
                else:
                    self._write_system(f"Invalid index [{idx}]")
            except ValueError:
                self._write_system("Index must be a number")
        else:
            self._write_system("/fav list | /fav add <text> | /fav del <n>")

    # ── /modify handler ───────────────────────────────────────

    def _handle_modify(self, text: str) -> None:
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
                lines.append(f"  [{i}] {role}: {preview}...")
            self._write_system("\n".join(lines))

        elif sub == "view":
            if len(parts) < 3:
                self._write_system("usage: /modify view <index>"); return
            try:
                idx = int(parts[2])
            except ValueError:
                self._write_system("index must be a number"); return
            turns = conv_store.list_turns(self.session_id)
            if idx < 0 or idx >= len(turns):
                self._write_system(f"index out of range (0-{len(turns)-1})"); return
            t = turns[idx]
            role = "User" if t["role"] == "user" else "Assistant"
            self._write_system(f"[{idx}] {role}:\n{t['content']}")

        elif sub in ("del", "delete", "rm"):
            if len(parts) < 3:
                self._write_system("usage: /modify del <index|start-end>"); return
            arg = parts[2].strip()
            if "-" in arg and not arg.startswith("-"):
                try:
                    start, end = arg.split("-", 1)
                    count = conv_store.delete_turns_range(self.session_id, int(start), int(end) + 1)
                    self._write_system(f"Deleted {count} turns")
                except ValueError:
                    self._write_system("invalid range")
            else:
                try:
                    idx = int(arg)
                except ValueError:
                    self._write_system("index must be a number"); return
                if conv_store.delete_turn(self.session_id, idx):
                    self._write_system(f"Deleted turn [{idx}]")
                else:
                    self._write_system(f"Failed to delete [{idx}]")

        elif sub == "edit":
            if len(parts) < 3:
                self._write_system("usage: /modify edit <index>"); return
            try:
                idx = int(parts[2])
            except ValueError:
                self._write_system("index must be a number"); return
            turns = conv_store.list_turns(self.session_id)
            if idx < 0 or idx >= len(turns):
                self._write_system(f"index out of range (0-{len(turns)-1})"); return
            t = turns[idx]
            role = "User" if t["role"] == "user" else "Assistant"
            def _on_edit_done(result: str | None) -> None:
                if result is None:
                    self._write_system("Edit cancelled")
                elif conv_store.edit_turn(self.session_id, idx, result):
                    self._write_system(f"Updated turn [{idx}]")
                else:
                    self._write_system(f"Failed to save [{idx}]")
            self.push_screen(EditScreen(idx, role, t["content"]), callback=_on_edit_done)

        elif sub == "clear":
            if conv_store.clear_turns(self.session_id):
                self._write_system("Context cleared")
            else:
                self._write_system("Failed to clear")

        elif sub == "summary":
            summary = conv_store.get_summary(self.session_id)
            self._write_system(f"Summary: {summary}" if summary else "(no summary yet)")

        elif sub == "compress":
            self._write_system("Compressing...")
            self._run_compress()

        else:
            self._write_system(
                "/modify: list | view <n> | del <n> | edit <n> | clear | summary | compress"
            )

    @work(exclusive=True, thread=True)
    def _run_compress(self) -> None:
        import asyncio as _aio
        conv_store = self.runner.session_manager.conv_store
        llm = self.runner._react_params.get("llm")
        try:
            loop = _aio.new_event_loop()
            loop.run_until_complete(conv_store.maybe_summarize(self.session_id, llm=llm))
            loop.close()
            self.call_from_thread(self._write_system, "Context compressed")
        except Exception as e:
            self.call_from_thread(self._write_system, f"Compress failed: {e}")

    # ── /evolve and /absorb handlers ────────────────────────────

    @work(exclusive=True, thread=True)
    def _run_evolve(self, text: str) -> None:
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
                self.call_from_thread(self._write_system, f"Running evolution ({sub})...")
                report = loop.run_until_complete(ctrl.evolve(mode=sub, llm=llm))
                self.call_from_thread(self._write_system, report.summary())
            else:
                self.call_from_thread(self._write_system,
                    "/evolve: propose | diagnose | auto | full | history")
        except Exception as e:
            self.call_from_thread(self._write_system, f"Evolution error: {e}")
        finally:
            loop.close()

    @work(exclusive=True, thread=True)
    def _run_absorb(self, text: str) -> None:
        import asyncio as _aio
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self.call_from_thread(self._write_system, "usage: /absorb <git_url>")
            return
        git_url = parts[1].strip()
        self.call_from_thread(self._write_system, f"Analyzing: {git_url}...")

        from evolution.controller import EvolutionController
        ctrl = EvolutionController()
        llm = self.runner._react_params.get("llm")

        loop = _aio.new_event_loop()
        try:
            report = loop.run_until_complete(ctrl.absorb_project(git_url, auto_apply=False, llm=llm))
            self.call_from_thread(self._write_system, report.summary())
        except Exception as e:
            self.call_from_thread(self._write_system, f"Absorb error: {e}")
        finally:
            loop.close()

    # ── /directives handler ─────────────────────────────────────

    def _handle_directives(self, text: str) -> None:
        parts = text.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else "list"
        conv_store = self.runner.session_manager.conv_store
        user_id = "cli_user"

        if sub == "list":
            persistent = conv_store.load_persistent_directives(user_id)
            if not persistent:
                self._write_system("(no persistent directives)")
            else:
                lines = [f"Directives ({len(persistent)}):"]
                for i, d in enumerate(persistent):
                    lines.append(f"  [{i}] {d}")
                self._write_system("\n".join(lines))
        elif sub == "add":
            if len(parts) < 3:
                self._write_system("usage: /directives add <rule>"); return
            directive = parts[2].strip()
            if conv_store.add_persistent_directive(user_id, directive):
                self._write_system(f"Added: {directive}")
            else:
                self._write_system("Already exists")
        elif sub in ("del", "rm"):
            if len(parts) < 3:
                self._write_system("usage: /directives del <index>"); return
            try:
                idx = int(parts[2])
            except ValueError:
                self._write_system("index must be a number"); return
            if conv_store.remove_persistent_directive(user_id, idx):
                self._write_system(f"Removed [{idx}]")
            else:
                self._write_system(f"Invalid index [{idx}]")
        elif sub == "clear":
            conv_store.save_persistent_directives(user_id, [])
            self._write_system("All directives cleared")
        else:
            self._write_system("/directives: list | add <rule> | del <n> | clear")

    # ── Chat output helpers ───────────────────────────────────

    def _write_chat(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        # Raw panel
        raw = self.query_one("#chat-log-raw", TextArea)
        raw.insert(f"\n{role}:\n{content}\n", raw.document.end)
        raw.scroll_end(animate=False)
        # Rich panel
        rich_log = self.query_one("#chat-log-rich", RichLog)
        role_style = "bold cyan" if role == "Assistant" else "bold green" if role == "You" else "bold red"
        rich_log.write(RichText(f"\n{role}:", style=role_style))
        rich_log.write(RichMarkdown(content))

    def _write_system(self, msg: str) -> None:
        self._messages.append({"role": "system", "content": msg})
        raw = self.query_one("#chat-log-raw", TextArea)
        raw.insert(f"\n--- {msg}\n", raw.document.end)
        raw.scroll_end(animate=False)
        rich_log = self.query_one("#chat-log-rich", RichLog)
        rich_log.write(RichText(f"--- {msg}", style="dim"))

    def _clear_chat(self) -> None:
        self._messages.clear()
        self.query_one("#chat-log-raw", TextArea).clear()
        self.query_one("#chat-log-rich", RichLog).clear()

    def _clear_activity(self) -> None:
        self.query_one("#activity-log", RichLog).clear()
        self.query_one("#progress-bar", Static).update("")

    # ── Session/model helpers ─────────────────────────────────

    def _refresh_sessions(self) -> None:
        try:
            lv = self.query_one("#session-list", ListView)
            lv.clear()
            sessions = self.runner.session_manager.list_sessions("cli_user", limit=15)
            self._sessions_cache = sessions
            for s in sessions:
                sid = s["session_id"][-8:]
                turns = s.get("turns", 0)
                preview = s.get("preview", "")[:14] or "(empty)"
                active = s["session_id"] == self.session_id
                marker = "▸" if active else "○"
                lv.append(ListItem(Label(f"{marker} {sid} [{turns}] {preview}")))
        except NoMatches:
            pass

    def _show_sessions(self) -> None:
        sessions = self.runner.session_manager.list_sessions("cli_user", limit=10)
        if not sessions:
            self._write_system("(no sessions)"); return
        lines = []
        for s in sessions:
            sid = s["session_id"][-16:]
            lines.append(f"  {sid}  {s['turns']} turns  {s.get('preview', '')[:40]}")
        self._write_system("\n".join(lines))

    def _handle_resume(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self._write_system("usage: /resume <session_id_fragment>"); return
        target = parts[1].strip()
        sessions = self.runner.session_manager.list_sessions("cli_user", limit=50)
        matched = next((s for s in sessions if target in s["session_id"]), None)
        if matched:
            self.session_id = matched["session_id"]
            self.runner.session_manager.resume_session("cli_user", "cli", self.session_id)
            self._clear_chat()
            self._load_session_history()
            self._refresh_sessions()
            self._update_top_bar()
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
            self._update_top_bar()
        else:
            available = ", ".join(m["id"] for m in self.runner.llm_manager.list_models())
            self._write_system(f"Unknown model: {model_id}\nAvailable: {available}")


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
