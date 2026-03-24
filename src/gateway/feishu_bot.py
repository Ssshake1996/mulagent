"""Feishu Bot — 飞书长连接入口。

通过 WebSocket 长连接接收飞书消息，直接调用 mul-agent pipeline，
将结果回复到飞书。无需公网 IP。

进度展示：单条消息原地更新（类似 status line），每个节点完成时刷新。

启动方式：
    PYTHONPATH=src python -m gateway.feishu_bot
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import threading
import time
import uuid
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from pathlib import Path

from common.config import get_settings
from common.db import create_session_factory
from common.llm import LLMManager
from common.vector import ensure_collection, get_qdrant_client
from evolution.trace import record_task_trace
from gateway.adapter import SessionManager
from graph.conversation import ConversationStore
from graph.orchestrator import run_react

logger = logging.getLogger("feishu_bot")

# ── Global resources (initialized once) ──────────────────────────────
_react_params: dict[str, Any] = {}  # params for run_react()
_lark_client: lark.Client | None = None
_db_session_factory = None
_session_mgr: SessionManager | None = None

# Message dedup: prevent re-processing retried messages (LRU via OrderedDict)
from collections import OrderedDict
_seen_message_ids: OrderedDict[str, None] = OrderedDict()
_SEEN_MAX = 1000  # cap to avoid unbounded growth

# Per-user task queues: serialize tasks per user to avoid rate-limit blowout
_user_queues: dict[str, asyncio.Queue] = {}
_user_workers: set[str] = set()  # users that already have a running consumer
_TASK_TIMEOUT: int = 640  # Overwritten by config in _init_resources()

# Feishu card content limit (safe threshold below 30KB API limit)
_CARD_TEXT_LIMIT = 4000

# Feedback tracking: task_key → {user_input, result, session_id, message_id}
_feedback_tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
_FEEDBACK_TASKS_MAX = 200

# Running tasks: task_key → asyncio.Event (set = cancelled)
_running_tasks: dict[str, asyncio.Event] = {}


# Tool display names for progress
_TOOL_LABELS: dict[str, str] = {
    "web_search": "搜索",
    "knowledge_recall": "知识检索",
    "web_fetch": "网页抓取",
    "read_file": "文件读取",
    "list_dir": "目录浏览",
    "execute_shell": "命令执行",
    "code_run": "代码运行",
    "write_file": "文件写入",
    "edit_file": "文件编辑",
    "delegate": "专家委托",
    "deep_research": "深度研究",
    "security_scan": "安全扫描",
    "docs_lookup": "文档查询",
    "codemap": "代码结构分析",
    "browser_fetch": "浏览器渲染",
    "sql_query": "数据库查询",
    "git_ops": "Git操作",
    "github_ops": "GitHub操作",
}


def _init_resources():
    """Initialize all shared resources (called once at startup)."""
    global _react_params, _lark_client, _db_session_factory, _TASK_TIMEOUT, _session_mgr

    settings = get_settings()

    # LLM
    llm_manager = LLMManager()
    default_llm = llm_manager.default

    # Database (optional)
    _db_session_factory = None
    if settings.database and settings.database.url:
        try:
            _db_session_factory = create_session_factory()
            logger.info("Database connected")
        except Exception as e:
            logger.warning("Database unavailable: %s", e)

    # Qdrant
    qdrant = get_qdrant_client()
    collection_name = settings.qdrant.collection_name if settings.qdrant else "case_library"
    ensure_collection(qdrant, collection_name)

    # Session manager (shared with CLI / Desktop via same data dir)
    _session_mgr = SessionManager(ConversationStore())

    # Timeout: react.timeout (inner) + 40s buffer for LLM synthesis on timeout
    _TASK_TIMEOUT = settings.react.timeout + 40

    # Set shared LLM for embedding fallback (tier 2: keyword extraction)
    from common.vector import set_shared_llm
    set_shared_llm(default_llm)

    # ReAct params — much simpler than the old pipeline
    _react_params = dict(
        llm=default_llm,
        qdrant=qdrant,
        collection_name=collection_name,
        timeout=settings.react.timeout,
    )

    # Lark API client (for sending replies)
    _lark_client = (
        lark.Client.builder()
        .app_id(settings.feishu.app_id)
        .app_secret(settings.feishu.app_secret)
        .log_level(lark.LogLevel.WARNING)
        .build()
    )

    # ── Session cleanup: remove old conversations ──
    _session_mgr.conv_store.cleanup_old_sessions(max_age_days=30)

    logger.info(
        "Resources initialized: LLM=%s, Qdrant=ready, mode=ReAct",
        llm_manager.default_id,
    )


# ── Async event loop in background thread ────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        t = threading.Thread(target=_loop.run_forever, daemon=True)
        t.start()
    return _loop


# ── Feishu message helpers ───────────────────────────────────────────

def _split_long_text(text: str, limit: int = _CARD_TEXT_LIMIT) -> list[str]:
    """Split text into chunks that fit within Feishu card limit.

    Splits at paragraph boundaries when possible.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Find a good split point (paragraph boundary)
        split_at = text.rfind("\n\n", 0, limit)
        if split_at < limit // 2:
            split_at = text.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


def _build_card(text: str, *, with_feedback: bool = False, task_key: str = "",
                with_retry: bool = False, retry_text: str = "") -> str:
    """Build a Feishu interactive card JSON with markdown content.

    Args:
        with_feedback: If True, append 👍👎 feedback buttons.
        task_key: Unique key for this task, used in button callback value.
        with_retry: If True, append a retry button.
        retry_text: Original user input for retry.
    """
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": text,
        }
    ]

    actions = []
    if with_feedback and task_key:
        actions.extend([
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "👍 有帮助"},
                "type": "primary",
                "value": {"action": "feedback", "rating": 5, "task_key": task_key},
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "👎 不满意"},
                "type": "danger",
                "value": {"action": "feedback", "rating": 1, "task_key": task_key},
            },
        ])

    if with_retry and task_key:
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "🔄 重试"},
            "type": "default",
            "value": {"action": "retry", "task_key": task_key, "text": retry_text[:500]},
        })

    if actions:
        elements.append({"tag": "hr"})
        elements.append({"tag": "action", "actions": actions})

    card = {"elements": elements}
    return json.dumps(card, ensure_ascii=False)


def _reply_card(message_id: str, text: str) -> str | None:
    """Reply with a card message. Returns the new message_id for later patching."""
    body = ReplyMessageRequestBody.builder() \
        .msg_type("interactive") \
        .content(_build_card(text)) \
        .build()
    req = ReplyMessageRequest.builder() \
        .message_id(message_id) \
        .request_body(body) \
        .build()
    resp = _lark_client.im.v1.message.reply(req)
    if not resp.success():
        logger.error("Reply failed: code=%s, msg=%s", resp.code, resp.msg)
        return None
    return resp.data.message_id


def _send_card(chat_id: str, text: str) -> str | None:
    """Send a new card to a chat (used for split messages). Returns message_id."""
    body = CreateMessageRequestBody.builder() \
        .msg_type("interactive") \
        .receive_id(chat_id) \
        .content(_build_card(text)) \
        .build()
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(body) \
        .build()
    resp = _lark_client.im.v1.message.create(req)
    if not resp.success():
        logger.error("Send card failed: code=%s, msg=%s", resp.code, resp.msg)
        return None
    return resp.data.message_id


def _patch_card(message_id: str, text: str, *, with_feedback: bool = False,
                task_key: str = "", with_retry: bool = False, retry_text: str = ""):
    """Update an existing card message in-place (飞书 patch 仅支持卡片)."""
    body = PatchMessageRequestBody.builder() \
        .content(_build_card(text, with_feedback=with_feedback, task_key=task_key,
                             with_retry=with_retry, retry_text=retry_text)) \
        .build()
    req = PatchMessageRequest.builder() \
        .message_id(message_id) \
        .request_body(body) \
        .build()
    resp = _lark_client.im.v1.message.patch(req)
    if not resp.success():
        logger.error("Patch failed: code=%s, msg=%s", resp.code, resp.msg)


# ── Status line rendering ────────────────────────────────────────────

class ProgressTracker:
    """Tracks ReAct loop progress, renders status line, patches Feishu card."""

    def __init__(self, status_msg_id: str, task_key: str = "", max_rounds: int = 15):
        self.status_msg_id = status_msg_id
        self.task_key = task_key
        self.max_rounds = max_rounds
        self.start = time.monotonic()
        self.current_round = 0
        self.actions: list[str] = []  # completed action lines
        self.cancelled = False

    def _build_progress_card(self, text: str) -> str:
        """Build a progress card with cancel button."""
        elements: list[dict] = [
            {"tag": "markdown", "content": text},
        ]
        if self.task_key and not self.cancelled:
            elements.append({
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "⏹ 取消任务"},
                        "type": "default",
                        "value": {"action": "cancel", "task_key": self.task_key},
                    },
                ],
            })
        return json.dumps({"elements": elements}, ensure_ascii=False)

    def render(self) -> str:
        lines = []
        if self.actions:
            for action in self.actions[-6:]:  # Show last 6 actions
                lines.append(action)

        if self.current_round > 0:
            # Progress bar with percentage
            pct = min(95, int(self.current_round / self.max_rounds * 100))
            filled = pct // 5
            bar = "█" * filled + "░" * (20 - filled)
            lines.append(f"\n{bar} {pct}%")
            lines.append(f"🔄 **思考中** (round {self.current_round}/{self.max_rounds})")

        elapsed = time.monotonic() - self.start
        lines.append(f"⏱ {elapsed:.1f}s")
        return "\n".join(lines) if lines else "🔄 **启动中**..."

    def update_card(self):
        """Update the progress card with cancel button."""
        body = PatchMessageRequestBody.builder() \
            .content(self._build_progress_card(self.render())) \
            .build()
        req = PatchMessageRequest.builder() \
            .message_id(self.status_msg_id) \
            .request_body(body) \
            .build()
        resp = _lark_client.im.v1.message.patch(req)
        if not resp.success():
            logger.error("Patch failed: code=%s, msg=%s", resp.code, resp.msg)

    async def on_progress(self, round_num: int, action: str, detail: str):
        """Callback from react_loop for each round/tool call."""
        # Check for cancellation
        cancel_event = _running_tasks.get(self.task_key)
        if cancel_event and cancel_event.is_set():
            self.cancelled = True
            raise asyncio.CancelledError("Task cancelled by user")

        self.current_round = round_num

        if action == "tool_call" and detail:
            label = _TOOL_LABELS.get(detail, detail)
            self.actions.append(f"🔧 [{round_num}] {label}")
        elif action == "thinking":
            pass  # Just update round number

        self.update_card()


# ── Streaming task execution ─────────────────────────────────────────

async def _run_task_react(user_input: str, session_id: str, tracker: ProgressTracker):
    """Execute task using the ReAct orchestrator with progress callbacks.

    Loads conversation history and accumulated directives from the session
    to enable multi-turn dialogue.
    """
    conv = _session_mgr.conv_store if _session_mgr else None

    # Load conversation context
    history = ""
    session_directives = None
    if conv:
        history = conv.get_history_for_prompt(session_id, max_turns=10)
        session_directives = conv.get_directives(session_id) or None
        conv.append_turn(session_id, "user", user_input)

    result = await run_react(
        user_input=user_input,
        on_progress=tracker.on_progress,
        conversation_history=history,
        session_directives=session_directives,
        **_react_params,  # includes timeout from config
    )

    # Record assistant turn and persist directives
    if conv:
        output = result.get("final_output", "")
        conv.append_turn(session_id, "assistant", output)
        directives = result.get("directives", [])
        if directives:
            conv.save_directives(session_id, directives)

    return result


async def _record_trace(user_input: str, session_id: str, result: dict[str, Any]):
    """Persist trace to DB (fire-and-forget)."""
    if _db_session_factory is None:
        return
    try:
        async with _db_session_factory() as db:
            await record_task_trace(
                db,
                session_id=session_id,
                user_input=user_input,
                intent=result.get("intent", "react"),
                dag_plan=None,
                subtask_results={},
                final_output=result.get("final_output", ""),
                status=result.get("status", "unknown"),
                subtasks=[],
            )
    except Exception as e:
        logger.warning("Trace recording failed: %s", e)


# ── Message handler ──────────────────────────────────────────────────

# Supported file types for text extraction
_TEXT_FILE_EXTS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml",
                   ".py", ".js", ".ts", ".java", ".go", ".rs", ".sh", ".sql",
                   ".html", ".css", ".log", ".conf", ".ini", ".toml"}

_TEMP_DIR = Path("/tmp/mulagent_files")


def _download_resource(message_id: str, file_key: str, res_type: str = "file") -> bytes | None:
    """Download a message resource (image/file) from Feishu.

    Args:
        message_id: The Feishu message ID.
        file_key: The file_key from message content.
        res_type: "image" or "file".

    Returns:
        File bytes, or None on failure.
    """
    from lark_oapi.api.im.v1 import GetMessageResourceRequest

    req = (
        GetMessageResourceRequest.builder()
        .message_id(message_id)
        .file_key(file_key)
        .type(res_type)
        .build()
    )
    resp = _lark_client.im.v1.message_resource.get(req)
    if resp.code != 0:
        logger.warning("Resource download failed: code=%s, msg=%s", resp.code, resp.msg)
        return None
    return resp.file.read() if resp.file else None


def _extract_message(data) -> tuple[str | None, list[str]]:
    """Extract user input from a Feishu message event.

    Supports:
    - text: plain text messages
    - image: download image, save to temp, return path reference
    - file: download file, extract text content if possible
    - post (rich text): extract plain text from post content

    Returns:
        (text_content, list_of_file_paths) — text may be None if nothing extractable.
    """
    file_paths: list[str] = []
    try:
        msg = data.event.message
        msg_type = msg.message_type
        content = json.loads(msg.content)
        message_id = msg.message_id

        if msg_type == "text":
            return content.get("text", "").strip(), file_paths

        if msg_type == "image":
            image_key = content.get("image_key", "")
            if not image_key:
                return None, file_paths
            img_bytes = _download_resource(message_id, image_key, "image")
            if img_bytes:
                _TEMP_DIR.mkdir(parents=True, exist_ok=True)
                img_path = _TEMP_DIR / f"{image_key}.png"
                img_path.write_bytes(img_bytes)
                file_paths.append(str(img_path))
                return f"[用户发送了一张图片，已保存至 {img_path}，请分析图片内容]", file_paths
            return "[用户发送了一张图片，但下载失败]", file_paths

        if msg_type == "file":
            file_key = content.get("file_key", "")
            file_name = content.get("file_name", "unknown_file")
            if not file_key:
                return None, file_paths
            file_bytes = _download_resource(message_id, file_key, "file")
            if file_bytes:
                _TEMP_DIR.mkdir(parents=True, exist_ok=True)
                file_path = _TEMP_DIR / file_name
                file_path.write_bytes(file_bytes)
                file_paths.append(str(file_path))

                # Try to extract text from known text formats
                ext = file_path.suffix.lower()
                if ext in _TEXT_FILE_EXTS and len(file_bytes) < 200_000:
                    try:
                        text_content = file_bytes.decode("utf-8", errors="replace")
                        # Truncate very long files
                        if len(text_content) > 5000:
                            text_content = text_content[:5000] + f"\n... (文件共 {len(file_bytes)} 字节，已截取前 5000 字符)"
                        return (
                            f"[用户发送了文件: {file_name}]\n"
                            f"文件内容:\n```\n{text_content}\n```",
                            file_paths,
                        )
                    except Exception:
                        pass

                return (
                    f"[用户发送了文件: {file_name}，已保存至 {file_path}，"
                    f"大小 {len(file_bytes)} 字节。请使用 read_file 工具读取]",
                    file_paths,
                )
            return f"[用户发送了文件: {file_name}，但下载失败]", file_paths

        if msg_type == "post":
            # Rich text: extract plain text from all paragraphs
            texts = []
            for lang_content in content.values():
                title = lang_content.get("title", "")
                if title:
                    texts.append(title)
                for para in lang_content.get("content", []):
                    for elem in para:
                        if elem.get("tag") == "text":
                            texts.append(elem.get("text", ""))
                        elif elem.get("tag") == "a":
                            texts.append(elem.get("href", ""))
            return "\n".join(texts).strip() or None, file_paths

        # Unsupported message type
        logger.info("Unsupported message type: %s", msg_type)
        return None, file_paths

    except (json.JSONDecodeError, AttributeError, Exception) as e:
        logger.warning("Failed to extract message: %s", e)
        return None, file_paths


def _extract_user_id(data) -> str:
    """Extract sender user_id from a Feishu message event."""
    try:
        return data.event.sender.sender_id.user_id or "unknown"
    except (AttributeError, Exception):
        return "unknown"


def _extract_chat_id(data) -> str:
    """Extract chat_id from a Feishu message event.

    Returns a unique identifier for the conversation context:
    - Private chat: the chat_id between user and bot
    - Group chat: the group's chat_id
    """
    try:
        return data.event.message.chat_id or "unknown"
    except (AttributeError, Exception):
        return "unknown"


_HELP_TEXT = """\
**可用命令**

| 命令 | 说明 |
|------|------|
| `/new` | 开启新会话（清除上下文） |
| `/resume` | 查看最近的会话列表 |
| `/resume <id>` | 恢复指定会话 |
| `/help` | 显示本帮助 |

**支持的消息类型**: 文字、图片、文件（代码/文本）、富文本

直接发送消息即可下发任务，支持多轮对话。"""


def _on_message(data) -> None:
    """Handle incoming Feishu message (P2 event handler, single arg).

    Supports both private chat and group chat (@mention).
    In group chat, only responds when the bot is @mentioned.
    """
    # ── Group chat: only respond to @mentions ──
    try:
        chat_type = data.event.message.chat_type
        if chat_type == "group":
            mentions = getattr(data.event.message, "mentions", None)
            if not mentions:
                return  # Not @mentioned, ignore
            # Check if any mention targets the bot
            bot_mentioned = False
            for m in mentions:
                if hasattr(m, "id") and hasattr(m.id, "app_id"):
                    bot_mentioned = True
                    break
                # Some SDK versions use name == bot name
                if hasattr(m, "key"):
                    bot_mentioned = True
                    break
            if not bot_mentioned:
                return
    except (AttributeError, Exception):
        pass  # Private chat or unable to determine — proceed normally

    text, file_paths = _extract_message(data)
    if not text:
        return

    # Strip @mention tags from text
    import re
    text = re.sub(r'@\S+\s*', '', text).strip()
    if not text:
        return

    message_id = data.event.message.message_id

    # ── Dedup: skip already-processed messages (LRU) ──
    if message_id in _seen_message_ids:
        logger.debug("Duplicate message %s, skipping", message_id)
        return
    _seen_message_ids[message_id] = None
    while len(_seen_message_ids) > _SEEN_MAX:
        _seen_message_ids.popitem(last=False)  # evict oldest

    user_id = _extract_user_id(data)
    chat_id = _extract_chat_id(data)
    cmd = text.strip().lower()

    # Handle /help command
    if cmd in ("/help", "/h"):
        _reply_card(message_id, _HELP_TEXT)
        return

    # Handle /new command — create new session and reply confirmation
    if cmd in ("/new", "/new session"):
        if _session_mgr:
            session_id = _session_mgr.new_session(user_id, chat_id)
            _reply_card(message_id, f"✅ 新会话已创建\n\nSession: `{session_id}`")
            logger.info("New session for user %s (chat %s): %s", user_id, chat_id[-8:], session_id)
        return

    # Handle /resume command — list or switch sessions
    if cmd == "/resume":
        if _session_mgr:
            sessions = _session_mgr.list_sessions(user_id, limit=8)
            if not sessions:
                _reply_card(message_id, "没有找到历史会话。\n\n发送消息即可开始新对话。")
            else:
                lines = ["**最近的会话**\n"]
                for s in sessions:
                    sid_short = s["session_id"][-12:]
                    lines.append(
                        f"- `{sid_short}` — {s['preview']}... "
                        f"({s['turns']}轮, {s['updated_at'][:10]})"
                    )
                lines.append("\n发送 `/resume <id>` 恢复会话")
                _reply_card(message_id, "\n".join(lines))
        else:
            _reply_card(message_id, "会话存储未初始化。")
        return

    if cmd.startswith("/resume "):
        target = text.strip().split(maxsplit=1)[1].strip()
        if _session_mgr:
            sessions = _session_mgr.list_sessions(user_id, limit=50)
            matched = None
            for s in sessions:
                if target in s["session_id"]:
                    matched = s
                    break
            if matched:
                _session_mgr.resume_session(user_id, chat_id, matched["session_id"])
                _reply_card(
                    message_id,
                    f"✅ 已恢复会话\n\n"
                    f"Session: `{matched['session_id']}`\n"
                    f"历史对话: {matched['turns']}轮\n"
                    f"预览: {matched['preview']}..."
                )
                logger.info("Resumed session %s for user %s", matched["session_id"], user_id)
            else:
                _reply_card(message_id, f"未找到匹配 `{target}` 的会话。\n\n发送 `/resume` 查看列表。")
        else:
            _reply_card(message_id, "会话存储未初始化。")
        return

    session_id = _session_mgr.get_or_create(user_id, chat_id) if _session_mgr else "fallback"
    # Ensure conversation file exists for this session
    if _session_mgr:
        _session_mgr.ensure_conversation(session_id, user_id)
    logger.info("Received from Feishu [%s] (chat:%s): %.80s...", session_id, chat_id[-8:], text)

    # Generate task_key for cancel/feedback tracking
    task_key = uuid.uuid4().hex[:12]

    # Send initial status card, get its message_id for patching
    settings = get_settings()
    tracker = ProgressTracker("__placeholder__", task_key=task_key,
                              max_rounds=settings.react.max_rounds)
    initial_text = "🔄 **思考中**..."
    status_msg_id = _reply_card(message_id, initial_text)
    if not status_msg_id:
        logger.error("Failed to send initial status message")
        return
    tracker.status_msg_id = status_msg_id

    # Register running task for cancellation support
    _running_tasks[task_key] = asyncio.Event()

    # Enqueue task for serial processing per user
    loop = _get_loop()
    task_item = (text, session_id, tracker, status_msg_id, task_key)

    if user_id not in _user_queues:
        _user_queues[user_id] = asyncio.Queue()

    _user_queues[user_id].put_nowait(task_item)

    queue_size = _user_queues[user_id].qsize()
    if queue_size > 1:
        _patch_card(status_msg_id, f"⏳ 排队中，前面还有 {queue_size - 1} 个任务...")

    # Start consumer for this user if not already running
    if user_id not in _user_workers:
        _user_workers.add(user_id)
        asyncio.run_coroutine_threadsafe(_user_consumer(user_id), loop)


# ── Per-user task consumer ────────────────────────────────────────────

async def _user_consumer(user_id: str):
    """Serial consumer: process tasks one-by-one for a given user."""
    queue = _user_queues[user_id]
    try:
        while not queue.empty():
            text, session_id, tracker, status_msg_id, task_key = queue.get_nowait()
            await _process_task(text, session_id, tracker, status_msg_id, task_key)
    finally:
        _user_workers.discard(user_id)


async def _process_task(
    text: str, session_id: str, tracker: ProgressTracker,
    status_msg_id: str, task_key: str,
):
    """Execute a single task with timeout protection and cancellation support."""
    try:
        result = await _run_task_react(text, session_id, tracker)

        output = result.get("final_output", "处理完成，但没有生成输出。")
        status = result.get("status", "unknown")
        elapsed = time.monotonic() - tracker.start

        # Build final message: actions summary + result
        action_lines = tracker.actions[-6:] if tracker.actions else []
        header = "\n".join(action_lines) if action_lines else "✅ 直接回答"
        timing_total = f"⏱ 总耗时 {elapsed:.1f}s"

        if status == "completed":
            final_text = f"{header}\n{timing_total}\n\n---\n\n{output}"
        else:
            final_text = f"{header}\n{timing_total}\n\n⚠️ 状态: {status}\n\n{output}"

        # Register for feedback tracking
        _feedback_tasks[task_key] = {
            "user_input": text,
            "result": result,
            "session_id": session_id,
            "message_id": status_msg_id,
        }
        while len(_feedback_tasks) > _FEEDBACK_TASKS_MAX:
            _feedback_tasks.popitem(last=False)

        # ── Long reply splitting (#7) ──
        chunks = _split_long_text(final_text)
        if len(chunks) == 1:
            _patch_card(status_msg_id, final_text, with_feedback=True, task_key=task_key)
        else:
            # First chunk goes into the existing card
            _patch_card(status_msg_id, chunks[0] + f"\n\n*(共 {len(chunks)} 部分，第 1 部分)*",
                       with_feedback=True, task_key=task_key)
            # Remaining chunks as reply cards
            for i, chunk in enumerate(chunks[1:], 2):
                chunk_text = f"*(第 {i}/{len(chunks)} 部分)*\n\n{chunk}"
                _reply_card(status_msg_id, chunk_text)

        await _record_trace(text, session_id, result)

        # ── Metrics ──
        from common.logging_config import metrics
        metrics.record_task(
            "react", elapsed, status,
            tools_used=result.get("tools_used", []),
        )

        # ── Self-evolution: extract and store experience ──
        if status == "completed":
            await _evolve_experience(text, result)

        # ── Conversation summarization (#1) ──
        if _session_mgr:
            try:
                await _session_mgr.conv_store.maybe_summarize(session_id, llm=_react_params.get("llm"))
            except Exception as e:
                logger.debug("Summarization skipped: %s", e)

    except asyncio.CancelledError:
        elapsed = time.monotonic() - tracker.start
        logger.info("Task cancelled by user after %.0fs: %.80s...", elapsed, text)
        _patch_card(status_msg_id, f"⏹ 任务已取消（{elapsed:.1f}s）")

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - tracker.start
        logger.error("Task timed out after %.0fs: %.80s...", elapsed, text)
        _patch_card(status_msg_id,
                    f"⏰ 任务超时（{elapsed:.0f}s，上限 {_TASK_TIMEOUT}s）\n\n请简化任务后重试。",
                    with_retry=True, task_key=task_key, retry_text=text)
        # Store for retry
        _feedback_tasks[task_key] = {"user_input": text, "session_id": session_id,
                                      "message_id": status_msg_id, "result": {}}

    except Exception as e:
        logger.exception("Task execution failed")
        elapsed = time.monotonic() - tracker.start
        _patch_card(status_msg_id, f"❌ 处理失败 ({elapsed:.1f}s)\n\n{e}",
                    with_retry=True, task_key=task_key, retry_text=text)
        # Store for retry
        _feedback_tasks[task_key] = {"user_input": text, "session_id": session_id,
                                      "message_id": status_msg_id, "result": {}}

    finally:
        # Clean up running task
        _running_tasks.pop(task_key, None)


# ── Self-evolution: extract experience from completed tasks ───────────

async def _evolve_experience(user_input: str, result: dict[str, Any]):
    """Extract and store experience from a completed task (fire-and-forget)."""
    qdrant = _react_params.get("qdrant")
    llm = _react_params.get("llm")
    collection_name = _react_params.get("collection_name", "case_library")

    if qdrant is None or llm is None:
        return

    try:
        from common.vector import text_to_embedding_async
        from evolution.experience import extract_experience, store_experience

        experience = await extract_experience(result, llm=llm)
        if experience:
            embedding = await text_to_embedding_async(user_input)
            import uuid
            task_id = str(uuid.uuid4())
            await store_experience(qdrant, collection_name, experience, embedding, task_id)
            logger.info("Experience stored for Feishu task: %.50s...", user_input)
    except Exception as e:
        logger.warning("Experience evolution failed: %s", e)


# ── Card action callback (feedback buttons) ──────────────────────────

def _make_toast(type_: str, content: str):
    """Build a P2CardActionTriggerResponse with a toast."""
    from lark_oapi.event.callback.model.p2_card_action_trigger import (
        P2CardActionTriggerResponse, CallBackToast,
    )
    resp = P2CardActionTriggerResponse()
    toast = CallBackToast()
    toast.type = type_
    toast.content = content
    resp.toast = toast
    return resp


def _on_card_action(data):
    """Handle card action callback (👍👎 and cancel buttons).

    Called by Feishu SDK when user clicks an interactive card button.
    Returns a P2CardActionTriggerResponse with toast.
    """
    try:
        action = data.event.action
        value = action.value or {}
        action_type = value.get("action", "")
        task_key = value.get("task_key", "")
        user_id = data.event.operator.user_id or "unknown"

        if action_type == "cancel":
            # ── Cancel running task ──
            cancel_event = _running_tasks.get(task_key)
            if cancel_event:
                cancel_event.set()
                logger.info("Cancel requested by user %s for task %s", user_id, task_key)
                return _make_toast("info", "⏹ 正在取消任务...")
            return _make_toast("info", "任务已完成或不存在")

        if action_type == "retry":
            # ── Retry failed task ──
            retry_text = value.get("text", "")
            if not retry_text:
                return _make_toast("info", "重试信息丢失")
            task_info = _feedback_tasks.pop(task_key, None)
            session_id = task_info["session_id"] if task_info else f"feishu_{user_id}_retry"
            logger.info("Retry requested by user %s: %.80s...", user_id, retry_text)

            # Re-enqueue the task
            loop = _get_loop()
            # Update the existing card to show retrying
            msg_id = task_info["message_id"] if task_info else None
            if msg_id:
                _patch_card(msg_id, "🔄 **重试中**...")
                new_task_key = uuid.uuid4().hex[:12]
                settings = get_settings()
                tracker = ProgressTracker(msg_id, task_key=new_task_key,
                                         max_rounds=settings.react.max_rounds)
                _running_tasks[new_task_key] = asyncio.Event()
                task_item = (retry_text, session_id, tracker, msg_id, new_task_key)
                if user_id not in _user_queues:
                    _user_queues[user_id] = asyncio.Queue()
                _user_queues[user_id].put_nowait(task_item)
                if user_id not in _user_workers:
                    _user_workers.add(user_id)
                    asyncio.run_coroutine_threadsafe(_user_consumer(user_id), loop)
            return _make_toast("info", "🔄 正在重试...")

        if action_type == "feedback":
            # ── Feedback ──
            rating = value.get("rating", 0)

            if not task_key or task_key not in _feedback_tasks:
                logger.debug("Feedback for unknown task_key: %s", task_key)
                return _make_toast("info", "该任务反馈已过期")

            task_info = _feedback_tasks.pop(task_key)
            emoji = "👍" if rating >= 4 else "👎"
            logger.info("Feedback received: %s (rating=%d) from user %s for %.50s...",
                         emoji, rating, user_id, task_info["user_input"])

            # Fire-and-forget: process feedback asynchronously
            loop = _get_loop()
            asyncio.run_coroutine_threadsafe(
                _handle_feedback(rating, task_info, user_id), loop,
            )

            return _make_toast("success", f"感谢反馈 {emoji}")

        return None

    except Exception as e:
        logger.warning("Card action handling failed: %s", e)
        return None


async def _handle_feedback(rating: int, task_info: dict, user_id: str):
    """Process feedback: store negative experience for low ratings, boost for high."""
    qdrant = _react_params.get("qdrant")
    llm = _react_params.get("llm")
    collection_name = _react_params.get("collection_name", "case_library")

    try:
        from common.vector import text_to_embedding_async

        user_input = task_info["user_input"]
        result = task_info["result"]

        if rating <= 2 and llm is not None and qdrant is not None:
            # Low rating: store negative experience
            from evolution.experience import extract_experience, store_experience

            experience = await extract_experience(result, llm=llm)
            if experience:
                # Mark as negative experience
                experience["problem_pattern"] = f"[NEGATIVE] {experience.get('problem_pattern', '')}"
                experience["tips"] = f"用户反馈不满意。{experience.get('tips', '')}"
                embedding = await text_to_embedding_async(user_input)
                task_id = f"feedback_{uuid.uuid4().hex[:8]}"
                await store_experience(qdrant, collection_name, experience, embedding, task_id)
                logger.info("Negative experience stored from feedback for: %.50s...", user_input)

        elif rating >= 4 and qdrant is not None:
            # High rating: log positive feedback (experience already stored in _evolve_experience)
            logger.info("Positive feedback for: %.50s...", user_input)

    except Exception as e:
        logger.warning("Feedback processing failed: %s", e)


# ── Entry point ──────────────────────────────────────────────────────

def main():
    from common.logging_config import setup_logging

    settings = get_settings()
    # Use JSON logging in production (when debug=False)
    setup_logging(json_format=not settings.debug)

    if not settings.feishu.app_id or not settings.feishu.app_secret:
        logger.error("Feishu app_id and app_secret must be set in config/settings.yaml")
        sys.exit(1)

    logger.info("Initializing mul-agent resources...")
    _init_resources()

    # Build event handler (messages + card actions)
    handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(_on_message)
        .register_p2_card_action_trigger(_on_card_action)
        .build()
    )

    # WebSocket long-connection client
    ws_client = lark.ws.Client(
        app_id=settings.feishu.app_id,
        app_secret=settings.feishu.app_secret,
        event_handler=handler,
        log_level=lark.LogLevel.INFO,
    )

    logger.info("Feishu bot starting (WebSocket long-connection)...")
    logger.info("Send a message to the bot in Feishu to test.")

    # Handle graceful shutdown
    def _shutdown(signum, frame):
        logger.info("Shutting down...")
        if _loop and not _loop.is_closed():
            _loop.call_soon_threadsafe(_loop.stop)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # This blocks — maintains the WebSocket connection
    ws_client.start()


if __name__ == "__main__":
    main()
