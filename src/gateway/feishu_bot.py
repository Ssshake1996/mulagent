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

from agents.adapter import AdapterFactory
from agents.registry import load_registry
from common.config import get_settings
from common.db import create_session_factory
from common.llm import LLMManager
from common.vector import ensure_collection, get_qdrant_client
from evolution.trace import record_task_trace
from graph.orchestrator import build_graph

logger = logging.getLogger("feishu_bot")

# ── Global resources (initialized once) ──────────────────────────────
_graph_params: dict[str, Any] = {}  # params for build_graph(), minus on_subtask_progress
_lark_client: lark.Client | None = None
_db_session_factory = None

# Session management: user_id → session_id
_user_sessions: dict[str, str] = {}

# Message dedup: prevent re-processing retried messages
_seen_message_ids: set[str] = set()
_SEEN_MAX = 1000  # cap to avoid unbounded growth

# Per-user task queues: serialize tasks per user to avoid rate-limit blowout
_user_queues: dict[str, asyncio.Queue] = {}
_user_workers: set[str] = set()  # users that already have a running consumer
_TASK_TIMEOUT = 300  # 5 minutes global timeout per task


def _get_or_create_session(user_id: str) -> str:
    """Get current session for user, or create a new one."""
    if user_id not in _user_sessions:
        _user_sessions[user_id] = f"feishu_{user_id}_{uuid.uuid4().hex[:8]}"
    return _user_sessions[user_id]


def _new_session(user_id: str) -> str:
    """Force create a new session for user."""
    session_id = f"feishu_{user_id}_{uuid.uuid4().hex[:8]}"
    _user_sessions[user_id] = session_id
    return session_id

# Node display names
_NODE_LABELS: dict[str, str] = {
    "dispatch": "意图识别",
    "plan": "任务分解",
    "execute": "Agent 执行",
    "quality_check": "质量检查",
}
_NODE_ORDER = ["dispatch", "plan", "execute", "quality_check"]


def _init_resources():
    """Initialize all shared resources (called once at startup)."""
    global _graph_params, _lark_client, _db_session_factory

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

    # Agent registry + adapter
    registry = load_registry()
    use_openclaw = settings.openclaw.enabled if settings.openclaw else False
    openclaw_timeout = settings.openclaw.timeout if settings.openclaw else 120
    factory = AdapterFactory(
        llm_client=default_llm,
        use_openclaw=use_openclaw,
        openclaw_timeout=openclaw_timeout,
    )

    llm_light = llm_manager.get_light() if llm_manager else None

    # Save graph build params (graph is built per-request to support per-request callbacks)
    _graph_params = dict(
        registry=registry,
        adapter_factory=factory,
        llm=default_llm,
        llm_light=llm_light,
        qdrant=qdrant,
        collection_name=collection_name,
    )

    # Lark API client (for sending replies)
    _lark_client = (
        lark.Client.builder()
        .app_id(settings.feishu.app_id)
        .app_secret(settings.feishu.app_secret)
        .log_level(lark.LogLevel.WARNING)
        .build()
    )

    logger.info(
        "Resources initialized: LLM=%s, OpenClaw=%s, Qdrant=ready",
        llm_manager.default_id,
        use_openclaw,
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

def _build_card(text: str) -> str:
    """Build a Feishu interactive card JSON with markdown content."""
    card = {
        "elements": [
            {
                "tag": "markdown",
                "content": text,
            }
        ],
    }
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


def _patch_card(message_id: str, text: str):
    """Update an existing card message in-place (飞书 patch 仅支持卡片)."""
    body = PatchMessageRequestBody.builder() \
        .content(_build_card(text)) \
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
    """Tracks pipeline + subtask progress, renders status line, patches Feishu card."""

    def __init__(self, status_msg_id: str):
        self.status_msg_id = status_msg_id
        self.start = time.monotonic()
        self.completed_nodes: dict[str, int] = {}
        self.current_node: str | None = "dispatch"
        # Subtask tracking
        self.subtask_lines: list[str] = []  # rendered subtask progress lines
        self.current_subtask: str | None = None

    def render(self) -> str:
        lines = []
        for node in _NODE_ORDER:
            label = _NODE_LABELS.get(node, node)
            if node in self.completed_nodes:
                ms = self.completed_nodes[node]
                lines.append(f"✅ {label} {ms/1000:.1f}s")
            elif node == self.current_node:
                lines.append(f"🔄 **{label}**...")
                # Show subtask detail under execute
                if node == "execute" and self.subtask_lines:
                    for sl in self.subtask_lines:
                        lines.append(f"    {sl}")
            else:
                lines.append(f"⬜ {label}")

        elapsed = time.monotonic() - self.start
        lines.append(f"\n⏱ 总耗时 {elapsed:.1f}s")
        return "\n".join(lines)

    def update_card(self):
        _patch_card(self.status_msg_id, self.render())

    async def on_subtask_progress(self, name: str, status: str, index: int, total: int):
        """Callback from execute_node for each subtask start/done."""
        if status == "running":
            self.subtask_lines = [
                sl for sl in self.subtask_lines if not sl.startswith("🔄")
            ]
            self.subtask_lines.append(f"🔄 [{index}/{total}] {name}")
            self.current_subtask = name
        elif status == "done":
            self.subtask_lines = [
                sl.replace(f"🔄 [{index}/{total}] {name}", f"✅ [{index}/{total}] {name}")
                if f"🔄 [{index}/{total}] {name}" in sl else sl
                for sl in self.subtask_lines
            ]
            # If no replacement happened (e.g. name mismatch), just mark done
            done_line = f"✅ [{index}/{total}] {name}"
            if done_line not in self.subtask_lines:
                self.subtask_lines = [
                    sl for sl in self.subtask_lines if not sl.startswith("🔄")
                ]
                self.subtask_lines.append(done_line)
            self.current_subtask = None

        self.update_card()


# ── Streaming task execution ─────────────────────────────────────────

async def _run_task_streaming(user_input: str, session_id: str, tracker: ProgressTracker):
    """Execute task with streaming, patching the status message at each step."""
    # Build graph with subtask callback
    graph = build_graph(**_graph_params, on_subtask_progress=tracker.on_subtask_progress)
    compiled = graph.compile()

    final_result: dict[str, Any] = {}

    async for event in compiled.astream(
        {"user_input": user_input, "session_id": session_id},
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            # Collect timing data
            timing_data = node_output.get("_timing", {})
            for tn, tv in timing_data.items():
                tracker.completed_nodes[tn] = tv

            # Determine what's running next
            status = node_output.get("status", "")
            if status == "executing":
                tracker.current_node = "execute"
            elif status == "quality_check":
                tracker.current_node = "quality_check"
            elif node_name in _NODE_ORDER:
                if node_name not in tracker.completed_nodes:
                    elapsed_ms = int((time.monotonic() - tracker.start) * 1000)
                    tracker.completed_nodes[node_name] = elapsed_ms
                idx = _NODE_ORDER.index(node_name)
                tracker.current_node = _NODE_ORDER[idx + 1] if idx + 1 < len(_NODE_ORDER) else None

            final_result.update(node_output)
            tracker.update_card()

    return final_result


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
                intent=result.get("intent", "general"),
                dag_plan=result.get("subtasks"),
                subtask_results=result.get("subtask_results", {}),
                final_output=result.get("final_output", ""),
                status=result.get("status", "unknown"),
                subtasks=result.get("subtasks", []),
            )
    except Exception as e:
        logger.warning("Trace recording failed: %s", e)


# ── Message handler ──────────────────────────────────────────────────

def _extract_text(data) -> str | None:
    """Extract plain text from a Feishu message event."""
    try:
        msg = data.event.message
        if msg.message_type != "text":
            return None
        content = json.loads(msg.content)
        return content.get("text", "").strip()
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        logger.warning("Failed to extract text: %s", e)
        return None


def _extract_user_id(data) -> str:
    """Extract sender user_id from a Feishu message event."""
    try:
        return data.event.sender.sender_id.user_id or "unknown"
    except (AttributeError, Exception):
        return "unknown"


_HELP_TEXT = """\
**可用命令**

| 命令 | 说明 |
|------|------|
| `/new` | 开启新会话（清除上下文） |
| `/help` | 显示本帮助 |

直接发送文字即可下发任务。"""


def _on_message(data) -> None:
    """Handle incoming Feishu message (P2 event handler, single arg)."""
    text = _extract_text(data)
    if not text:
        return

    message_id = data.event.message.message_id

    # ── Dedup: skip already-processed messages ──
    if message_id in _seen_message_ids:
        logger.debug("Duplicate message %s, skipping", message_id)
        return
    if len(_seen_message_ids) >= _SEEN_MAX:
        # Evict oldest half to cap memory
        to_remove = list(_seen_message_ids)[:_SEEN_MAX // 2]
        for mid in to_remove:
            _seen_message_ids.discard(mid)
    _seen_message_ids.add(message_id)

    user_id = _extract_user_id(data)
    cmd = text.strip().lower()

    # Handle /help command
    if cmd in ("/help", "/h"):
        _reply_card(message_id, _HELP_TEXT)
        return

    # Handle /new command — create new session and reply confirmation
    if cmd in ("/new", "/new session"):
        session_id = _new_session(user_id)
        _reply_card(message_id, f"✅ 新会话已创建\n\nSession: `{session_id}`")
        logger.info("New session for user %s: %s", user_id, session_id)
        return

    session_id = _get_or_create_session(user_id)
    logger.info("Received from Feishu [%s]: %.80s...", session_id, text)

    # Send initial status card, get its message_id for patching
    tracker = ProgressTracker("__placeholder__")
    initial_text = tracker.render()
    status_msg_id = _reply_card(message_id, initial_text)
    if not status_msg_id:
        logger.error("Failed to send initial status message")
        return
    tracker.status_msg_id = status_msg_id

    # Enqueue task for serial processing per user
    loop = _get_loop()
    task_item = (text, session_id, tracker, status_msg_id)

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
            text, session_id, tracker, status_msg_id = queue.get_nowait()
            await _process_task(text, session_id, tracker, status_msg_id)
    finally:
        _user_workers.discard(user_id)


async def _process_task(text: str, session_id: str, tracker: ProgressTracker, status_msg_id: str):
    """Execute a single task with timeout protection."""
    try:
        result = await asyncio.wait_for(
            _run_task_streaming(text, session_id, tracker),
            timeout=_TASK_TIMEOUT,
        )

        output = result.get("final_output", "处理完成，但没有生成输出。")
        status = result.get("status", "unknown")
        elapsed = time.monotonic() - tracker.start

        # Build final message: all nodes done + result
        done_lines = []
        for node in _NODE_ORDER:
            label = _NODE_LABELS.get(node, node)
            if node in tracker.completed_nodes:
                done_lines.append(f"✅ {label} {tracker.completed_nodes[node]/1000:.1f}s")
            else:
                done_lines.append(f"✅ {label}")

        header = "\n".join(done_lines)
        timing_total = f"⏱ 总耗时 {elapsed:.1f}s"

        if status == "completed":
            final_text = f"{header}\n{timing_total}\n\n---\n\n{output}"
        else:
            final_text = f"{header}\n{timing_total}\n\n⚠️ 状态: {status}\n\n{output}"

        _patch_card(status_msg_id, final_text)
        await _record_trace(text, session_id, result)

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - tracker.start
        logger.error("Task timed out after %.0fs: %.80s...", elapsed, text)
        _patch_card(status_msg_id, f"⏰ 任务超时（{elapsed:.0f}s，上限 {_TASK_TIMEOUT}s）\n\n请简化任务后重试。")

    except Exception as e:
        logger.exception("Task execution failed")
        elapsed = time.monotonic() - tracker.start
        _patch_card(status_msg_id, f"❌ 处理失败 ({elapsed:.1f}s)\n\n{e}")


# ── Entry point ──────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    settings = get_settings()

    if not settings.feishu.app_id or not settings.feishu.app_secret:
        logger.error("Feishu app_id and app_secret must be set in config/settings.yaml")
        sys.exit(1)

    logger.info("Initializing mul-agent resources...")
    _init_resources()

    # Build event handler
    handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(_on_message)
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
