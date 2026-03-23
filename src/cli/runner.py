"""AgentRunner — unified task executor for CLI mode.

Wrapper around run_react() that initializes the **full** dependency stack
(LLM + PostgreSQL + Redis + Qdrant), identical to the API server / Feishu bot.

Run ``scripts/setup.sh`` to start the required infrastructure services
before launching the CLI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

from common.config import get_settings, load_settings, reload_settings
from common.db import create_session_factory
from common.llm import LLMManager
from common.vector import ensure_collection, get_qdrant_client, set_shared_llm
from gateway.adapter import SessionManager
from graph.conversation import ConversationStore
from graph.orchestrator import run_react

logger = logging.getLogger(__name__)


class AgentRunner:
    """Full-stack task executor for CLI / Desktop mode.

    Initializes the same dependency set as the API server:
    LLM + PostgreSQL + Redis + Qdrant.  Services that are unavailable
    degrade gracefully (traces disabled, checkpoint disabled, etc.)
    but the core ReAct loop always works.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        model_override: str | None = None,
    ):
        # ── 1. Settings ───────────────────────────────────────────
        if config_path:
            reload_settings()  # clear lru_cache
            self._settings = load_settings(config_path)
        else:
            self._settings = get_settings()

        # ── 2. LLM ───────────────────────────────────────────────
        self._llm_manager = LLMManager()
        self._model_id = model_override or self._llm_manager.default_id
        self._llm = self._llm_manager.get(self._model_id)
        if self._llm is None:
            logger.error("No LLM configured. Check config/settings.yaml → llm section.")
            sys.exit(1)

        # ── 3. PostgreSQL (optional — traces disabled if unavailable)
        self._db_session_factory = None
        if self._settings.database and self._settings.database.url:
            try:
                self._db_session_factory = create_session_factory()
                logger.info("Database connected: %s", self._settings.database.url.split("@")[-1])
            except Exception as e:
                logger.warning("Database unavailable, traces disabled: %s", e)

        # ── 4. Redis (optional — checkpoint & cache disabled if unavailable)
        #       Redis is lazy-initialized via common.redis_client.get_redis()
        #       on first use, so nothing to do here. Just verify connectivity.
        self._redis_ok = False
        try:
            import asyncio
            from common.redis_client import get_redis

            async def _ping():
                r = await get_redis()
                return r is not None

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Inside an existing event loop — skip sync ping, Redis
                # will be checked lazily on first use.
                pass
            else:
                self._redis_ok = asyncio.run(_ping())
        except Exception:
            pass

        if self._redis_ok:
            logger.info("Redis connected: %s", self._settings.redis.url)
        else:
            logger.warning("Redis unavailable, checkpoint & cache disabled")

        # ── 5. Qdrant (remote preferred, in-memory fallback) ─────
        self._qdrant = get_qdrant_client()  # remote first, falls back to in-memory
        collection = self._settings.qdrant.collection_name
        ensure_collection(self._qdrant, collection)

        # ── 6. Embedding fallback ────────────────────────────────
        set_shared_llm(self._llm)

        # ── 7. Session manager (shares data dir with Feishu) ─────
        self._session_mgr = SessionManager(ConversationStore())

        # ── 8. Frozen react params ───────────────────────────────
        self._react_params: dict[str, Any] = {
            "llm": self._llm,
            "qdrant": self._qdrant,
            "collection_name": collection,
            "timeout": self._settings.react.timeout,
        }

        logger.info(
            "AgentRunner ready: model=%s, pg=%s, redis=%s, qdrant=%s",
            self._model_id,
            "ok" if self._db_session_factory else "off",
            "ok" if self._redis_ok else "off",
            "remote" if not getattr(self._qdrant, '_local', True) else "in-memory",
        )

    # ── Public API ────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        session_id: str,
        on_progress: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Execute *user_input* via the ReAct loop.

        Returns the same result dict as ``run_react()``:
            {"final_output": str, "status": str, "intent": str, ...}
        """
        conv = self._session_mgr.conv_store

        # Load multi-turn context
        history = conv.get_history_for_prompt(session_id, max_turns=10)
        directives = conv.get_directives(session_id) or None
        conv.append_turn(session_id, "user", user_input)

        result = await run_react(
            user_input=user_input,
            on_progress=on_progress,
            conversation_history=history,
            session_directives=directives,
            **self._react_params,
        )

        # Persist assistant reply and directives
        output = result.get("final_output", "")
        conv.append_turn(session_id, "assistant", output)
        new_directives = result.get("directives", [])
        if new_directives:
            conv.save_directives(session_id, new_directives)

        # Record execution trace to PostgreSQL (if available)
        if self._db_session_factory:
            try:
                from evolution.trace import record_task_trace
                async with self._db_session_factory() as db_session:
                    await record_task_trace(
                        db_session,
                        session_id=session_id,
                        user_input=user_input,
                        intent=result.get("intent", "react"),
                        dag_plan=None,
                        subtask_results={},
                        final_output=output,
                        status=result.get("status", "unknown"),
                        subtasks=[],
                    )
            except Exception as e:
                logger.debug("Failed to record trace: %s", e)

        return result

    def switch_model(self, model_id: str) -> bool:
        """Hot-switch the LLM model. Returns False if model_id is invalid."""
        llm = self._llm_manager.get(model_id)
        if llm is None:
            return False
        self._llm = llm
        self._model_id = model_id
        self._react_params["llm"] = llm
        set_shared_llm(llm)
        return True

    # ── Accessors ─────────────────────────────────────────────────

    @property
    def session_manager(self) -> SessionManager:
        return self._session_mgr

    @property
    def llm_manager(self) -> LLMManager:
        return self._llm_manager

    @property
    def current_model(self) -> str:
        return self._model_id

    @property
    def db_session_factory(self):
        return self._db_session_factory
