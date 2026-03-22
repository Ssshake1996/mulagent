"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

import structlog

from common.config import get_settings
from common.db import create_session_factory
from common.llm import LLMManager
from common.vector import ensure_collection, get_qdrant_client
from gateway.routes import init_dependencies, router
from gateway.streaming import init_stream_dependencies, stream_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    settings = get_settings()

    # Initialize multi-model LLM manager
    llm_manager = LLMManager()
    default_llm = llm_manager.default

    if default_llm is not None:
        logger.info(
            "LLM initialized",
            default=llm_manager.default_id,
            available=[m["id"] for m in llm_manager.list_models()],
        )
    else:
        logger.info("No LLM configured, running in mock mode")

    # Initialize database session factory
    db_session_factory = None
    if settings.database and settings.database.url:
        try:
            db_session_factory = create_session_factory()
            logger.info("Database connected", url=settings.database.url.split("@")[-1])
        except Exception as e:
            logger.warning("Database unavailable, traces disabled", error=str(e))

    # Initialize Qdrant (in-memory fallback if remote unavailable)
    qdrant = get_qdrant_client()
    collection_name = settings.qdrant.collection_name if settings.qdrant else "case_library"
    ensure_collection(qdrant, collection_name)
    logger.info("Qdrant case library ready", collection=collection_name)

    # Inject into route modules
    init_dependencies(
        llm=default_llm, llm_manager=llm_manager,
        db_session_factory=db_session_factory, qdrant=qdrant, collection_name=collection_name,
    )
    init_stream_dependencies(
        llm=default_llm, llm_manager=llm_manager,
        db_session_factory=db_session_factory, qdrant=qdrant, collection_name=collection_name,
    )

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    app.include_router(stream_router, prefix="/api/v1")
    return app


app = create_app()
