from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.api.errors import APIError, api_error_handler, unhandled_error_handler, validation_error_handler
from src.runtime.worker import RunWorker
from src.storage.sqlite_store import SQLiteStore

from .routers.batches import router as batches_router
from .routers.catalog import router as catalog_router
from .routers.health import router as health_router
from .routers.memories import router as memories_router
from .routers.reasoningbank import router as reasoningbank_router
from .routers.runs import router as runs_router


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("C2XC_CORS_ORIGINS", "").strip()
    if not raw:
        # Safe local defaults: allow typical dev ports.
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        v = raw.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        # Reconcile stuck 'running' runs from a previous process.
        if _env_bool("C2XC_RECONCILE_ON_STARTUP", True):
            store = SQLiteStore()
            try:
                reconciled = store.reconcile_running_runs()
                app.state.reconciled_running_runs = int(reconciled)
            finally:
                store.close()
        else:
            app.state.reconciled_running_runs = 0

        # Start a single background worker (single-instance assumption).
        if _env_bool("C2XC_ENABLE_WORKER", True):
            worker = RunWorker()
            worker.start()
            app.state.run_worker = worker
        try:
            yield
        finally:
            worker = getattr(app.state, "run_worker", None)
            if worker is not None:
                worker.stop()

    app = FastAPI(title="C2XC-Agent API", version="0.3.0", lifespan=lifespan)

    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    # CORS (for WebUI in dev / local deployments).
    origins = _cors_origins_from_env()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api/v1", tags=["system"])
    app.include_router(batches_router, prefix="/api/v1", tags=["batches"])
    app.include_router(catalog_router, prefix="/api/v1", tags=["catalog"])
    app.include_router(memories_router, prefix="/api/v1", tags=["memories"])
    app.include_router(runs_router, prefix="/api/v1", tags=["runs"])
    app.include_router(reasoningbank_router, prefix="/api/v1", tags=["reasoningbank"])

    if _env_bool("C2XC_ENABLE_DEBUG_ENDPOINTS", False):
        @app.get("/api/v1/_debug/env", include_in_schema=False)
        def debug_env() -> dict[str, Any]:
            # Avoid leaking secrets: only show non-sensitive values.
            return {
                "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "",
                "LLM_MODEL": os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "",
                "C2XC_LLM_REASONING_EFFORT": os.getenv("C2XC_LLM_REASONING_EFFORT") or "",
                "C2XC_LLM_VERBOSITY": os.getenv("C2XC_LLM_VERBOSITY") or "",
                "C2XC_EMBEDDING_API_BASE": os.getenv("C2XC_EMBEDDING_API_BASE") or os.getenv("EMBEDDING_API_BASE") or "",
                "C2XC_EMBEDDING_MODEL": os.getenv("C2XC_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "",
                "C2XC_EMBEDDING_DIM": os.getenv("C2XC_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM") or "",
                "C2XC_EMBEDDING_SEND_DIMENSIONS": os.getenv("C2XC_EMBEDDING_SEND_DIMENSIONS") or os.getenv("EMBEDDING_SEND_DIMENSIONS") or "",
                "LIGHTRAG_KB_PRINCIPLES_DIR": os.getenv("LIGHTRAG_KB_PRINCIPLES_DIR", ""),
                "LIGHTRAG_KB_MODULATION_DIR": os.getenv("LIGHTRAG_KB_MODULATION_DIR", ""),
                "C2XC_SQLITE_PATH": os.getenv("C2XC_SQLITE_PATH", "data/app.db"),
                "C2XC_RB_CHROMA_DIR": os.getenv("C2XC_RB_CHROMA_DIR", "data/chroma"),
            }

    return app


app = create_app()
