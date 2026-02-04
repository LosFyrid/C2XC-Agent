from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header
from fastapi import Query
from pydantic import BaseModel, Field

from src.api.errors import APIError
from src.api.pagination import Cursor, CursorError, decode_cursor, encode_cursor
from src.config.load_config import default_config_path, load_app_config
from src.storage.sqlite_store import SQLiteStore


router = APIRouter()


class CreateBatchRequest(BaseModel):
    user_request: str = Field(default="", description="User request text. If empty, backend may choose a default.")
    n_runs: int = Field(default=1, ge=1)
    recipes_per_run: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.7)
    dry_run: bool = Field(default=False)
    overrides: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=1)
    extra: dict[str, Any] = Field(default_factory=dict)


@router.get("/batches")
def list_batches(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    status: list[str] | None = Query(default=None),
    include_hidden: bool = Query(default=False, description="Include soft-hidden batches."),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        cursor_obj: Cursor | None = None
        if cursor:
            try:
                cursor_obj = decode_cursor(cursor)
            except CursorError as e:
                raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e

        page = store.list_batches_page(
            limit=int(limit),
            cursor=(cursor_obj.created_at, cursor_obj.item_id) if cursor_obj is not None else None,
            statuses=status or None,
            include_hidden=bool(include_hidden),
        )
        next_cursor = page.get("next_cursor")
        if next_cursor is not None:
            created_at, batch_id = next_cursor
            page["next_cursor"] = encode_cursor(Cursor(created_at=float(created_at), item_id=str(batch_id)))
        return page
    finally:
        store.close()


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        row = store.get_batch(batch_id=batch_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Batch not found.")
        config_snapshot = json.loads(str(row["config_json"]))
        return {
            "batch": {
                "batch_id": row["batch_id"],
                "created_at": float(row["created_at"]),
                "started_at": float(row["started_at"]) if row["started_at"] is not None else None,
                "ended_at": float(row["ended_at"]) if row["ended_at"] is not None else None,
                "status": row["status"],
                "user_request": row["user_request"],
                "n_runs": int(row["n_runs"]),
                "recipes_per_run": int(row["recipes_per_run"]),
                "config_snapshot": config_snapshot,
                "error": row["error"],
                "hidden_at": float(row["hidden_at"]) if row["hidden_at"] is not None else None,
            }
        }
    finally:
        store.close()


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        row = store.get_batch(batch_id=batch_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Batch not found.")
        reason = ""
        if body:
            reason = str(body.get("reason") or "").strip()
        cancel_id = store.request_cancel(target_type="batch", target_id=batch_id, reason=reason or None)
        return {"cancel_id": cancel_id, "status": "requested"}
    finally:
        store.close()


@router.post("/batches/{batch_id}/hide")
def hide_batch(batch_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Soft-hide a batch (and its runs) from list endpoints.

    This does NOT delete data and does NOT cancel running work.
    """
    store = SQLiteStore()
    try:
        row = store.get_batch(batch_id=batch_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Batch not found.")

        _reason = ""
        if body:
            _reason = str(body.get("reason") or "").strip()
        # Reason is currently ignored (could be logged in a future audit table).
        _ = _reason

        ts = store.hide_batch(batch_id=batch_id)
        return {"batch_id": batch_id, "hidden_at": ts, "status": "hidden"}
    finally:
        store.close()


@router.post("/batches/{batch_id}/unhide")
def unhide_batch(batch_id: str) -> dict[str, Any]:
    """Undo `hide_batch` (best-effort restore)."""
    store = SQLiteStore()
    try:
        row = store.get_batch(batch_id=batch_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Batch not found.")
        store.unhide_batch(batch_id=batch_id)
        return {"batch_id": batch_id, "status": "visible"}
    finally:
        store.close()


@router.post("/batches")
def create_batch(
    body: CreateBatchRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        cfg = load_app_config()
        if body.n_runs < 1 or body.n_runs > cfg.limits.n_runs_max:
            raise APIError(
                status_code=400,
                code="invalid_argument",
                message=f"n_runs must be in [1..{cfg.limits.n_runs_max}].",
                details={"n_runs": body.n_runs},
            )
        if body.recipes_per_run < 1 or body.recipes_per_run > cfg.limits.recipes_per_run_max:
            raise APIError(
                status_code=400,
                code="invalid_argument",
                message=f"recipes_per_run must be in [1..{cfg.limits.recipes_per_run_max}].",
                details={"recipes_per_run": body.recipes_per_run},
            )

        user_request = (body.user_request or "").strip()
        if not user_request:
            user_request = (
                "Generate catalyst recipes for photocatalytic CO2 reduction/coupling. "
                "Primary objective: high selectivity and high activity for ethylene (C2H4). "
                "System is M1M2–TiO2 / Zr-BTB with fixed BTB linker; small_molecule_modifier must contain -COOH."
            )

        # Config snapshot (stored in SQLite for replayability).
        overrides = dict(body.overrides or {})
        config_snapshot: dict[str, Any] = {
            "config_path": str(default_config_path()),
            "n_runs": int(body.n_runs),
            "recipes_per_run": int(body.recipes_per_run),
            "temperature": float(body.temperature),
            "dry_run": bool(body.dry_run),
            "kb_principles_dir": str(
                overrides.get("kb_principles_dir") or os.getenv("LIGHTRAG_KB_PRINCIPLES_DIR", "")
            ),
            "kb_modulation_dir": str(
                overrides.get("kb_modulation_dir") or os.getenv("LIGHTRAG_KB_MODULATION_DIR", "")
            ),
            "llm_model": str(overrides.get("llm_model") or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""),
            "openai_api_base": str(
                overrides.get("openai_api_base")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or ""
            ),
            # RB learn (extract/merge) can use a different chat model/base than the main run.
            # If omitted, worker will fall back to (llm_model, openai_api_base).
            "rb_llm_model": str(
                overrides.get("rb_llm_model")
                or os.getenv("C2XC_RB_LEARN_LLM_MODEL")
                or ""
            ),
            "rb_openai_api_base": str(
                overrides.get("rb_openai_api_base")
                or os.getenv("C2XC_RB_LEARN_OPENAI_API_BASE")
                or ""
            ),
            # Embeddings can be configured separately from chat (e.g. different gateways/models).
            # Store non-sensitive values only (no keys).
            "embedding_model": str(
                overrides.get("embedding_model")
                or os.getenv("C2XC_EMBEDDING_MODEL")
                or os.getenv("EMBEDDING_MODEL")
                or ""
            ),
            "embedding_api_base": str(
                overrides.get("embedding_api_base")
                or os.getenv("C2XC_EMBEDDING_API_BASE")
                or os.getenv("EMBEDDING_API_BASE")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or ""
            ),
            "embedding_dim": str(
                overrides.get("embedding_dim")
                or os.getenv("C2XC_EMBEDDING_DIM")
                or os.getenv("EMBEDDING_DIM")
                or ""
            ),
            "embedding_send_dimensions": str(
                overrides.get("embedding_send_dimensions")
                or os.getenv("C2XC_EMBEDDING_SEND_DIMENSIONS")
                or os.getenv("EMBEDDING_SEND_DIMENSIONS")
                or ""
            ),
            "overrides": overrides,
        }

        if not bool(body.dry_run):
            # Fail fast for obvious missing dependencies. (Worker will still record detailed errors.)
            missing: list[str] = []
            if not os.getenv("OPENAI_API_KEY", "").strip():
                missing.append("OPENAI_API_KEY")
            kb_principles_dir = str(config_snapshot.get("kb_principles_dir") or "").strip()
            kb_modulation_dir = str(config_snapshot.get("kb_modulation_dir") or "").strip()
            if not kb_principles_dir:
                missing.append("LIGHTRAG_KB_PRINCIPLES_DIR")
            if not kb_modulation_dir:
                missing.append("LIGHTRAG_KB_MODULATION_DIR")

            # Dependency imports: keep error messages explicit for UI users.
            try:
                import openai  # noqa: F401
            except Exception:
                missing.append("python:openai")
            try:
                import lightrag  # noqa: F401
            except Exception:
                missing.append("python:lightrag")

            # Basic filesystem sanity: prefer failing at request time rather than queueing doomed runs.
            if kb_principles_dir and not Path(kb_principles_dir).expanduser().exists():
                missing.append("path:LIGHTRAG_KB_PRINCIPLES_DIR")
            if kb_modulation_dir and not Path(kb_modulation_dir).expanduser().exists():
                missing.append("path:LIGHTRAG_KB_MODULATION_DIR")

            if missing:
                raise APIError(
                    status_code=503,
                    code="dependency_unavailable",
                    message="Missing required runtime configuration for normal runs.",
                    details={"missing": missing},
                )

        # Idempotency: hash the *raw* request body (not the derived defaults).
        req_obj = body.model_dump(mode="json")
        req_json = json.dumps(req_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        request_hash = hashlib.sha256(req_json.encode("utf-8")).hexdigest()

        with store.transaction(mode="IMMEDIATE"):
            if idempotency_key:
                existing = store.get_idempotency(str(idempotency_key))
                if existing is not None:
                    if str(existing["request_hash"]) != request_hash:
                        raise APIError(
                            status_code=409,
                            code="conflict",
                            message="Idempotency-Key was already used with a different request body.",
                        )
                    return json.loads(str(existing["response_json"]))

            batch = store.create_batch(
                user_request=user_request,
                n_runs=int(body.n_runs),
                recipes_per_run=int(body.recipes_per_run),
                config=config_snapshot,
                commit=False,
            )
            runs = [
                store.create_run(batch_id=batch.batch_id, run_index=i + 1, commit=False)
                for i in range(int(body.n_runs))
            ]

            response: dict[str, Any] = {
            "batch": {
                "batch_id": batch.batch_id,
                "created_at": batch.created_at,
                "status": batch.status,
                "user_request": batch.user_request,
                "n_runs": batch.n_runs,
                "recipes_per_run": batch.recipes_per_run,
                "started_at": None,
                "ended_at": None,
                "config_snapshot": config_snapshot,
                "error": None,
            },
            "runs": [
                {
                    "run_id": r.run_id,
                    "batch_id": r.batch_id,
                    "run_index": r.run_index,
                    "created_at": r.created_at,
                    "status": r.status,
                    "started_at": None,
                    "ended_at": None,
                    "error": None,
                }
                for r in runs
            ],
        }

            if idempotency_key:
                store.put_idempotency(
                    key=str(idempotency_key),
                    request_hash=request_hash,
                    response_json=json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                    commit=False,
                )

            return response
    finally:
        store.close()
