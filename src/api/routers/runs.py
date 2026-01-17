from __future__ import annotations

import math
import json
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.api.errors import APIError
from src.api.pagination import Cursor, CursorError, decode_cursor, encode_cursor
from src.runtime.reasoningbank_jobs import enqueue_rb_learn_job
from src.storage.sqlite_store import SQLiteStore
from src.tools.pubchem import resolve_pubchem


router = APIRouter()


class FeedbackProductInput(BaseModel):
    product_id: str = Field(min_length=1)
    value: float


class UpsertRunFeedbackRequest(BaseModel):
    score: float | None = Field(default=None)
    pros: str = Field(default="")
    cons: str = Field(default="")
    other: str = Field(default="")
    products: list[FeedbackProductInput] = Field(default_factory=list)
    schema_version: int = Field(default=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class ResolveModifierChecksRequest(BaseModel):
    force: bool = Field(default=False, description="Force recompute even if cached in trace events.")


class ModifierCheckItem(BaseModel):
    query: str
    normalized_query: str
    status: str
    cid: int | None = None
    canonical_smiles: str | None = None
    inchikey: str | None = None
    has_cooh: bool | None = None
    error: str | None = None


class ResolveModifierChecksResponse(BaseModel):
    run_id: str
    items: list[ModifierCheckItem]


@router.get("/runs/{run_id}/modifier_checks")
def get_run_modifier_checks(run_id: str) -> ResolveModifierChecksResponse:
    """Read cached modifier checks for a run (no recompute).

    WebUI should prefer this endpoint to avoid triggering external PubChem requests on page load.
    Use POST /modifier_checks to compute (best-effort) and cache the result.
    """
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")

        cached = store.get_latest_event(run_id=run_id, event_type="modifier_checks")
        if cached is None:
            raise APIError(status_code=404, code="not_found", message="Modifier checks not found.")

        try:
            payload = json.loads(str(cached["payload_json"]))
            items_any = payload.get("items") if isinstance(payload, dict) else None
            items = items_any if isinstance(items_any, list) else []
            return ResolveModifierChecksResponse(
                run_id=run_id,
                items=[ModifierCheckItem(**it) for it in items if isinstance(it, dict)],
            )
        except Exception as e:
            # Cached payload is invalid; treat as missing so the caller can recompute via POST.
            raise APIError(status_code=404, code="not_found", message="Modifier checks not found.") from e
    finally:
        store.close()


@router.get("/runs")
def list_runs(
    batch_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    status: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        cursor_obj: Cursor | None = None
        if cursor:
            try:
                cursor_obj = decode_cursor(cursor)
            except CursorError as e:
                raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e

        page = store.list_runs_page(
            batch_id=(batch_id.strip() if batch_id else None),
            limit=int(limit),
            cursor=(cursor_obj.created_at, cursor_obj.item_id) if cursor_obj is not None else None,
            statuses=status or None,
        )
        next_cursor = page.get("next_cursor")
        if next_cursor is not None:
            created_at, run_id = next_cursor
            page["next_cursor"] = encode_cursor(Cursor(created_at=float(created_at), item_id=str(run_id)))
        return page
    finally:
        store.close()


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        row = store.get_run(run_id=run_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")
        return {
            "run": {
                "run_id": row["run_id"],
                "batch_id": row["batch_id"],
                "run_index": int(row["run_index"]),
                "created_at": float(row["created_at"]),
                "started_at": float(row["started_at"]) if row["started_at"] is not None else None,
                "ended_at": float(row["ended_at"]) if row["ended_at"] is not None else None,
                "status": row["status"],
                "error": row["error"],
            }
        }
    finally:
        store.close()


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        row = store.get_run(run_id=run_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")
        reason = ""
        if body:
            reason = str(body.get("reason") or "").strip()
        cancel_id = store.request_cancel(target_type="run", target_id=run_id, reason=reason or None)
        return {"cancel_id": cancel_id, "status": "requested"}
    finally:
        store.close()


@router.get("/runs/{run_id}/output")
def get_run_output(run_id: str) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        row = store.get_latest_event(run_id=run_id, event_type="final_output")
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Run output not found.")
        payload = json.loads(row["payload_json"])
        return {
            "recipes_json": payload.get("recipes_json") or {},
            "citations": payload.get("citations") or {},
            "memory_ids": payload.get("memory_ids") or [],
        }
    finally:
        store.close()


@router.post("/runs/{run_id}/modifier_checks")
def resolve_run_modifier_checks(run_id: str, body: ResolveModifierChecksRequest | None = None) -> ResolveModifierChecksResponse:
    """Resolve small_molecule_modifier via PubChem and compute a best-effort COOH signal.

    This is intended for WebUI display / manual review. It must NOT block the run itself.
    Results are cached as a trace event (best-effort) to avoid repeated external requests.
    """
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")

        force = bool(body.force) if body is not None else False
        cached = store.get_latest_event(run_id=run_id, event_type="modifier_checks")
        if cached is not None and not force:
            try:
                payload = json.loads(str(cached["payload_json"]))
                items_any = payload.get("items") if isinstance(payload, dict) else None
                items = items_any if isinstance(items_any, list) else []
                return ResolveModifierChecksResponse(
                    run_id=run_id,
                    items=[ModifierCheckItem(**it) for it in items if isinstance(it, dict)],
                )
            except Exception:
                # Fall through to recompute.
                pass

        out_row = store.get_latest_event(run_id=run_id, event_type="final_output")
        if out_row is None:
            raise APIError(status_code=404, code="not_found", message="Run output not found.")
        payload = json.loads(str(out_row["payload_json"]))
        recipes = (
            ((payload.get("recipes_json") or {}) if isinstance(payload.get("recipes_json"), dict) else {}).get("recipes")
        )
        recipes_list = recipes if isinstance(recipes, list) else []
        modifiers: list[str] = []
        seen: set[str] = set()
        for r in recipes_list:
            if not isinstance(r, dict):
                continue
            mod = str(r.get("small_molecule_modifier") or "").strip()
            if not mod or mod in seen:
                continue
            seen.add(mod)
            modifiers.append(mod)

        resolved_items: list[ModifierCheckItem] = []
        for mod in modifiers:
            res = resolve_pubchem(mod)
            resolved_items.append(
                ModifierCheckItem(
                    query=res.query,
                    normalized_query=res.normalized_query,
                    status=res.status,
                    cid=res.cid,
                    canonical_smiles=res.canonical_smiles,
                    inchikey=res.inchikey,
                    has_cooh=res.has_cooh,
                    error=res.error,
                )
            )

        # Best-effort cache: never fail the endpoint due to trace write issues.
        # (UI display must not be blocked by caching failures.)
        try:
            store.append_event(
                run_id,
                "modifier_checks",
                {
                    # sqlite3.Row behaves like a mapping but does not implement .get()
                    "ts": float(out_row["created_at"]) if out_row["created_at"] is not None else None,
                    "items": [it.model_dump(mode="python") for it in resolved_items],
                },
            )
        except Exception:
            pass

        return ResolveModifierChecksResponse(run_id=run_id, items=resolved_items)
    finally:
        store.close()


@router.get("/runs/{run_id}/feedback")
def get_run_feedback(run_id: str) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")
        payload = store.get_feedback_for_run(run_id=run_id)
        if payload is None:
            raise APIError(status_code=404, code="not_found", message="Feedback not found.")
        return payload
    finally:
        store.close()


@router.put("/runs/{run_id}/feedback")
def upsert_run_feedback(run_id: str, body: UpsertRunFeedbackRequest) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")

        if body.score is not None and not math.isfinite(float(body.score)):
            raise APIError(status_code=400, code="invalid_argument", message="score must be a finite number.")

        try:
            store.upsert_feedback_for_run(
                run_id=run_id,
                score=float(body.score) if body.score is not None else None,
                pros=body.pros,
                cons=body.cons,
                other=body.other,
                products=[p.model_dump(mode="python") for p in body.products],
                schema_version=int(body.schema_version),
                extra=body.extra,
            )
        except ValueError as e:
            raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e

        payload = store.get_feedback_for_run(run_id=run_id)
        assert payload is not None

        # Recommended by spec: saving/updating feedback should trigger strict RB re-learn (async).
        # The RB worker will:
        # - rollback the previous applied delta for this run (if any),
        # - learn again using the latest feedback,
        # - record a new delta.
        try:
            enqueue_rb_learn_job(store, run_id=run_id)
        except Exception:
            # Best-effort: feedback save must not fail just because RB enqueue failed.
            # Failure will be visible via missing rb_job or via worker logs.
            pass

        return payload
    finally:
        store.close()


@router.get("/runs/{run_id}/events")
def list_run_events(
    run_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    event_type: list[str] | None = Query(default=None),
    include_payload: bool = Query(default=False),
    since: float | None = Query(default=None),
    until: float | None = Query(default=None),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")

        cursor_obj: Cursor | None = None
        if cursor:
            try:
                cursor_obj = decode_cursor(cursor)
            except CursorError as e:
                raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e

        page = store.list_events_page(
            run_id=run_id,
            limit=int(limit),
            cursor=(cursor_obj.created_at, cursor_obj.item_id) if cursor_obj is not None else None,
            event_types=event_type or None,
            include_payload=bool(include_payload),
            since=since,
            until=until,
        )
        next_cursor = page.get("next_cursor")
        if next_cursor is not None:
            created_at, event_id = next_cursor
            page["next_cursor"] = encode_cursor(Cursor(created_at=float(created_at), item_id=str(event_id)))
        return page
    finally:
        store.close()


@router.get("/runs/{run_id}/events/{event_id}")
def get_run_event(run_id: str, event_id: str) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        row = store.get_event(run_id=run_id, event_id=event_id)
        if row is None:
            raise APIError(status_code=404, code="not_found", message="Event not found.")
        return {
            "event_id": row["event_id"],
            "run_id": row["run_id"],
            "created_at": float(row["created_at"]),
            "event_type": row["event_type"],
            "payload": json.loads(row["payload_json"]),
        }
    finally:
        store.close()


@router.get("/runs/{run_id}/evidence")
def list_run_evidence(
    run_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    include_content: bool = Query(default=False),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")

        cursor_obj: Cursor | None = None
        if cursor:
            try:
                cursor_obj = decode_cursor(cursor)
            except CursorError as e:
                raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e

        page = store.list_evidence_page(
            run_id=run_id,
            limit=int(limit),
            cursor=(cursor_obj.created_at, cursor_obj.item_id) if cursor_obj is not None else None,
            include_content=bool(include_content),
        )
        next_cursor = page.get("next_cursor")
        if next_cursor is not None:
            created_at, alias = next_cursor
            page["next_cursor"] = encode_cursor(Cursor(created_at=float(created_at), item_id=str(alias)))
        return page
    finally:
        store.close()


@router.get("/runs/{run_id}/evidence/{alias}")
def get_run_evidence(run_id: str, alias: str) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        if store.get_run(run_id=run_id) is None:
            raise APIError(status_code=404, code="not_found", message="Run not found.")

        cleaned = (alias or "").strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1].strip()

        item = store.get_evidence_item(run_id=run_id, alias=cleaned)
        if item is None:
            raise APIError(status_code=404, code="not_found", message="Evidence not found.")
        return item
    finally:
        store.close()
