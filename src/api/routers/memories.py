from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_reasoningbank_store
from src.api.errors import APIError
from src.api.pagination import Cursor, CursorError, decode_cursor, encode_cursor
from src.storage.reasoningbank_store import (
    MemoryItem,
    ReasoningBankError,
    ReasoningBankStore,
)
from src.storage.sqlite_store import SQLiteStore


router = APIRouter()


def _item_to_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "mem_id": item.mem_id,
        "status": item.status,
        "role": item.role,
        "type": item.type,
        "content": item.content,
        "source_run_id": item.source_run_id,
        "created_at": float(item.created_at),
        "updated_at": float(item.updated_at),
        "schema_version": int(item.schema_version),
        "extra": item.extra,
    }

def _ensure_rb_mem_index(store: SQLiteStore, rb: ReasoningBankStore) -> None:
    """Ensure SQLite rb_mem_index has data.

    - New DBs will naturally start empty.
    - Existing deployments upgrading from schema<5 will have an empty index even if Chroma has data.
    We do a one-time best-effort backfill by scanning Chroma metadata (no documents).
    """
    try:
        if store.count_rb_mem_index() > 0:
            return
    except Exception:
        # Table missing or schema not migrated; ignore.
        return

    # Best-effort backfill: may be slow once, but makes browse stable afterwards.
    items = rb.list_all(role=None, status=None, type=None, include_content=False)
    payloads: list[dict[str, Any]] = []
    for it in items:
        payloads.append(
            {
                "mem_id": it.mem_id,
                "created_at": float(it.created_at),
                "updated_at": float(it.updated_at),
                "status": it.status,
                "role": it.role,
                "type": it.type,
                "source_run_id": it.source_run_id,
                "schema_version": int(it.schema_version),
            }
        )
    store.upsert_rb_mem_index_many(payloads)


def _sync_rb_mem_index(store: SQLiteStore, item: MemoryItem) -> None:
    store.upsert_rb_mem_index(
        mem_id=item.mem_id,
        created_at=float(item.created_at),
        updated_at=float(item.updated_at),
        status=str(item.status),
        role=str(item.role),
        type=str(item.type),
        source_run_id=str(item.source_run_id or "") or None,
        schema_version=int(item.schema_version),
    )


class CreateMemoryRequest(BaseModel):
    role: str = Field(default="global")
    status: str = Field(default="active")
    type: str = Field(default="manual_note", description="Only manual_note can be created via API.")
    content: str = Field(min_length=1)
    schema_version: int = Field(default=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class PatchMemoryRequest(BaseModel):
    status: str | None = Field(default=None)
    role: str | None = Field(default=None)
    type: str | None = Field(default=None)
    content: str | None = Field(default=None)
    schema_version: int | None = Field(default=None)
    extra: dict[str, Any] | None = Field(default=None)


@router.get("/memories")
def list_memories(
    rb: ReasoningBankStore = Depends(get_reasoningbank_store),
    query: str | None = Query(default=None),
    role: list[str] | None = Query(default=None),
    status: list[str] | None = Query(default=None),
    type: list[str] | None = Query(default=None),  # noqa: A002
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
) -> dict[str, Any]:
    q = (query or "").strip()
    if q:
        if cursor:
            # Deliberate: do NOT support deep pagination for semantic search results.
            raise APIError(
                status_code=400,
                code="invalid_argument",
                message="Search pagination is not supported for query mode. Omit cursor and increase limit if needed.",
            )

        results = rb.query(query=q, n_results=int(limit), role=role, status=status, type=type)
        window = results[: int(limit)]

        items: list[dict[str, Any]] = []
        for r in window:
            item: MemoryItem = r["item"]
            distance = r.get("distance")
            d = _item_to_dict(item)
            if distance is not None:
                d["distance"] = float(distance)
            matched = r.get("matched_claims")
            if isinstance(matched, list) and matched:
                # Optional: surface claim-level match hints for UI/debugging.
                d["matched_claims"] = matched[:6]
            items.append(d)

        return {"items": items, "has_more": False, "next_cursor": None}

    # Browsing mode: stable newest-first pagination by created_at + mem_id.
    try:
        cursor_obj: Cursor | None = None
        if cursor:
            cursor_obj = decode_cursor(cursor)
    except CursorError as e:
        raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e

    store = SQLiteStore()
    try:
        _ensure_rb_mem_index(store, rb)

        page = store.list_rb_mem_index_page(
            limit=int(limit),
            cursor=(float(cursor_obj.created_at), str(cursor_obj.item_id)) if cursor_obj is not None else None,
            role=role,
            status=status,
            type=type,
        )
        mem_ids = [str(it["mem_id"]) for it in (page.get("items") or [])]
        fetched = rb.get_many(mem_ids=mem_ids, include_content=True)
        by_id = {m.mem_id: m for m in fetched}

        items: list[dict[str, Any]] = []
        for mid in mem_ids:
            it = by_id.get(mid)
            if it is None:
                continue
            items.append(_item_to_dict(it))

        next_cursor = None
        next_tuple = page.get("next_cursor")
        if next_tuple is not None:
            created_at, mem_id = next_tuple
            next_cursor = encode_cursor(Cursor(created_at=float(created_at), item_id=str(mem_id)))

        return {"items": items, "has_more": bool(page.get("has_more")), "next_cursor": next_cursor}
    finally:
        store.close()


@router.get("/memories/{mem_id}")
def get_memory(
    mem_id: str,
    rb: ReasoningBankStore = Depends(get_reasoningbank_store),
) -> dict[str, Any]:
    item = rb.get(mem_id=mem_id)
    if item is None:
        raise APIError(status_code=404, code="not_found", message="Memory not found.")
    return {"memory": _item_to_dict(item)}


@router.post("/memories")
def create_memory(
    body: CreateMemoryRequest,
    rb: ReasoningBankStore = Depends(get_reasoningbank_store),
) -> dict[str, Any]:
    if body.type != "manual_note":
        raise APIError(status_code=400, code="invalid_argument", message="Only type=manual_note is allowed for POST /memories.")

    store = SQLiteStore()
    try:
        item = rb.upsert(
            mem_id=None,
            status=body.status,
            role=body.role,
            type=body.type,
            content=body.content,
            source_run_id=None,
            schema_version=int(body.schema_version),
            extra=body.extra,
        )
        _sync_rb_mem_index(store, item)
        store.append_mem_edit_log(
            mem_id=item.mem_id,
            actor="user",
            reason="create_manual_note",
            before={},
            after=_item_to_dict(item),
        )
        return {"memory": _item_to_dict(item)}
    except ReasoningBankError as e:
        raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e
    finally:
        store.close()


@router.patch("/memories/{mem_id}")
def patch_memory(
    mem_id: str,
    body: PatchMemoryRequest,
    rb: ReasoningBankStore = Depends(get_reasoningbank_store),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        existing = rb.get(mem_id=mem_id)
        if existing is None:
            raise APIError(status_code=404, code="not_found", message="Memory not found.")

        updated = rb.upsert(
            mem_id=existing.mem_id,
            status=body.status if body.status is not None else existing.status,
            role=body.role if body.role is not None else existing.role,
            type=body.type if body.type is not None else existing.type,
            content=body.content if body.content is not None else existing.content,
            source_run_id=existing.source_run_id,
            schema_version=int(body.schema_version) if body.schema_version is not None else existing.schema_version,
            extra=body.extra if body.extra is not None else existing.extra,
            preserve_created_at=True,
        )
        _sync_rb_mem_index(store, updated)
        store.append_mem_edit_log(
            mem_id=updated.mem_id,
            actor="user",
            reason="patch",
            before=_item_to_dict(existing),
            after=_item_to_dict(updated),
        )
        return {"memory": _item_to_dict(updated)}
    except ReasoningBankError as e:
        raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e
    finally:
        store.close()


@router.post("/memories/{mem_id}/archive")
def archive_memory(
    mem_id: str,
    body: dict[str, Any] | None = None,
    rb: ReasoningBankStore = Depends(get_reasoningbank_store),
) -> dict[str, Any]:
    store = SQLiteStore()
    try:
        existing = rb.get(mem_id=mem_id)
        if existing is None:
            raise APIError(status_code=404, code="not_found", message="Memory not found.")
        reason = ""
        if body:
            reason = str(body.get("reason") or "").strip()

        archived = rb.archive(mem_id=existing.mem_id)
        _sync_rb_mem_index(store, archived)
        store.append_mem_edit_log(
            mem_id=archived.mem_id,
            actor="user",
            reason=reason or "archive",
            before=_item_to_dict(existing),
            after=_item_to_dict(archived),
        )
        return {"memory": _item_to_dict(archived)}
    except ReasoningBankError as e:
        raise APIError(status_code=400, code="invalid_argument", message=str(e)) from e
    finally:
        store.close()
