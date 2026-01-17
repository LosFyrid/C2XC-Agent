from __future__ import annotations

import json
from typing import Any

from src.storage.reasoningbank_store import MemoryItem, ReasoningBankError, ReasoningBankStore
from src.storage.sqlite_store import RBJobRecord, SQLiteStore


def _memory_to_dict(item: MemoryItem) -> dict[str, Any]:
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


def _row_to_job(row: Any) -> RBJobRecord:
    return RBJobRecord(
        rb_job_id=str(row["rb_job_id"]),
        run_id=str(row["run_id"]),
        kind=str(row["kind"]),
        created_at=float(row["created_at"]),
        status=str(row["status"]),
    )


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


def enqueue_rb_learn_job(store: SQLiteStore, *, run_id: str) -> RBJobRecord:
    """Enqueue a ReasoningBank learn job for a run (idempotent-ish).

    Semantics:
    - If a learn job is already queued for the run, return that job (dedupe).
    - If a learn job is running and feedback is updated again, enqueue ONE additional queued job
      as a "latest compensation" (so the run eventually learns from the newest feedback).
    """
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required.")

    created: RBJobRecord | None = None
    reason: str = "enqueue"
    supersedes: str | None = None

    # Use an IMMEDIATE transaction to avoid duplicate queued jobs across concurrent HTTP requests.
    with store.transaction(mode="IMMEDIATE"):
        queued = store.get_latest_rb_job_for_run(run_id=rid, kind="learn", statuses=["queued"])
        if queued is not None:
            return _row_to_job(queued)

        running = store.get_latest_rb_job_for_run(run_id=rid, kind="learn", statuses=["running"])
        if running is not None:
            supersedes = str(running["rb_job_id"])
            reason = "latest_compensation"

        created = store.create_rb_job(
            run_id=rid,
            kind="learn",
            extra={"enqueue_reason": reason, "supersedes_rb_job_id": supersedes or ""},
            commit=False,
        )

    assert created is not None
    store.append_event(
        rid,
        "rb_learn_queued",
        {"rb_job_id": created.rb_job_id, "kind": "learn", "status": created.status, "reason": reason, "supersedes_rb_job_id": supersedes},
    )
    return created


def _restore_snapshot(rb: ReasoningBankStore, snapshot: dict[str, Any], *, actor: str, store: SQLiteStore, reason: str) -> MemoryItem:
    mem_id = str(snapshot.get("mem_id") or "").strip()
    if not mem_id:
        raise ReasoningBankError("Invalid snapshot: missing mem_id")

    before = rb.get(mem_id=mem_id)
    restored = rb.upsert(
        mem_id=mem_id,
        status=str(snapshot.get("status") or "active"),
        role=str(snapshot.get("role") or "global"),
        type=str(snapshot.get("type") or "manual_note"),
        content=str(snapshot.get("content") or ""),
        source_run_id=str(snapshot.get("source_run_id") or "").strip() or None,
        schema_version=int(snapshot.get("schema_version") or 1),
        extra=dict(snapshot.get("extra") or {}),
        now_ts=float(snapshot.get("updated_at") or 0.0) or None,
        preserve_created_at=True,
        validate=False,
    )

    store.append_mem_edit_log(
        mem_id=restored.mem_id,
        actor=actor,
        reason=reason,
        before=_memory_to_dict(before) if before is not None else {},
        after=_memory_to_dict(restored),
    )
    _sync_rb_mem_index(store, restored)
    return restored


def rollback_rb_delta(
    store: SQLiteStore,
    *,
    rb: ReasoningBankStore,
    run_id: str,
    delta_id: str | None,
    reason: str | None = None,
) -> str:
    """Rollback a previously applied RB delta (strict rollback semantics)."""
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required.")

    # Choose delta.
    if delta_id:
        row = store.get_rb_delta(delta_id=delta_id)
        if row is None:
            raise ValueError("delta not found")
        if str(row["run_id"]) != rid:
            raise ValueError("delta does not belong to run")
        delta_row = row
    else:
        deltas = store.list_rb_deltas_for_run(run_id=rid)
        applied = [d for d in deltas if str(d.get("status") or "") == "applied"]
        if not applied:
            raise ValueError("no applied delta to rollback")
        # list_rb_deltas_for_run returns newest-first.
        delta_row = store.get_rb_delta(delta_id=str(applied[0]["delta_id"]))
        if delta_row is None:
            raise ValueError("delta not found")

    did = str(delta_row["delta_id"])
    status = str(delta_row["status"])
    if status == "rolled_back":
        return did

    try:
        ops = json.loads(str(delta_row["ops_json"] or "[]"))
        if not isinstance(ops, list):
            ops = []
    except Exception:
        ops = []

    store.append_event(
        rid,
        "rb_rollback_started",
        {"delta_id": did, "reason": str(reason or "") or None, "n_ops": len(ops)},
    )

    actor = "rb_rollback"
    rollback_reason = f"rollback_delta:{did}"

    # Strict rollback: apply inverse in reverse order.
    for op in reversed(ops):
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("op") or "").strip()
        mem_id = str(op.get("mem_id") or "").strip()
        if not mem_id:
            continue

        if op_type == "add":
            # Rollback "add" by archiving (soft delete) so trace references remain viewable.
            before = rb.get(mem_id=mem_id)
            if before is None:
                continue
            after = rb.archive(mem_id=mem_id)
            store.append_mem_edit_log(
                mem_id=mem_id,
                actor=actor,
                reason=rollback_reason,
                before=_memory_to_dict(before),
                after=_memory_to_dict(after),
            )
            _sync_rb_mem_index(store, after)
            continue

        if op_type in {"update", "archive"}:
            snap = op.get("before") if isinstance(op.get("before"), dict) else None
            if snap is None:
                # Best-effort fallback: if we can't restore exact snapshot, at least unarchive archived items.
                before = rb.get(mem_id=mem_id)
                if before is None:
                    continue
                if op_type == "archive" and before.status == "archived":
                    restored = rb.upsert(
                        mem_id=before.mem_id,
                        status="active",
                        role=before.role,
                        type=before.type,
                        content=before.content,
                        source_run_id=before.source_run_id,
                        schema_version=before.schema_version,
                        extra=before.extra,
                        preserve_created_at=True,
                        validate=False,
                    )
                    store.append_mem_edit_log(
                        mem_id=mem_id,
                        actor=actor,
                        reason=rollback_reason,
                        before=_memory_to_dict(before),
                        after=_memory_to_dict(restored),
                    )
                    _sync_rb_mem_index(store, restored)
                continue

            _restore_snapshot(rb, snap, actor=actor, store=store, reason=rollback_reason)
            continue

        # Unknown ops are ignored for forward-compatibility.

    store.mark_rb_delta_rolled_back(delta_id=did, reason=str(reason or "") or None)
    store.append_event(
        rid,
        "rb_rollback_completed",
        {"delta_id": did, "status": "rolled_back"},
    )
    return did
