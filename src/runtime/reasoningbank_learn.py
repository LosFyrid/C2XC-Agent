from __future__ import annotations

import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, cast

from src.config.load_config import AppConfig
from src.llm.openai_compat import OpenAICompatibleChatClient
from src.reasoningbank.rbmem_claims_v1 import (
    RBMEM_CLAIMS_V1_HEADER,
    RBMemClaimsV1ParseError,
    RBMemClaimsV1ValidationError,
    is_rbmem_claims_v1,
    parse_rbmem_claims_v1,
    validate_rbmem_claims_v1,
)
from src.runtime.reasoningbank_jobs import rollback_rb_delta
from src.storage.reasoningbank_store import MemoryItem, ReasoningBankError, ReasoningBankStore
from src.storage.sqlite_store import SQLiteStore
from src.utils.json_extract import JSONExtractionError, extract_first_json_object
from src.utils.template import render_template


class RBLearnError(RuntimeError):
    pass


@dataclass(frozen=True)
class RBLearnSnapshot:
    """Immutable snapshot for a single RB learn job (prevents mixed reads)."""

    snapshot_version: int
    run_id: str
    rb_job_id: str
    trace_cutoff_ts: float
    feedback_id: str
    feedback_updated_at: float
    final_output_event_id: str | None


@dataclass
class RBLearnDerefBudget:
    max_calls_total: int
    max_full_calls: int
    max_chars_total: int
    excerpt_chars: int
    full_chars: int

    used_calls_total: int = 0
    used_full_calls: int = 0
    used_chars_total: int = 0

    def _consume(self, *, full: bool, n_chars: int) -> None:
        self.used_calls_total += 1
        if full:
            self.used_full_calls += 1
        self.used_chars_total += max(0, int(n_chars))

    def can_open_any(self) -> bool:
        return self.used_calls_total < int(self.max_calls_total) and self.used_chars_total < int(self.max_chars_total)

    def can_open_full(self) -> bool:
        return self.used_full_calls < int(self.max_full_calls) and self.can_open_any()


_FORBIDDEN_TRACE_EVENT_TYPES = {
    # Main run model logs.
    "llm_request",
    "llm_response",
    # RB learn model logs (this module).
    "rb_llm_request",
    "rb_llm_response",
}

_KB_ALIAS_RE = re.compile(r"\[C\d+(?:\s*,\s*C\d+)*\]")


def _strip_kb_aliases(text: str) -> str:
    """Remove run-local KB aliases like [C12] from strings before persisting to RB."""
    return _KB_ALIAS_RE.sub("", text or "").strip()


def _strip_kb_aliases_any(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_kb_aliases(value)
    if isinstance(value, list):
        return [_strip_kb_aliases_any(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_kb_aliases_any(v) for k, v in value.items()}
    return value


def _build_facts_digest(*, run_id: str, run_output_json: dict[str, Any], feedback_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a system-provided FACTS digest for RB learn.

    Hard rule: do not persist KB alias tokens like [C12] into RB memories.
    """
    rid = (run_id or "").strip()
    recipes_any = (run_output_json.get("recipes_json") or {}).get("recipes") if isinstance(run_output_json.get("recipes_json"), dict) else None
    recipes = recipes_any if isinstance(recipes_any, list) else []
    recipe_facts: list[dict[str, Any]] = []
    for r in recipes[:5]:
        if not isinstance(r, dict):
            continue
        recipe_facts.append(
            {
                "M1": str(r.get("M1") or "").strip(),
                "M2": str(r.get("M2") or "").strip(),
                "atomic_ratio": str(r.get("atomic_ratio") or "").strip(),
                "small_molecule_modifier": str(r.get("small_molecule_modifier") or "").strip(),
            }
        )

    fb = feedback_payload.get("feedback") if isinstance(feedback_payload, dict) else None
    fb_obj = fb if isinstance(fb, dict) else {}
    products_any = fb_obj.get("products")
    products = products_any if isinstance(products_any, list) else []

    products_clean: list[dict[str, Any]] = []
    total_value = 0.0
    for p in products:
        if not isinstance(p, dict):
            continue
        try:
            v = float(p.get("value") or 0.0)
        except Exception:
            v = 0.0
        try:
            frac = float(p.get("fraction") or 0.0)
        except Exception:
            frac = 0.0
        total_value += max(0.0, v)
        products_clean.append(
            {
                "product_name": str(p.get("product_name") or ""),
                "value": v,
                "fraction": frac,
            }
        )

    # Rank by fraction (selectivity signal) but keep value too (activity proxy).
    products_top = sorted(products_clean, key=lambda x: float(x.get("fraction") or 0.0), reverse=True)[:8]

    out: dict[str, Any] = {
        "run_id": rid,
        "recipes": recipe_facts,
        "feedback": {
            "score": fb_obj.get("score"),
            "pros": str(fb_obj.get("pros") or ""),
            "cons": str(fb_obj.get("cons") or ""),
            "other": str(fb_obj.get("other") or ""),
            "products_top": products_top,
            "activity_total_value": float(total_value),
        },
    }
    return cast(dict[str, Any], _strip_kb_aliases_any(out))


def _build_candidate_query_seed(*, facts_digest: dict[str, Any]) -> str:
    """Build a compact semantic query seed for retrieving related RB memories."""
    parts: list[str] = []
    run_id = str(facts_digest.get("run_id") or "").strip()
    if run_id:
        parts.append(f"run_id={run_id}")

    recipes = facts_digest.get("recipes")
    if isinstance(recipes, list):
        for r in recipes[:3]:
            if not isinstance(r, dict):
                continue
            s = " ".join(
                [
                    str(r.get("M1") or "").strip(),
                    str(r.get("M2") or "").strip(),
                    str(r.get("atomic_ratio") or "").strip(),
                    str(r.get("small_molecule_modifier") or "").strip(),
                ]
            ).strip()
            if s:
                parts.append(s)

    fb = facts_digest.get("feedback")
    if isinstance(fb, dict):
        products = fb.get("products_top")
        if isinstance(products, list):
            names = [str(p.get("product_name") or "").strip() for p in products if isinstance(p, dict)]
            names = [n for n in names if n]
            if names:
                parts.append("products_top=" + ",".join(names[:6]))
        pros = str(fb.get("pros") or "").strip()
        cons = str(fb.get("cons") or "").strip()
        other = str(fb.get("other") or "").strip()
        text = " ".join([pros, cons, other]).strip()
        if text:
            # Keep it short; this is just for semantic retrieval.
            if len(text) > 600:
                text = text[:600] + "…"
            parts.append(text)

    return "\n".join([p for p in parts if p]).strip()


def _collect_candidate_mem_ids(
    *,
    cfg: AppConfig,
    rb: ReasoningBankStore,
    run_output_json: dict[str, Any],
    trace_digest: dict[str, Any],
    facts_digest: dict[str, Any],
) -> list[str]:
    """Select a bounded candidate set of existing memories to update via claim verdicts."""
    used: list[str] = []
    seen: set[str] = set()

    # 1) Strong signal: mem_ids explicitly referenced by final output.
    mids_any = run_output_json.get("memory_ids")
    if isinstance(mids_any, list):
        for mid in mids_any:
            s = str(mid or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            used.append(s)

    # 2) Strong signal: mem_search results in trace digest (what the model actually looked at).
    latest_mem_search = trace_digest.get("latest_mem_search")
    if isinstance(latest_mem_search, dict):
        results_any = latest_mem_search.get("results")
        if isinstance(results_any, list):
            for r in results_any:
                if not isinstance(r, dict):
                    continue
                mid = str(r.get("mem_id") or "").strip()
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                used.append(mid)

    # 3) Recall: semantic claim-search on run facts digest.
    seed = _build_candidate_query_seed(facts_digest=facts_digest)
    sem_k = int(cfg.reasoningbank.learn_candidate_semantic_top_k)
    sem_results = rb.query(
        query=seed,
        n_results=max(sem_k, 1),
        status=["active"],
        type=["reasoningbank_item"],
    )
    sem_ranked: list[str] = []
    for r in sem_results:
        it: MemoryItem = r["item"]
        if it.mem_id in seen:
            continue
        seen.add(it.mem_id)
        sem_ranked.append(it.mem_id)

    cap = int(cfg.reasoningbank.learn_candidate_max_items)
    out = used[:]
    for mid in sem_ranked:
        if len(out) >= cap:
            break
        out.append(mid)
    return out


def _normalize_alias(value: str) -> str:
    s = (value or "").strip()
    if s.startswith("[") and s.endswith("]") and len(s) >= 3:
        s = s[1:-1].strip()
    return s


def _clamp_int(value: Any, *, default: int, min_v: int, max_v: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(default)
    if v < int(min_v):
        return int(min_v)
    if v > int(max_v):
        return int(max_v)
    return int(v)


def _truncate_strings(value: Any, *, max_len: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_len=int(max_len))
    if isinstance(value, list):
        return [_truncate_strings(v, max_len=max_len) for v in value]
    if isinstance(value, dict):
        return {k: _truncate_strings(v, max_len=max_len) for k, v in value.items()}
    return value


def _sanitize_event_payload(event_type: str, payload: Any, *, max_str_len: int) -> dict[str, Any]:
    obj = payload if isinstance(payload, dict) else {}

    et = str(event_type or "").strip()
    if et == "recap_info":
        # Treat recap_info as "facts": keep what was done and the result; drop internal "think" if present.
        cleaned = dict(obj)
        cleaned.pop("think", None)
        return cast(dict[str, Any], _truncate_strings(cleaned, max_len=max_str_len))

    # Default: keep payload but truncate string fields to control size.
    return cast(dict[str, Any], _truncate_strings(obj, max_len=max_str_len))


def _rb_learn_deref_tools_schema() -> list[dict[str, Any]]:
    """Tool schemas for the RB learn extractor (B-scheme: factual originals only)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "rb_list_events",
                "description": (
                    "List factual trace events available for this run within the RB learn snapshot. "
                    "Use this to discover event_id values to open. "
                    "NOTE: LLM request/response logs are not accessible."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event_types": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rb_open_event",
                "description": (
                    "Open a factual trace event by event_id (within snapshot cutoff). "
                    "Forbidden: llm_request/llm_response and rb_llm_request/rb_llm_response."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event_id": {"type": "string"},
                        "mode": {"type": "string", "enum": ["excerpt", "full"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["event_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rb_open_memory",
                "description": "Open a ReasoningBank memory by mem_id (original content from Chroma).",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mem_id": {"type": "string"},
                        "mode": {"type": "string", "enum": ["excerpt", "full"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["mem_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rb_open_evidence",
                "description": (
                    "Open run evidence text by alias (e.g. C12 or P3) or canonical ref (kb:.../pubchem:...). "
                    "This returns the original evidence content recorded during the run."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "alias": {"type": "string"},
                        "ref": {"type": "string"},
                        "mode": {"type": "string", "enum": ["excerpt", "full"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rb_open_feedback",
                "description": "Open the experiment feedback JSON (factual record).",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mode": {"type": "string", "enum": ["excerpt", "full"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rb_open_run_output",
                "description": "Open the run final_output JSON (factual record).",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mode": {"type": "string", "enum": ["excerpt", "full"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    ]


@dataclass
class _RBLearnDerefContext:
    store: SQLiteStore
    rb: ReasoningBankStore
    cfg: AppConfig
    snapshot: RBLearnSnapshot
    budget: RBLearnDerefBudget
    feedback_payload: dict[str, Any]
    run_output_json: dict[str, Any]


def _tool_error(*, code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": {"code": str(code), "message": str(message)}}
    if details:
        out["error"]["details"] = details
    return out


def _append_rb_source_opened(
    ctx: _RBLearnDerefContext,
    *,
    source_type: str,
    source_id: str,
    mode_requested: str,
    mode_used: str,
    truncated: bool,
    returned_chars: int,
    reason: str | None,
    error_code: str | None = None,
) -> None:
    ctx.store.append_event(
        ctx.snapshot.run_id,
        "rb_source_opened",
        {
            "ts": time.time(),
            "rb_job_id": ctx.snapshot.rb_job_id,
            "snapshot_version": int(ctx.snapshot.snapshot_version),
            "trace_cutoff_ts": float(ctx.snapshot.trace_cutoff_ts),
            "feedback_id": ctx.snapshot.feedback_id,
            "feedback_updated_at": float(ctx.snapshot.feedback_updated_at),
            "final_output_event_id": ctx.snapshot.final_output_event_id,
            "source_type": str(source_type),
            "source_id": str(source_id),
            "mode_requested": str(mode_requested),
            "mode_used": str(mode_used),
            "truncated": bool(truncated),
            "returned_chars": int(returned_chars),
            "error_code": str(error_code) if error_code else None,
            "reason": str(reason or "") or None,
            "budget": {
                "used_calls_total": int(ctx.budget.used_calls_total),
                "used_full_calls": int(ctx.budget.used_full_calls),
                "used_chars_total": int(ctx.budget.used_chars_total),
                "max_calls_total": int(ctx.budget.max_calls_total),
                "max_full_calls": int(ctx.budget.max_full_calls),
                "max_chars_total": int(ctx.budget.max_chars_total),
            },
        },
    )


def _resolve_mode(ctx: _RBLearnDerefContext, mode_raw: Any) -> tuple[str, int, bool]:
    requested = str(mode_raw or "").strip().lower() or "excerpt"
    if requested not in {"excerpt", "full"}:
        requested = "excerpt"

    if not ctx.budget.can_open_any():
        return ("blocked", 0, False)

    if requested == "full" and not ctx.budget.can_open_full():
        return ("excerpt", int(ctx.budget.excerpt_chars), True)

    used_mode = requested
    max_chars = int(ctx.budget.full_chars) if used_mode == "full" else int(ctx.budget.excerpt_chars)
    return (used_mode, max_chars, False)


def _deref_list_events(ctx: _RBLearnDerefContext, args: dict[str, Any]) -> dict[str, Any]:
    if not ctx.budget.can_open_any():
        out = _tool_error(
            code="budget_exceeded",
            message="Cannot list events: dereference budget exhausted.",
            details={"budget": {"max_calls_total": ctx.budget.max_calls_total}},
        )
        _append_rb_source_opened(
            ctx,
            source_type="trace_events",
            source_id="list",
            mode_requested="excerpt",
            mode_used="blocked",
            truncated=False,
            returned_chars=len(json.dumps(out, ensure_ascii=False)),
            reason=str(args.get("reason") or ""),
            error_code="budget_exceeded",
        )
        return out

    requested_types = args.get("event_types")
    event_types: list[str] | None = None
    if isinstance(requested_types, list):
        event_types = [str(t) for t in requested_types if str(t).strip()]
        if not event_types:
            event_types = None

    # Default: show "most useful factual events" for learning.
    default_types = [
        "final_output",
        "run_failed",
        "recap_info",
        "kb_query",
        "kb_get",
        "kb_list",
        "mem_search",
        "mem_get",
        "mem_list",
        "citations_resolved",
        "memories_resolved",
    ]
    effective_types = event_types or default_types

    filtered: list[str] = []
    blocked: list[str] = []
    for t in effective_types:
        et = str(t).strip()
        if not et:
            continue
        if et in _FORBIDDEN_TRACE_EVENT_TYPES or et.startswith("llm_") or et.startswith("rb_llm_"):
            blocked.append(et)
            continue
        filtered.append(et)

    limit = _clamp_int(
        args.get("limit"),
        default=int(ctx.cfg.reasoningbank.learn_deref_list_events_default_limit),
        min_v=1,
        max_v=int(ctx.cfg.reasoningbank.learn_deref_list_events_max_limit),
    )

    rows: list[dict[str, Any]] = []
    if filtered:
        rows = ctx.store.list_latest_events(
            run_id=ctx.snapshot.run_id,
            limit=int(limit),
            event_types=filtered,
            include_payload=True,
            until=float(ctx.snapshot.trace_cutoff_ts),
        )

    items: list[dict[str, Any]] = []
    for r in rows:
        et = str(r.get("event_type") or "")
        payload = r.get("payload")
        summary = ""
        if isinstance(payload, dict):
            if et == "kb_query":
                q = _truncate(str(payload.get("query") or ""), max_len=160)
                kb_name = str(payload.get("kb_name") or payload.get("kb_namespace") or "")
                agent = str(payload.get("agent") or "")
                summary = f"agent={agent} kb={kb_name} query={q}"
            elif et == "recap_info":
                agent = str(payload.get("agent") or "")
                task_name = str(payload.get("task_name") or "")
                recap_state = str(payload.get("recap_state") or "")
                summary = f"agent={agent} state={recap_state} task={task_name}"
            elif et == "mem_search":
                q = _truncate(str(payload.get("query") or ""), max_len=160)
                agent = str(payload.get("agent") or "")
                summary = f"agent={agent} query={q}"
            elif et == "final_output":
                recipes_json = payload.get("recipes_json")
                n_recipes = 0
                if isinstance(recipes_json, dict) and isinstance(recipes_json.get("recipes"), list):
                    n_recipes = len(recipes_json.get("recipes") or [])
                summary = f"recipes={n_recipes}"
            elif et == "run_failed":
                err = _truncate(str(payload.get("error") or ""), max_len=160)
                summary = f"error={err}"
        items.append(
            {
                "event_id": str(r.get("event_id") or ""),
                "created_at": float(r.get("created_at") or 0.0),
                "event_type": et,
                "summary": summary,
            }
        )

    out: dict[str, Any] = {
        "ok": True,
        "snapshot": {
            "trace_cutoff_ts": float(ctx.snapshot.trace_cutoff_ts),
            "feedback_id": ctx.snapshot.feedback_id,
            "feedback_updated_at": float(ctx.snapshot.feedback_updated_at),
        },
        "blocked_event_types": blocked,
        "items": items,
    }
    out_s = json.dumps(out, ensure_ascii=False)
    ctx.budget._consume(full=False, n_chars=len(out_s))
    _append_rb_source_opened(
        ctx,
        source_type="trace_events",
        source_id="list",
        mode_requested="excerpt",
        mode_used="excerpt",
        truncated=False,
        returned_chars=len(out_s),
        reason=str(args.get("reason") or ""),
    )
    return out


def _deref_open_event(ctx: _RBLearnDerefContext, args: dict[str, Any]) -> dict[str, Any]:
    event_id = str(args.get("event_id") or "").strip()
    if not event_id:
        out = _tool_error(code="invalid_argument", message="event_id is required.")
        ctx.budget._consume(full=False, n_chars=len(json.dumps(out, ensure_ascii=False)))
        return out

    used_mode, max_chars, degraded = _resolve_mode(ctx, args.get("mode"))
    if used_mode == "blocked":
        out = _tool_error(code="budget_exceeded", message="Cannot open event: dereference budget exhausted.")
        _append_rb_source_opened(
            ctx,
            source_type="event",
            source_id=event_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used="blocked",
            truncated=False,
            returned_chars=len(json.dumps(out, ensure_ascii=False)),
            reason=str(args.get("reason") or ""),
            error_code="budget_exceeded",
        )
        return out

    row = ctx.store.get_event(run_id=ctx.snapshot.run_id, event_id=event_id)
    if row is None:
        out = _tool_error(code="not_found", message="Event not found.")
        out_s = json.dumps(out, ensure_ascii=False)
        ctx.budget._consume(full=False, n_chars=len(out_s))
        _append_rb_source_opened(
            ctx,
            source_type="event",
            source_id=event_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used=used_mode,
            truncated=False,
            returned_chars=len(out_s),
            reason=str(args.get("reason") or ""),
            error_code="not_found",
        )
        return out

    created_at = float(row["created_at"])
    if created_at > float(ctx.snapshot.trace_cutoff_ts):
        out = _tool_error(
            code="snapshot_out_of_bounds",
            message="Event is outside the RB learn snapshot cutoff (newer than trace_cutoff_ts).",
            details={"event_created_at": created_at, "trace_cutoff_ts": float(ctx.snapshot.trace_cutoff_ts)},
        )
        out_s = json.dumps(out, ensure_ascii=False)
        ctx.budget._consume(full=False, n_chars=len(out_s))
        _append_rb_source_opened(
            ctx,
            source_type="event",
            source_id=event_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used=used_mode,
            truncated=False,
            returned_chars=len(out_s),
            reason=str(args.get("reason") or ""),
            error_code="snapshot_out_of_bounds",
        )
        return out

    event_type = str(row["event_type"])
    if event_type in _FORBIDDEN_TRACE_EVENT_TYPES or event_type.startswith("llm_") or event_type.startswith("rb_llm_"):
        out = _tool_error(
            code="forbidden_event_type",
            message=f"Access to event_type={event_type!r} is forbidden in RB learn (facts-only policy).",
            details={"event_type": event_type},
        )
        out_s = json.dumps(out, ensure_ascii=False)
        ctx.budget._consume(full=False, n_chars=len(out_s))
        _append_rb_source_opened(
            ctx,
            source_type="event",
            source_id=event_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used=used_mode,
            truncated=False,
            returned_chars=len(out_s),
            reason=str(args.get("reason") or ""),
            error_code="forbidden_event_type",
        )
        return out

    try:
        payload_any = json.loads(str(row["payload_json"] or "{}"))
    except Exception:
        payload_any = {}

    payload = _sanitize_event_payload(event_type, payload_any, max_str_len=max_chars)
    out: dict[str, Any] = {
        "ok": True,
        "event": {
            "event_id": event_id,
            "run_id": ctx.snapshot.run_id,
            "created_at": created_at,
            "event_type": event_type,
            "payload": payload,
            "mode": used_mode,
            "degraded_from_full": bool(degraded),
        },
    }
    out_s = json.dumps(out, ensure_ascii=False)
    ctx.budget._consume(full=(used_mode == "full"), n_chars=len(out_s))
    _append_rb_source_opened(
        ctx,
        source_type="event",
        source_id=event_id,
        mode_requested=str(args.get("mode") or "excerpt"),
        mode_used=used_mode,
        truncated=False,
        returned_chars=len(out_s),
        reason=str(args.get("reason") or ""),
    )
    return out


def _deref_open_memory(ctx: _RBLearnDerefContext, args: dict[str, Any]) -> dict[str, Any]:
    mem_id = str(args.get("mem_id") or "").strip()
    if not mem_id:
        out = _tool_error(code="invalid_argument", message="mem_id is required.")
        ctx.budget._consume(full=False, n_chars=len(json.dumps(out, ensure_ascii=False)))
        return out

    used_mode, max_chars, degraded = _resolve_mode(ctx, args.get("mode"))
    if used_mode == "blocked":
        out = _tool_error(code="budget_exceeded", message="Cannot open memory: dereference budget exhausted.")
        _append_rb_source_opened(
            ctx,
            source_type="memory",
            source_id=mem_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used="blocked",
            truncated=False,
            returned_chars=len(json.dumps(out, ensure_ascii=False)),
            reason=str(args.get("reason") or ""),
            error_code="budget_exceeded",
        )
        return out

    item = ctx.rb.get(mem_id=mem_id)
    if item is None:
        out = _tool_error(code="not_found", message="Memory not found.")
        out_s = json.dumps(out, ensure_ascii=False)
        ctx.budget._consume(full=False, n_chars=len(out_s))
        _append_rb_source_opened(
            ctx,
            source_type="memory",
            source_id=mem_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used=used_mode,
            truncated=False,
            returned_chars=len(out_s),
            reason=str(args.get("reason") or ""),
            error_code="not_found",
        )
        return out

    full_text = str(item.content or "")
    truncated = len(full_text) > int(max_chars)
    content = _truncate(full_text, max_len=int(max_chars))

    out: dict[str, Any] = {
        "ok": True,
        "memory": {
            "mem_id": item.mem_id,
            "status": item.status,
            "role": item.role,
            "type": item.type,
            "source_run_id": item.source_run_id,
            "created_at": float(item.created_at),
            "updated_at": float(item.updated_at),
            "schema_version": int(item.schema_version),
            "content": content,
            "mode": used_mode,
            "truncated": bool(truncated),
            "degraded_from_full": bool(degraded),
        },
    }
    out_s = json.dumps(out, ensure_ascii=False)
    ctx.budget._consume(full=(used_mode == "full"), n_chars=len(out_s))
    _append_rb_source_opened(
        ctx,
        source_type="memory",
        source_id=mem_id,
        mode_requested=str(args.get("mode") or "excerpt"),
        mode_used=used_mode,
        truncated=bool(truncated),
        returned_chars=len(out_s),
        reason=str(args.get("reason") or ""),
    )
    return out


def _find_kb_evidence_in_run(
    store: SQLiteStore,
    *,
    run_id: str,
    trace_cutoff_ts: float,
    alias: str | None,
    ref: str | None,
) -> dict[str, Any] | None:
    want_alias = _normalize_alias(alias or "")
    want_ref = str(ref or "").strip()

    cursor: tuple[float, str] | None = None
    while True:
        page = store.list_events_page(
            run_id=run_id,
            limit=200,
            cursor=cursor,
            event_types=["kb_query", "pubchem_query"],
            include_payload=True,
            since=None,
            until=float(trace_cutoff_ts),
        )
        items = page.get("items") or []
        for ev in items:
            payload = ev.get("payload")
            if not isinstance(payload, dict):
                continue
            results = payload.get("results")
            if not isinstance(results, list):
                continue
            for r in results:
                if not isinstance(r, dict):
                    continue
                a = _normalize_alias(str(r.get("alias") or ""))
                rr = str(r.get("ref") or "").strip()
                if want_alias and a == want_alias:
                    return dict(r)
                if want_ref and rr and rr == want_ref:
                    return dict(r)

        if not bool(page.get("has_more")):
            return None

        last = items[-1]
        cursor = (float(last.get("created_at") or 0.0), str(last.get("event_id") or ""))


def _deref_open_evidence(ctx: _RBLearnDerefContext, args: dict[str, Any]) -> dict[str, Any]:
    alias = str(args.get("alias") or "").strip()
    ref = str(args.get("ref") or "").strip()

    if bool(alias) == bool(ref):
        out = _tool_error(code="invalid_argument", message="Provide exactly one of {alias, ref}.")
        ctx.budget._consume(full=False, n_chars=len(json.dumps(out, ensure_ascii=False)))
        return out

    used_mode, max_chars, degraded = _resolve_mode(ctx, args.get("mode"))
    if used_mode == "blocked":
        out = _tool_error(code="budget_exceeded", message="Cannot open evidence: dereference budget exhausted.")
        _append_rb_source_opened(
            ctx,
            source_type="evidence",
            source_id=alias or ref,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used="blocked",
            truncated=False,
            returned_chars=len(json.dumps(out, ensure_ascii=False)),
            reason=str(args.get("reason") or ""),
            error_code="budget_exceeded",
        )
        return out

    found = _find_kb_evidence_in_run(
        ctx.store,
        run_id=ctx.snapshot.run_id,
        trace_cutoff_ts=float(ctx.snapshot.trace_cutoff_ts),
        alias=alias if alias else None,
        ref=ref if ref else None,
    )
    if found is None:
        out = _tool_error(code="not_found", message="Evidence not found in run trace (kb_query/pubchem_query).")
        out_s = json.dumps(out, ensure_ascii=False)
        ctx.budget._consume(full=False, n_chars=len(out_s))
        _append_rb_source_opened(
            ctx,
            source_type="evidence",
            source_id=alias or ref,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used=used_mode,
            truncated=False,
            returned_chars=len(out_s),
            reason=str(args.get("reason") or ""),
            error_code="not_found",
        )
        return out

    content_full = str(found.get("content") or "")
    truncated = len(content_full) > int(max_chars)
    content = _truncate(content_full, max_len=int(max_chars))

    out: dict[str, Any] = {
        "ok": True,
        "evidence": {
            "alias": _normalize_alias(str(found.get("alias") or "")),
            "ref": str(found.get("ref") or ""),
            "source": str(found.get("source") or ""),
            "kb_namespace": str(found.get("kb_namespace") or ""),
            "lightrag_chunk_id": str(found.get("lightrag_chunk_id") or "") or None,
            "content": content,
            "mode": used_mode,
            "truncated": bool(truncated),
            "degraded_from_full": bool(degraded),
        },
    }
    out_s = json.dumps(out, ensure_ascii=False)
    ctx.budget._consume(full=(used_mode == "full"), n_chars=len(out_s))
    _append_rb_source_opened(
        ctx,
        source_type="evidence",
        source_id=alias or ref,
        mode_requested=str(args.get("mode") or "excerpt"),
        mode_used=used_mode,
        truncated=bool(truncated),
        returned_chars=len(out_s),
        reason=str(args.get("reason") or ""),
    )
    return out


def _deref_open_feedback(ctx: _RBLearnDerefContext, args: dict[str, Any]) -> dict[str, Any]:
    used_mode, max_chars, degraded = _resolve_mode(ctx, args.get("mode"))
    if used_mode == "blocked":
        out = _tool_error(code="budget_exceeded", message="Cannot open feedback: dereference budget exhausted.")
        _append_rb_source_opened(
            ctx,
            source_type="feedback",
            source_id=ctx.snapshot.feedback_id,
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used="blocked",
            truncated=False,
            returned_chars=len(json.dumps(out, ensure_ascii=False)),
            reason=str(args.get("reason") or ""),
            error_code="budget_exceeded",
        )
        return out

    payload = _truncate_strings(ctx.feedback_payload, max_len=max_chars)
    out: dict[str, Any] = {
        "ok": True,
        "feedback": payload,
        "mode": used_mode,
        "degraded_from_full": bool(degraded),
    }
    out_s = json.dumps(out, ensure_ascii=False)
    ctx.budget._consume(full=(used_mode == "full"), n_chars=len(out_s))
    _append_rb_source_opened(
        ctx,
        source_type="feedback",
        source_id=ctx.snapshot.feedback_id,
        mode_requested=str(args.get("mode") or "excerpt"),
        mode_used=used_mode,
        truncated=False,
        returned_chars=len(out_s),
        reason=str(args.get("reason") or ""),
    )
    return out


def _deref_open_run_output(ctx: _RBLearnDerefContext, args: dict[str, Any]) -> dict[str, Any]:
    used_mode, max_chars, degraded = _resolve_mode(ctx, args.get("mode"))
    if used_mode == "blocked":
        out = _tool_error(code="budget_exceeded", message="Cannot open run output: dereference budget exhausted.")
        _append_rb_source_opened(
            ctx,
            source_type="run_output",
            source_id=ctx.snapshot.final_output_event_id or "",
            mode_requested=str(args.get("mode") or "excerpt"),
            mode_used="blocked",
            truncated=False,
            returned_chars=len(json.dumps(out, ensure_ascii=False)),
            reason=str(args.get("reason") or ""),
            error_code="budget_exceeded",
        )
        return out

    payload = _truncate_strings(ctx.run_output_json, max_len=max_chars)
    out: dict[str, Any] = {
        "ok": True,
        "run_output": payload,
        "mode": used_mode,
        "degraded_from_full": bool(degraded),
    }
    out_s = json.dumps(out, ensure_ascii=False)
    ctx.budget._consume(full=(used_mode == "full"), n_chars=len(out_s))
    _append_rb_source_opened(
        ctx,
        source_type="run_output",
        source_id=ctx.snapshot.final_output_event_id or "",
        mode_requested=str(args.get("mode") or "excerpt"),
        mode_used=used_mode,
        truncated=False,
        returned_chars=len(out_s),
        reason=str(args.get("reason") or ""),
    )
    return out


def _execute_deref_tool(ctx: _RBLearnDerefContext, *, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "rb_list_events":
        return _deref_list_events(ctx, args)
    if name == "rb_open_event":
        return _deref_open_event(ctx, args)
    if name == "rb_open_memory":
        return _deref_open_memory(ctx, args)
    if name == "rb_open_evidence":
        return _deref_open_evidence(ctx, args)
    if name == "rb_open_feedback":
        return _deref_open_feedback(ctx, args)
    if name == "rb_open_run_output":
        return _deref_open_run_output(ctx, args)
    return _tool_error(code="unknown_tool", message=f"Unknown tool: {name}")


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


def _truncate(text: str, *, max_len: int) -> str:
    s = str(text or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _latest_event_payload(
    store: SQLiteStore,
    *,
    run_id: str,
    event_type: str,
    until: float | None,
) -> dict[str, Any] | None:
    rows = store.list_latest_events(
        run_id=run_id,
        limit=1,
        event_types=[event_type],
        include_payload=True,
        until=until,
    )
    if not rows:
        return None
    payload = rows[0].get("payload")
    return payload if isinstance(payload, dict) else None


def _shrink_kb_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # kb_query payload may include full chunk content; keep only lightweight metadata.
    out: dict[str, Any] = {}
    out["ts"] = payload.get("ts")
    out["agent"] = payload.get("agent")
    out["kb_name"] = payload.get("kb_name")
    out["mode"] = payload.get("mode")
    out["top_k"] = payload.get("top_k")
    out["query"] = _truncate(str(payload.get("query") or ""), max_len=320)

    results = payload.get("results")
    if isinstance(results, list):
        slim: list[dict[str, Any]] = []
        for r in results[:12]:
            if not isinstance(r, dict):
                continue
            slim.append(
                {
                    "alias": str(r.get("alias") or ""),
                    "ref": str(r.get("ref") or ""),
                    "source": str(r.get("source") or ""),
                    "kb_namespace": str(r.get("kb_namespace") or ""),
                    "lightrag_chunk_id": str(r.get("lightrag_chunk_id") or "") or None,
                }
            )
        out["results"] = slim
    return out


def _build_run_trace_digest(store: SQLiteStore, *, snapshot: RBLearnSnapshot) -> dict[str, Any]:
    """Build a compact trace digest to make RB extraction more 'experience-driven'.

    This is intentionally lightweight:
    - No raw LLM prompts/responses
    - No full KB chunk content
    - Focus on tool usage + resolved citations/memories
    """
    rid = (snapshot.run_id or "").strip()
    if not rid:
        return {}

    event_counts = store.count_event_types_for_run(run_id=rid, until=float(snapshot.trace_cutoff_ts))

    latest_mem_search = _latest_event_payload(store, run_id=rid, event_type="mem_search", until=snapshot.trace_cutoff_ts)
    latest_memories_resolved = _latest_event_payload(store, run_id=rid, event_type="memories_resolved", until=snapshot.trace_cutoff_ts)
    latest_citations_resolved = _latest_event_payload(store, run_id=rid, event_type="citations_resolved", until=snapshot.trace_cutoff_ts)
    latest_run_failed = _latest_event_payload(store, run_id=rid, event_type="run_failed", until=snapshot.trace_cutoff_ts)

    recent_kb_queries_raw = store.list_latest_events(
        run_id=rid,
        limit=3,
        event_types=["kb_query"],
        include_payload=True,
        until=snapshot.trace_cutoff_ts,
    )
    recent_kb_queries: list[dict[str, Any]] = []
    for e in recent_kb_queries_raw:
        payload = e.get("payload")
        if isinstance(payload, dict):
            recent_kb_queries.append(_shrink_kb_query_payload(payload))

    return {
        "snapshot": {
            "snapshot_version": int(snapshot.snapshot_version),
            "trace_cutoff_ts": float(snapshot.trace_cutoff_ts),
            "feedback_id": snapshot.feedback_id,
            "feedback_updated_at": float(snapshot.feedback_updated_at),
            "final_output_event_id": snapshot.final_output_event_id,
        },
        "event_counts": event_counts,
        "latest_mem_search": latest_mem_search,
        "latest_memories_resolved": latest_memories_resolved,
        "latest_citations_resolved": latest_citations_resolved,
        "recent_kb_queries": recent_kb_queries,
        "latest_run_failed": latest_run_failed,
    }


def _extractor_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rb_extract_items",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["global", "orchestrator", "mof_expert", "tio2_expert"],
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["reasoningbank_item", "manual_note"],
                                },
                                "content": {"type": "string", "minLength": 1},
                                "extra": {"type": "object"},
                            },
                            "required": ["role", "type", "content"],
                        },
                    },
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "mem_id": {"type": "string", "minLength": 1},
                                "claim_id": {"type": "string", "minLength": 1},
                                "verdict": {"type": "string", "enum": ["support", "contradict", "irrelevant"]},
                                "notes": {"type": "string"},
                            },
                            "required": ["mem_id", "claim_id", "verdict"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    }


def _merge_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rb_merge_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "extra": {"type": "object"},
                },
                "required": ["content"],
            },
        },
    }


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _ensure_list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _best_effort_similarity(distance: float | None) -> float | None:
    """Convert Chroma distance into a similarity-like score.

    Assumption (common case): cosine distance in [0..2], where lower is better.
    Similarity ~= 1 - distance.
    """
    if distance is None:
        return None
    try:
        return 1.0 - float(distance)
    except Exception:
        return None


def _render_rbmem_claims_v1(*, topic: str | None, scope: str | None, claims: list[dict[str, Any]]) -> str:
    lines: list[str] = [RBMEM_CLAIMS_V1_HEADER]
    if topic:
        lines.append(f"TOPIC={topic}")
    if scope:
        lines.append(f"SCOPE={scope}")
    lines.append("CLAIMS_JSON=" + json.dumps(claims, ensure_ascii=False, separators=(",", ":")))
    lines.append("")
    return "\n".join(lines)


def _inject_facts_into_rbmem_claims_v1(
    *,
    content: str,
    facts_digest: dict[str, Any],
    run_id: str,
) -> str:
    """Inject/override claim.facts for all claims with system facts."""
    doc = parse_rbmem_claims_v1(content)
    claim_dicts: list[dict[str, Any]] = []
    for c in doc.claims:
        d = dict(c.__dict__)
        # Always override facts with system digest + stable identifiers.
        d["facts"] = {
            **(facts_digest or {}),
            "source_run_ids": [run_id],
        }
        # Normalize expected structures with defaults (avoid over-constraint, but keep stable keys).
        if not isinstance(d.get("inference"), dict):
            inf = str(d.get("inference") or "").strip()
            d["inference"] = {"summary": inf} if inf else {"summary": ""}
        if not isinstance(d.get("constraint"), dict):
            d["constraint"] = {"avoid": [], "allow_positive": False}
        if "allow_positive" not in d["constraint"]:
            d["constraint"]["allow_positive"] = False
        if "avoid" not in d["constraint"]:
            d["constraint"]["avoid"] = []
        d["conditions"] = _ensure_list_str(d.get("conditions"))
        d["limitations"] = _ensure_list_str(d.get("limitations"))
        d["support"] = _ensure_dict(d.get("support"))
        d["contra"] = _ensure_dict(d.get("contra"))
        d["support"]["run_ids"] = _ensure_list_str(d["support"].get("run_ids"))
        d["support"]["count"] = int(d["support"].get("count") or len(d["support"]["run_ids"]) or 0)
        d["contra"]["run_ids"] = _ensure_list_str(d["contra"].get("run_ids"))
        d["contra"]["count"] = int(d["contra"].get("count") or len(d["contra"]["run_ids"]) or 0)
        claim_dicts.append(d)

    out = _render_rbmem_claims_v1(topic=doc.topic, scope=doc.scope, claims=claim_dicts)
    # Re-validate after injection (hard constraints, esp. KB alias).
    validate_rbmem_claims_v1(out, max_claims=10, forbid_kb_alias=True)
    return out


def _apply_claim_verdicts_to_rbmem_claims_v1(
    *,
    content: str,
    run_id: str,
    verdicts: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Apply support/contra verdicts to claims and update statuses via a conservative state machine."""
    doc = parse_rbmem_claims_v1(content)
    claim_dicts: list[dict[str, Any]] = [dict(c.__dict__) for c in doc.claims]
    by_id: dict[str, dict[str, Any]] = {str(c.get("claim_id") or ""): c for c in claim_dicts}

    applied = 0
    skipped_unknown_claim = 0
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        cid = str(v.get("claim_id") or "").strip()
        verdict = str(v.get("verdict") or "").strip()
        if not cid or verdict not in {"support", "contradict", "irrelevant"}:
            continue
        claim = by_id.get(cid)
        if claim is None:
            skipped_unknown_claim += 1
            continue
        if verdict == "irrelevant":
            continue

        key = "support" if verdict == "support" else "contra"
        bucket = _ensure_dict(claim.get(key))
        run_ids = _ensure_list_str(bucket.get("run_ids"))
        if run_id not in run_ids:
            run_ids.append(run_id)
        bucket["run_ids"] = run_ids
        bucket["count"] = len(run_ids)
        claim[key] = bucket
        applied += 1

    # Conservative status update (claim-level; item can mix).
    deleted: list[str] = []
    for c in claim_dicts:
        status = str(c.get("status") or "").strip()
        support = _ensure_dict(c.get("support"))
        contra = _ensure_dict(c.get("contra"))
        support_ids = _ensure_list_str(support.get("run_ids"))
        contra_ids = _ensure_list_str(contra.get("run_ids"))
        support["run_ids"] = support_ids
        contra["run_ids"] = contra_ids
        support["count"] = int(support.get("count") or len(support_ids) or 0)
        contra["count"] = int(contra.get("count") or len(contra_ids) or 0)
        c["support"] = support
        c["contra"] = contra

        if status == "hypothesis" and int(support["count"]) >= 2 and int(contra["count"]) == 0:
            c["status"] = "conclusion"
        if status == "conclusion" and int(contra["count"]) >= 2:
            c["status"] = "hypothesis"
            lim = _ensure_list_str(c.get("limitations"))
            msg = "contradicted_by_multiple_runs; consider adding conditions/limitations"
            if msg not in lim:
                lim.append(msg)
            c["limitations"] = lim
        # Optional deletion rule: if a hypothesis is repeatedly contradicted and never supported, drop it.
        if status == "hypothesis" and int(support["count"]) == 0 and int(contra["count"]) >= 2:
            deleted.append(str(c.get("claim_id") or ""))

    if deleted and len(claim_dicts) > 1:
        claim_dicts = [c for c in claim_dicts if str(c.get("claim_id") or "") not in set(deleted)]

    out = _render_rbmem_claims_v1(topic=doc.topic, scope=doc.scope, claims=claim_dicts)
    validate_rbmem_claims_v1(out, max_claims=10, forbid_kb_alias=True)
    debug = {
        "applied": int(applied),
        "skipped_unknown_claim": int(skipped_unknown_claim),
        "deleted_claim_ids": deleted,
    }
    return out, debug


def _format_existing_memories(cfg: AppConfig, items: list[MemoryItem]) -> str:
    tpl = cfg.reasoningbank.context_template
    lines: list[str] = []
    for it in items:
        lines.append(
            render_template(
                tpl,
                {
                    "mem_id": it.mem_id,
                    "status": it.status,
                    "role": it.role,
                    "type": it.type,
                    "source_run_id": it.source_run_id or "",
                    "content": it.content,
                },
            ).strip()
        )
    return "\n\n".join([l for l in lines if l]).strip()


def _system_prompt(cfg: AppConfig) -> str:
    return "\n\n".join(
        [
            cfg.prompts.system_base.strip(),
            cfg.priors.system_description_md.strip(),
            cfg.priors.microenvironment_tio2_md.strip(),
            cfg.priors.microenvironment_mof_md.strip(),
        ]
    ).strip()


def _dry_run_extract_items(run_id: str) -> list[dict[str, Any]]:
    # Keep dry-run outputs compatible with the strict RBMEM_CLAIMS_V1 gate.
    # This ensures end-to-end pipeline tests stay representative of real RB items.
    return [
        {
            "role": "global",
            "type": "reasoningbank_item",
            "content": (
                "RBMEM_CLAIMS_V1\n"
                "TOPIC=DRY RUN synthetic RB item (pipeline validation)\n"
                "SCOPE=global\n"
                "CLAIMS_JSON="
                + json.dumps(
                    [
                        {
                            "claim_id": "c1",
                            "status": "fact",
                            "facts": {"source_run_ids": [run_id], "dry_run": True},
                            "inference": {"summary": "Dry-run: validates RB browse/learn/rollback plumbing."},
                            "constraint": {"avoid": [], "allow_positive": False},
                            "conditions": [],
                            "limitations": [],
                            "support": {"count": 0, "run_ids": []},
                            "contra": {"count": 0, "run_ids": []},
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ),
            "extra": {"dry_run": True, "confidence": 0.0, "tags": ["dry_run"]},
        },
        {
            "role": "orchestrator",
            "type": "reasoningbank_item",
            "content": (
                "RBMEM_CLAIMS_V1\n"
                "TOPIC=DRY RUN orchestrator RB item (pipeline validation)\n"
                "SCOPE=orchestrator\n"
                "CLAIMS_JSON="
                + json.dumps(
                    [
                        {
                            "claim_id": "c1",
                            "status": "fact",
                            "facts": {"source_run_ids": [run_id], "dry_run": True},
                            "inference": {"summary": "Dry-run: placeholder orchestrator memory. Do not use for science."},
                            "constraint": {"avoid": [], "allow_positive": False},
                            "conditions": [],
                            "limitations": [],
                            "support": {"count": 0, "run_ids": []},
                            "contra": {"count": 0, "run_ids": []},
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ),
            "extra": {"dry_run": True, "confidence": 0.0, "tags": ["dry_run"]},
        },
    ]


def learn_reasoningbank_for_run(
    store: SQLiteStore,
    *,
    rb: ReasoningBankStore,
    cfg: AppConfig,
    llm: OpenAICompatibleChatClient | None,
    run_id: str,
    rb_job_id: str,
) -> str:
    """Perform RB learn for a run and return the new delta_id.

    Implements:
      - strict rollback of previous applied deltas for this run
      - retrieval (existing memories)
      - extraction (LLM or dry-run)
      - consolidation (near-duplicate merge via LLM when available)
      - delta recording (SQLite)
    """
    rid = (run_id or "").strip()
    if not rid:
        raise RBLearnError("run_id is required.")
    if store.get_run(run_id=rid) is None:
        raise RBLearnError("Run not found.")

    trace_cutoff_ts = time.time()

    feedback_payload = store.get_feedback_for_run(run_id=rid)
    if feedback_payload is None:
        raise RBLearnError("Feedback not found (required for RB learn).")

    out_row = store.get_latest_event(run_id=rid, event_type="final_output")
    run_output_json: dict[str, Any] = {}
    final_output_event_id: str | None = None
    if out_row is not None:
        try:
            final_output_event_id = str(out_row["event_id"])
        except Exception:
            final_output_event_id = None
        try:
            run_output_json = json.loads(str(out_row["payload_json"]))
        except Exception:
            run_output_json = {}

    fb = feedback_payload.get("feedback") if isinstance(feedback_payload, dict) else None
    fb_id = str((fb or {}).get("feedback_id") or "").strip()
    fb_updated_at = float((fb or {}).get("updated_at") or 0.0)
    snapshot = RBLearnSnapshot(
        snapshot_version=1,
        run_id=rid,
        rb_job_id=rb_job_id,
        trace_cutoff_ts=float(trace_cutoff_ts),
        feedback_id=fb_id,
        feedback_updated_at=fb_updated_at,
        final_output_event_id=final_output_event_id,
    )

    budget = RBLearnDerefBudget(
        max_calls_total=int(cfg.reasoningbank.learn_deref_max_calls_total),
        max_full_calls=int(cfg.reasoningbank.learn_deref_max_full_calls),
        max_chars_total=int(cfg.reasoningbank.learn_deref_max_chars_total),
        excerpt_chars=int(cfg.reasoningbank.learn_deref_excerpt_chars),
        full_chars=int(cfg.reasoningbank.learn_deref_full_chars),
    )

    store.append_event(
        rid,
        "rb_learn_snapshot",
        {
            "ts": time.time(),
            "rb_job_id": rb_job_id,
            "snapshot": {
                "snapshot_version": int(snapshot.snapshot_version),
                "trace_cutoff_ts": float(snapshot.trace_cutoff_ts),
                "feedback_id": snapshot.feedback_id,
                "feedback_updated_at": float(snapshot.feedback_updated_at),
                "final_output_event_id": snapshot.final_output_event_id,
            },
            "budget": {
                "max_calls_total": int(budget.max_calls_total),
                "max_full_calls": int(budget.max_full_calls),
                "max_chars_total": int(budget.max_chars_total),
                "excerpt_chars": int(budget.excerpt_chars),
                "full_chars": int(budget.full_chars),
            },
            "policy": {
                "facts_only": True,
                "forbidden_trace_event_types": sorted(_FORBIDDEN_TRACE_EVENT_TYPES),
            },
        },
    )

    # Strict rollback: ensure the current RB state for this run is reverted to pre-learn before re-learning.
    deltas = store.list_rb_deltas_for_run(run_id=rid)
    applied_delta_ids = [str(d["delta_id"]) for d in deltas if str(d.get("status") or "") == "applied"]
    for did in applied_delta_ids:
        rollback_rb_delta(
            store,
            rb=rb,
            run_id=rid,
            delta_id=did,
            reason="auto_rollback_before_relearn",
        )

    # System FACTS digest (ground truth; used for both extraction + claim verdict updates).
    facts_digest = _build_facts_digest(
        run_id=rid,
        run_output_json=run_output_json,
        feedback_payload=feedback_payload,
    )

    dry_run = _env_bool("C2XC_RB_LEARN_DRY_RUN", False)
    extracted_items: list[dict[str, Any]]
    extracted_verdicts: list[dict[str, Any]] = []
    candidate_mem_ids: list[str] = []
    candidate_items: list[MemoryItem] = []

    if dry_run:
        extracted_items = _dry_run_extract_items(rid)
    else:
        if llm is None:
            raise RBLearnError("LLM is required for RB learn (set C2XC_RB_LEARN_DRY_RUN=1 for dry-run mode).")

        trace_digest = _build_run_trace_digest(store, snapshot=snapshot)

        # Candidate memories: bounded set of existing items we allow to be updated via claim verdicts.
        candidate_mem_ids = _collect_candidate_mem_ids(
            cfg=cfg,
            rb=rb,
            run_output_json=run_output_json,
            trace_digest=trace_digest,
            facts_digest=facts_digest,
        )
        candidate_items = rb.get_many(mem_ids=candidate_mem_ids, include_content=True)
        candidate_context = _format_existing_memories(cfg, candidate_items)

        prompt = render_template(
            cfg.reasoningbank.extract_prompt_template,
            {
                "run_id": rid,
                "run_output_json": json.dumps(run_output_json, ensure_ascii=False, indent=2),
                "feedback_json": json.dumps(feedback_payload, ensure_ascii=False, indent=2),
                "facts_digest_json": json.dumps(facts_digest, ensure_ascii=False, indent=2),
                "candidate_memories_context": candidate_context,
                "run_trace_digest_json": json.dumps(trace_digest, ensure_ascii=False, indent=2),
            },
        ).strip()

        system = _system_prompt(cfg)
        tools = _rb_learn_deref_tools_schema()
        deref_ctx = _RBLearnDerefContext(
            store=store,
            rb=rb,
            cfg=cfg,
            snapshot=snapshot,
            budget=budget,
            feedback_payload=feedback_payload,
            run_output_json=run_output_json,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        max_turns = 12
        turn = 0
        raw_content: str | None = None
        while True:
            store.append_event(
                rid,
                "rb_llm_request",
                {
                    "ts": time.time(),
                    "rb_job_id": rb_job_id,
                    "purpose": "extract",
                    "turn": int(turn),
                    "model": getattr(llm, "model", None),
                    "base_url": getattr(llm, "base_url", None),
                    "temperature": 0.2,
                    "response_schema": "rb_extract_items",
                    "snapshot": {
                        "snapshot_version": int(snapshot.snapshot_version),
                        "trace_cutoff_ts": float(snapshot.trace_cutoff_ts),
                        "feedback_id": snapshot.feedback_id,
                        "feedback_updated_at": float(snapshot.feedback_updated_at),
                        "final_output_event_id": snapshot.final_output_event_id,
                    },
                    "budget": {
                        "used_calls_total": int(budget.used_calls_total),
                        "used_full_calls": int(budget.used_full_calls),
                        "used_chars_total": int(budget.used_chars_total),
                        "max_calls_total": int(budget.max_calls_total),
                        "max_full_calls": int(budget.max_full_calls),
                        "max_chars_total": int(budget.max_chars_total),
                    },
                    # Store full system+prompt only on turn0 to avoid repeated large payloads.
                    "system": system if turn == 0 else None,
                    "prompt": prompt if turn == 0 else None,
                    "n_messages": len(messages),
                },
            )

            extra: dict[str, Any] = {"tools": tools, "tool_choice": "auto"}
            if not bool(getattr(llm, "enable_thinking", False)):
                extra["response_format"] = _extractor_response_format()
            try:
                raw = llm.chat_messages(messages=messages, temperature=0.2, extra=extra)
            except Exception as e:
                # Fallback: some providers do not accept response_format with tools.
                store.append_event(
                    rid,
                    "rb_llm_response",
                    {
                        "ts": time.time(),
                        "rb_job_id": rb_job_id,
                        "purpose": "extract",
                        "turn": int(turn),
                        "error": f"llm_call_failed_with_response_format: {e}",
                    },
                )
                raw = llm.chat_messages(messages=messages, temperature=0.2, extra={"tools": tools, "tool_choice": "auto"})

            store.append_event(
                rid,
                "rb_llm_response",
                {
                    "ts": time.time(),
                    "rb_job_id": rb_job_id,
                    "purpose": "extract",
                    "turn": int(turn),
                    "content": raw.content,
                    "raw": raw.raw,
                    "tool_calls": raw.tool_calls,
                },
            )

            if raw.tool_calls:
                # Append assistant tool-call message then tool outputs.
                messages.append({"role": "assistant", "content": raw.content or "", "tool_calls": raw.tool_calls})
                for tc in raw.tool_calls:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    name = str((fn or {}).get("name") or "").strip()
                    args_raw = (fn or {}).get("arguments") if isinstance(fn, dict) else ""
                    call_id = str(tc.get("id") or "").strip() or "tool_call"

                    parsed_args: dict[str, Any] = {}
                    if isinstance(args_raw, str) and args_raw.strip():
                        try:
                            obj = json.loads(args_raw)
                            parsed_args = obj if isinstance(obj, dict) else {}
                        except Exception:
                            parsed_args = {}

                    result = _execute_deref_tool(deref_ctx, name=name, args=parsed_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                turn += 1
                if turn >= max_turns:
                    raise RBLearnError("RB extractor exceeded maximum tool-calling turns.")
                continue

            raw_content = raw.content
            break

        if raw_content is None:
            raise RBLearnError("RB extractor produced no content.")
        try:
            obj = extract_first_json_object(raw_content)
        except JSONExtractionError as e:
            raise RBLearnError(f"RB extract output is not valid JSON: {e}") from e

        items_any = obj.get("items") if isinstance(obj, dict) else None
        extracted_items = items_any if isinstance(items_any, list) else []
        verdicts_any = obj.get("verdicts") if isinstance(obj, dict) else None
        extracted_verdicts = verdicts_any if isinstance(verdicts_any, list) else []

        # One-shot format repair (best-effort): if the extractor output is JSON but items are invalid,
        # ask the model to fix formatting only. This avoids polluting RB with malformed content.
        issues: list[str] = []
        for i, proposal in enumerate(extracted_items):
            if not isinstance(proposal, dict):
                continue
            typ = str(proposal.get("type") or "").strip() or "reasoningbank_item"
            content = str(proposal.get("content") or "").strip()
            if not content:
                issues.append(f"items[{i}].content: empty")
                continue
            if typ == "reasoningbank_item":
                if not is_rbmem_claims_v1(content):
                    issues.append(f"items[{i}].content: missing RBMEM_CLAIMS_V1 header")
                else:
                    try:
                        validate_rbmem_claims_v1(content, max_claims=10, forbid_kb_alias=True)
                    except RBMemClaimsV1ValidationError as e:
                        detail = "; ".join(e.issues[:4]) if e.issues else str(e)
                        issues.append(f"items[{i}].content: invalid RBMEM_CLAIMS_V1 ({detail})")

        if issues:
            fix_prompt = (
                "Your previous JSON output was rejected by validation.\n"
                "Fix FORMAT ONLY and return a single JSON object with the same keys: items, verdicts.\n"
                "- Do not add commentary.\n"
                "- Ensure reasoningbank_item.content is valid RBMEM_CLAIMS_V1.\n"
                "- Do not include KB aliases like [C12].\n"
                "- Do not include next-step experimental instructions in constraint.\n"
                f"Validation issues: {json.dumps(issues[:12], ensure_ascii=False)}\n"
                "Previous output (for editing):\n"
                f"{raw_content}\n"
            )
            store.append_event(
                rid,
                "rb_llm_request",
                {
                    "ts": time.time(),
                    "rb_job_id": rb_job_id,
                    "purpose": "extract_format_fix",
                    "model": getattr(llm, "model", None),
                    "base_url": getattr(llm, "base_url", None),
                    "temperature": 0.0,
                },
            )
            fixed = llm.chat_messages(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": fix_prompt}],
                temperature=0.0,
                extra={},
            )
            store.append_event(
                rid,
                "rb_llm_response",
                {
                    "ts": time.time(),
                    "rb_job_id": rb_job_id,
                    "purpose": "extract_format_fix",
                    "content": fixed.content,
                    "raw": fixed.raw,
                    "tool_calls": fixed.tool_calls,
                },
            )
            try:
                fixed_obj = extract_first_json_object(fixed.content)
                fixed_items_any = fixed_obj.get("items") if isinstance(fixed_obj, dict) else None
                extracted_items = fixed_items_any if isinstance(fixed_items_any, list) else extracted_items
                fixed_verdicts_any = fixed_obj.get("verdicts") if isinstance(fixed_obj, dict) else None
                extracted_verdicts = fixed_verdicts_any if isinstance(fixed_verdicts_any, list) else extracted_verdicts
            except JSONExtractionError:
                # Keep original output if the format-fix attempt fails.
                pass

    # Record delta ops for this learn job (includes verdict-driven updates + new additions/merges).
    ops: list[dict[str, Any]] = []

    # Phase A: apply claim verdict updates to candidate memories (bounded set).
    candidate_by_id: dict[str, MemoryItem] = {it.mem_id: it for it in candidate_items}
    verdicts_by_mem: dict[str, list[dict[str, Any]]] = {}
    for v in extracted_verdicts:
        if not isinstance(v, dict):
            continue
        mem_id = str(v.get("mem_id") or "").strip()
        if not mem_id:
            continue
        verdicts_by_mem.setdefault(mem_id, []).append(v)

    if verdicts_by_mem:
        verdict_applied: dict[str, Any] = {"updated": 0, "skipped_non_candidate": 0, "skipped_invalid": 0}
        for mem_id, vs in verdicts_by_mem.items():
            existing = candidate_by_id.get(mem_id)
            if existing is None:
                verdict_applied["skipped_non_candidate"] = int(verdict_applied["skipped_non_candidate"]) + 1
                continue
            if not is_rbmem_claims_v1(existing.content):
                verdict_applied["skipped_invalid"] = int(verdict_applied["skipped_invalid"]) + 1
                continue

            try:
                new_content, debug = _apply_claim_verdicts_to_rbmem_claims_v1(
                    content=existing.content,
                    run_id=rid,
                    verdicts=vs,
                )
            except Exception:
                verdict_applied["skipped_invalid"] = int(verdict_applied["skipped_invalid"]) + 1
                continue

            if new_content.strip() == existing.content.strip():
                continue

            before = existing
            after = rb.upsert(
                mem_id=before.mem_id,
                status=before.status,
                role=before.role,
                type=before.type,
                content=new_content,
                source_run_id=before.source_run_id,
                schema_version=max(int(before.schema_version), 2),
                extra=dict(before.extra or {}),
                preserve_created_at=True,
            )
            _sync_rb_mem_index(store, after)
            store.append_mem_edit_log(
                mem_id=after.mem_id,
                actor="rb_learn",
                reason=f"claim_verdicts:{rb_job_id}",
                before=_memory_to_dict(before),
                after=_memory_to_dict(after),
                extra={"debug": debug},
            )
            ops.append(
                {
                    "op": "update",
                    "mem_id": after.mem_id,
                    "before": _memory_to_dict(before),
                    "after": _memory_to_dict(after),
                    "reason": "claim_verdicts",
                    "debug": debug,
                }
            )
            verdict_applied["updated"] = int(verdict_applied["updated"]) + 1

        store.append_event(
            rid,
            "rb_claim_verdicts_applied",
            {
                "ts": time.time(),
                "rb_job_id": rb_job_id,
                "summary": verdict_applied,
                "n_verdicts": sum(len(v) for v in verdicts_by_mem.values()),
                "n_candidate_items": len(candidate_items),
            },
        )

    # Consolidation + apply changes, record delta ops.
    for proposal in extracted_items:
        if not isinstance(proposal, dict):
            continue
        role = str(proposal.get("role") or "").strip() or "global"
        typ = str(proposal.get("type") or "").strip() or "reasoningbank_item"
        content = str(proposal.get("content") or "").strip()
        extra = _ensure_dict(proposal.get("extra"))
        if not content:
            continue

        # Hard gate + system FACTS injection for structured RB items.
        if typ == "reasoningbank_item":
            if not is_rbmem_claims_v1(content):
                store.append_event(
                    rid,
                    "rb_item_rejected",
                    {
                        "ts": time.time(),
                        "rb_job_id": rb_job_id,
                        "reason": "missing_rbmem_claims_v1_header",
                        "role": role,
                        "type": typ,
                    },
                )
                continue
            try:
                content = _inject_facts_into_rbmem_claims_v1(
                    content=content,
                    facts_digest=facts_digest,
                    run_id=rid,
                )
            except (RBMemClaimsV1ParseError, RBMemClaimsV1ValidationError) as e:
                store.append_event(
                    rid,
                    "rb_item_rejected",
                    {
                        "ts": time.time(),
                        "rb_job_id": rb_job_id,
                        "reason": "invalid_rbmem_claims_v1",
                        "role": role,
                        "type": typ,
                        "error": str(e),
                    },
                )
                continue

        # Find near-duplicate candidate in active memories within same role/type.
        dup_candidates = rb.query(query=content, n_results=3, status=["active"], role=[role], type=[typ])
        chosen_existing: MemoryItem | None = None
        chosen_distance: float | None = None
        if dup_candidates:
            first = dup_candidates[0]
            chosen_existing = cast(MemoryItem, first["item"])
            chosen_distance = first.get("distance")

        similarity = _best_effort_similarity(chosen_distance)
        threshold = float(cfg.reasoningbank.near_duplicate_threshold)
        dup_debug: dict[str, Any] | None = None
        if chosen_existing is not None and similarity is not None:
            dup_debug = {
                "candidate_mem_id": chosen_existing.mem_id,
                "distance": float(chosen_distance) if chosen_distance is not None else None,
                "similarity": float(similarity),
                "threshold": float(threshold),
                "assumption": "similarity ~= 1 - distance",
            }

        if chosen_existing is not None and similarity is not None and similarity >= threshold:
            # Merge path: use LLM merge prompt if available; fallback to heuristic.
            merged_content = content
            merged_extra: dict[str, Any] = dict(chosen_existing.extra or {})
            merge_used = False

            if not dry_run and llm is not None:
                merge_prompt = render_template(
                    cfg.reasoningbank.merge_prompt_template,
                    {
                        "existing_item_json": json.dumps(_memory_to_dict(chosen_existing), ensure_ascii=False, indent=2),
                        "new_item_json": json.dumps(proposal, ensure_ascii=False, indent=2),
                    },
                ).strip()
                system = _system_prompt(cfg)
                store.append_event(
                    rid,
                    "rb_llm_request",
                    {
                        "ts": time.time(),
                        "rb_job_id": rb_job_id,
                        "purpose": "merge",
                        "mem_id": chosen_existing.mem_id,
                        "model": getattr(llm, "model", None),
                        "base_url": getattr(llm, "base_url", None),
                        "temperature": 0.0,
                        "response_schema": "rb_merge_result",
                        "system": system,
                        "prompt": merge_prompt,
                    },
                )
                merge_extra: dict[str, Any] = {}
                if not bool(getattr(llm, "enable_thinking", False)):
                    merge_extra["response_format"] = _merge_response_format()
                try:
                    raw_merge = llm.chat_messages(
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": merge_prompt}],
                        temperature=0.0,
                        extra=merge_extra,
                    )
                except Exception as e:
                    store.append_event(
                        rid,
                        "rb_llm_response",
                        {
                            "ts": time.time(),
                            "rb_job_id": rb_job_id,
                            "purpose": "merge",
                            "mem_id": chosen_existing.mem_id,
                            "error": f"llm_call_failed_with_response_format: {e}",
                        },
                    )
                    raw_merge = llm.chat_messages(
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": merge_prompt}],
                        temperature=0.0,
                        extra={},
                    )
                store.append_event(
                    rid,
                    "rb_llm_response",
                    {
                        "ts": time.time(),
                        "rb_job_id": rb_job_id,
                        "purpose": "merge",
                        "mem_id": chosen_existing.mem_id,
                        "content": raw_merge.content,
                        "raw": raw_merge.raw,
                        "tool_calls": raw_merge.tool_calls,
                    },
                )
                try:
                    merged_obj = extract_first_json_object(raw_merge.content)
                    merged_content = str(merged_obj.get("content") or "").strip()
                    if not merged_content:
                        merged_content = content
                    if merged_content == "NOT_DUPLICATE":
                        merged_content = content
                    else:
                        merge_used = True
                    merged_extra.update(_ensure_dict(merged_obj.get("extra")))
                except JSONExtractionError:
                    merged_content = content

            if not merge_used:
                # Heuristic: keep the longer statement and record the proposal in extra for traceability.
                if len(chosen_existing.content.strip()) >= len(content):
                    merged_content = chosen_existing.content
                merged_extra = dict(chosen_existing.extra or {})
                merged_extra.setdefault("merged_from", []).append(
                    {
                        "source_run_id": rid,
                        "proposal": content,
                    }
                )

            # Safety: avoid poisoning the store with invalid structured content.
            if chosen_existing.type == "reasoningbank_item":
                if not is_rbmem_claims_v1(merged_content):
                    merged_content = chosen_existing.content
                    merge_used = False
                else:
                    try:
                        validate_rbmem_claims_v1(merged_content, max_claims=10, forbid_kb_alias=True)
                    except RBMemClaimsV1ValidationError:
                        merged_content = chosen_existing.content
                        merge_used = False

            before = chosen_existing
            after = rb.upsert(
                mem_id=chosen_existing.mem_id,
                status=chosen_existing.status,
                role=chosen_existing.role,
                type=chosen_existing.type,
                content=merged_content,
                source_run_id=chosen_existing.source_run_id,
                schema_version=chosen_existing.schema_version,
                extra=merged_extra,
                preserve_created_at=True,
            )
            _sync_rb_mem_index(store, after)

            store.append_mem_edit_log(
                mem_id=after.mem_id,
                actor="rb_learn",
                reason=f"learn_merge:{rb_job_id}",
                before=_memory_to_dict(before),
                after=_memory_to_dict(after),
                extra={"near_duplicate": dup_debug, "merge_used": bool(merge_used)},
            )

            ops.append(
                {
                    "op": "update",
                    "mem_id": after.mem_id,
                    "before": _memory_to_dict(before),
                    "after": _memory_to_dict(after),
                    "near_duplicate": dup_debug,
                    "merge_used": bool(merge_used),
                }
            )
            continue

        # Add as a new item (keep conflicts by default).
        after = rb.upsert(
            mem_id=None,
            status="active",
            role=role,
            type=typ,
            content=content,
            source_run_id=rid,
            schema_version=2 if typ == "reasoningbank_item" else 1,
            extra={
                **extra,
                "source_run_id": rid,
                "strategy_version": cfg.reasoningbank.strategy_version,
            },
            preserve_created_at=True,
        )
        _sync_rb_mem_index(store, after)

        store.append_mem_edit_log(
            mem_id=after.mem_id,
            actor="rb_learn",
            reason=f"learn_add:{rb_job_id}",
            before={},
            after=_memory_to_dict(after),
            extra={"near_duplicate_checked": dup_debug},
        )

        ops.append(
            {
                "op": "add",
                "mem_id": after.mem_id,
                "after": _memory_to_dict(after),
                "near_duplicate_checked": dup_debug,
            }
        )

    if not ops:
        # Still record an empty delta for auditability.
        ops = []

    delta = store.create_rb_delta(
        run_id=rid,
        ops=ops,
        schema_version=1,
        extra={
            "rb_job_id": rb_job_id,
            "strategy_version": cfg.reasoningbank.strategy_version,
            "dry_run": dry_run,
        },
    )

    store.append_event(
        rid,
        "rb_learn_completed",
        {"rb_job_id": rb_job_id, "delta_id": delta.delta_id, "n_ops": len(ops), "dry_run": dry_run},
    )
    return delta.delta_id


def safe_execute_rb_learn_job(
    store: SQLiteStore,
    *,
    rb: ReasoningBankStore,
    cfg: AppConfig,
    llm: OpenAICompatibleChatClient | None,
    run_id: str,
    rb_job_id: str,
) -> str | None:
    try:
        return learn_reasoningbank_for_run(
            store,
            rb=rb,
            cfg=cfg,
            llm=llm,
            run_id=run_id,
            rb_job_id=rb_job_id,
        )
    except Exception as e:
        store.append_event(
            run_id,
            "rb_learn_failed",
            {"rb_job_id": rb_job_id, "error": str(e), "traceback": traceback.format_exc()},
        )
        return None
