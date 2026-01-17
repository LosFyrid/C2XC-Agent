from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from src.storage.reasoningbank_store import MemoryItem
from src.tools.citation_aliases import (
    AliasedKBChunk,
    extract_citation_aliases,
    extract_memory_ids,
    resolve_aliases,
)
from src.utils.json_extract import JSONExtractionError, extract_first_json_object
from src.utils.template import render_template

from .node import Node, RecapInfo
from .state import RecapState


class RecapError(RuntimeError):
    pass


def _now_ts() -> float:
    return time.time()


_ALLOWED_ROLES = {"orchestrator", "mof_expert", "tio2_expert"}
_ALLOWED_KB_NAMES = {"kb_principles", "kb_modulation"}
_ALLOWED_KB_MODES = {"mix", "local", "global", "hybrid", "naive"}
_ALLOWED_MEM_ROLES = {"global", "orchestrator", "mof_expert", "tio2_expert"}
_ALLOWED_MEM_STATUSES = {"active", "archived"}
_ALLOWED_MEM_TYPES = {"reasoningbank_item", "manual_note"}


# Structured output schema for ReCAP planning/refinement calls.
# Uses OpenAI-compatible `response_format` with a JSON Schema.
_RECAP_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "recap_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "think": {"type": "string"},
                "subtasks": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "task"},
                                    "role": {"type": "string", "enum": sorted(_ALLOWED_ROLES)},
                                    "task": {"type": "string"},
                                },
                                "required": ["type", "task"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "kb_search"},
                                    "kb_name": {"type": "string", "enum": sorted(_ALLOWED_KB_NAMES)},
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer", "minimum": 1},
                                    "mode": {"type": "string", "enum": sorted(_ALLOWED_KB_MODES)},
                                },
                                "required": ["type", "kb_name", "query"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "kb_get"},
                                    "alias": {"type": "string"},
                                },
                                "required": ["type", "alias"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "kb_list"},
                                    "limit": {"type": "integer", "minimum": 1},
                                },
                                "required": ["type"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "mem_search"},
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer", "minimum": 1},
                                    "role": {"type": "string", "enum": sorted(_ALLOWED_MEM_ROLES)},
                                    "status": {"type": "string", "enum": sorted(_ALLOWED_MEM_STATUSES)},
                                    "mem_type": {"type": "string", "enum": sorted(_ALLOWED_MEM_TYPES)},
                                },
                                "required": ["type", "query"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "mem_get"},
                                    "mem_id": {"type": "string"},
                                },
                                "required": ["type", "mem_id"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"const": "mem_list"},
                                    "limit": {"type": "integer", "minimum": 1},
                                },
                                "required": ["type"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {"type": {"const": "generate_recipes"}},
                                "required": ["type"],
                            },
                        ]
                    },
                },
                "result": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "object"},
                        {"type": "array"},
                    ]
                },
            },
            "required": ["think", "subtasks"],
        },
    },
}


def _recipes_response_format(*, recipes_per_run: int) -> dict[str, Any]:
    """JSON Schema for the final `generate_recipes` output.

    Keep this intentionally minimal and machine-consumable:
    - enforce object shape + required fields
    - enforce exact recipe count
    - do NOT enforce chemistry semantics in code (e.g., "-COOH" substring checks)
    """
    n = int(recipes_per_run)
    if n < 1:
        n = 1

    recipe_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "M1": {"type": "string", "minLength": 1},
            "M2": {"type": "string", "minLength": 1},
            "atomic_ratio": {"type": "string", "minLength": 1},
            "small_molecule_modifier": {"type": "string", "minLength": 1},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["M1", "M2", "atomic_ratio", "small_molecule_modifier", "rationale"],
    }

    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recipes": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": recipe_schema,
            },
            "overall_notes": {"type": "string"},
        },
        "required": ["recipes"],
    }

    return {
        "type": "json_schema",
        "json_schema": {"name": "generate_recipes_output", "strict": True, "schema": schema},
    }


def _parse_subtask(item: Any) -> dict[str, Any]:
    """Parse and minimally validate a single structured subtask."""
    if not isinstance(item, dict):
        raise JSONExtractionError(f"Invalid subtask: expected object, got {type(item).__name__}")
    stype = str(item.get("type", "")).strip()
    if not stype:
        raise JSONExtractionError("Invalid subtask: missing 'type'")

    if stype == "task":
        task = str(item.get("task", "")).strip()
        if not task:
            raise JSONExtractionError("Invalid task subtask: missing non-empty 'task'")
        role = str(item.get("role", "orchestrator") or "orchestrator").strip()
        if role not in _ALLOWED_ROLES:
            raise JSONExtractionError(
                f"Invalid task subtask: role must be one of {sorted(_ALLOWED_ROLES)}, got {role!r}"
            )
        return {"type": "task", "role": role, "task": task}

    if stype == "kb_search":
        kb_name = str(item.get("kb_name", "")).strip()
        if kb_name not in _ALLOWED_KB_NAMES:
            raise JSONExtractionError(
                f"Invalid kb_search: kb_name must be one of {sorted(_ALLOWED_KB_NAMES)}, got {kb_name!r}"
            )
        query = str(item.get("query", "")).strip()
        if not query:
            raise JSONExtractionError("Invalid kb_search: missing non-empty 'query'")

        top_k: int | None = None
        if item.get("top_k") is not None:
            try:
                top_k = int(item.get("top_k"))
            except Exception as e:
                raise JSONExtractionError(f"Invalid kb_search.top_k: {item.get('top_k')!r}") from e
            if top_k < 1:
                raise JSONExtractionError(f"Invalid kb_search.top_k: must be >=1, got {top_k}")

        mode: str | None = None
        if item.get("mode") is not None:
            mode = str(item.get("mode") or "").strip()
            if mode and mode not in _ALLOWED_KB_MODES:
                raise JSONExtractionError(
                    f"Invalid kb_search.mode: must be one of {sorted(_ALLOWED_KB_MODES)}, got {mode!r}"
                )

        out: dict[str, Any] = {"type": "kb_search", "kb_name": kb_name, "query": query}
        if top_k is not None:
            out["top_k"] = top_k
        if mode:
            out["mode"] = mode
        return out

    if stype == "kb_get":
        alias = str(item.get("alias", "")).strip()
        if alias.startswith("[") and alias.endswith("]"):
            alias = alias[1:-1].strip()
        if not alias:
            raise JSONExtractionError("Invalid kb_get: missing non-empty 'alias'")
        return {"type": "kb_get", "alias": alias}

    if stype == "kb_list":
        limit: int | None = None
        if item.get("limit") is not None:
            try:
                limit = int(item.get("limit"))
            except Exception as e:
                raise JSONExtractionError(f"Invalid kb_list.limit: {item.get('limit')!r}") from e
        out = {"type": "kb_list"}
        if limit is not None:
            out["limit"] = limit
        return out

    if stype == "mem_search":
        query = str(item.get("query", "")).strip()
        if not query:
            raise JSONExtractionError("Invalid mem_search: missing non-empty 'query'")

        top_k: int | None = None
        if item.get("top_k") is not None:
            try:
                top_k = int(item.get("top_k"))
            except Exception as e:
                raise JSONExtractionError(f"Invalid mem_search.top_k: {item.get('top_k')!r}") from e
            if top_k < 1:
                raise JSONExtractionError(f"Invalid mem_search.top_k: must be >=1, got {top_k}")

        role: str | None = None
        if item.get("role") is not None:
            role = str(item.get("role") or "").strip()
            if role and role not in _ALLOWED_MEM_ROLES:
                raise JSONExtractionError(
                    f"Invalid mem_search.role: must be one of {sorted(_ALLOWED_MEM_ROLES)}, got {role!r}"
                )

        status: str | None = None
        if item.get("status") is not None:
            status = str(item.get("status") or "").strip()
            if status and status not in _ALLOWED_MEM_STATUSES:
                raise JSONExtractionError(
                    f"Invalid mem_search.status: must be one of {sorted(_ALLOWED_MEM_STATUSES)}, got {status!r}"
                )

        mem_type: str | None = None
        if item.get("mem_type") is not None:
            mem_type = str(item.get("mem_type") or "").strip()
            if mem_type and mem_type not in _ALLOWED_MEM_TYPES:
                raise JSONExtractionError(
                    f"Invalid mem_search.mem_type: must be one of {sorted(_ALLOWED_MEM_TYPES)}, got {mem_type!r}"
                )

        out: dict[str, Any] = {"type": "mem_search", "query": query}
        if top_k is not None:
            out["top_k"] = top_k
        if role:
            out["role"] = role
        if status:
            out["status"] = status
        if mem_type:
            out["mem_type"] = mem_type
        return out

    if stype == "mem_get":
        mem_id = str(item.get("mem_id", "")).strip()
        if not mem_id:
            raise JSONExtractionError("Invalid mem_get: missing non-empty 'mem_id'")
        return {"type": "mem_get", "mem_id": mem_id}

    if stype == "mem_list":
        limit: int | None = None
        if item.get("limit") is not None:
            try:
                limit = int(item.get("limit"))
            except Exception as e:
                raise JSONExtractionError(f"Invalid mem_list.limit: {item.get('limit')!r}") from e
        out = {"type": "mem_list"}
        if limit is not None:
            out["limit"] = limit
        return out

    if stype == "generate_recipes":
        return {"type": "generate_recipes"}

    raise JSONExtractionError(
        "Invalid subtask.type: must be one of "
        "['task','kb_search','kb_get','kb_list','mem_search','mem_get','mem_list','generate_recipes'], "
        f"got {stype!r}"
    )


def _as_subtasks_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise JSONExtractionError(f"Invalid 'subtasks': expected array, got {type(value).__name__}")
    out: list[dict[str, Any]] = []
    for item in value:
        out.append(_parse_subtask(item))
    return out


def _parse_recap_info(text: str) -> RecapInfo:
    obj = extract_first_json_object(text)
    think = str(obj.get("think", "")).strip()
    subtasks = _as_subtasks_list(obj.get("subtasks", []))
    raw_result = obj.get("result", "")
    if isinstance(raw_result, (dict, list)):
        result = json.dumps(raw_result, ensure_ascii=False, indent=2)
    else:
        result = str(raw_result or "").strip()
    return RecapInfo(think=think, subtasks=subtasks, result=result)


def _format_kb_observation(
    *,
    kb_name: str,
    query: str,
    mode: str,
    top_k: int,
    aliased: list[Any],
) -> str:
    # We keep this simple and LLM-friendly: aliases + source + content.
    # Canonical refs are still stored in trace; we do not need to show kb:* here.
    lines: list[str] = []
    lines.append(f"KB search results: kb={kb_name} mode={mode} top_k={top_k}")
    lines.append(f'Query: "{query}"')
    lines.append("")
    if not aliased:
        lines.append("(no results)")
        return "\n".join(lines).strip()

    for a in aliased:
        # a is AliasedKBChunk, but keep typing loose to avoid import cycles here
        lines.append(f"[{a.alias}] source={a.source}")
        lines.append(a.content)
        lines.append("")

    return "\n".join(lines).strip()


def _trim_history(history: list[dict[str, Any]], *, max_rounds: int) -> list[dict[str, Any]]:
    """Trim message history to a sliding window of K rounds.

    We keep the *first* user message (global request) pinned, and then keep the
    last 2*K messages after that (user+assistant pairs).
    """
    if max_rounds <= 0:
        return history
    if len(history) <= 1:
        return history

    pinned = history[0:1]
    tail = history[1:]
    limit = max_rounds * 2
    if len(tail) <= limit:
        return pinned + tail
    return pinned + tail[-limit:]


def _format_subtasks_for_prompt(subtasks: list[dict[str, Any]]) -> str:
    if not subtasks:
        return "No remaining subtasks."
    return json.dumps(subtasks, ensure_ascii=False, indent=2)


@dataclass
class _RuntimeState:
    state: RecapState
    node_ptr: Node
    depth: int
    steps: int

    latest_obs: str
    remaining_subtasks: list[dict[str, Any]]

    done_task_name: str
    done_task_result: str
    previous_stage_task_name: str
    previous_stage_think: str

    # Citation aliases are GLOBAL within a single run, across multiple kb_search calls.
    # This makes multi-search evidence traceable and allows citing older evidence.
    kb_alias_map: dict[str, str]  # alias -> canonical kb ref (kb:...)
    kb_ref_to_alias: dict[str, str]  # canonical kb ref -> alias
    kb_all_aliased_chunks: list[AliasedKBChunk]  # unique, in first-seen order
    kb_alias_to_chunk: dict[str, AliasedKBChunk]  # alias -> chunk content/source
    kb_next_index: int  # next numeric suffix for alias allocation

    # "Focus" set: evidence the agent has explicitly used (via inline citations) or
    # re-opened (via kb_get). Used to keep generate_recipes prompts small.
    kb_focus_aliases: list[str]  # ordered, unique
    kb_focus_seen: set[str]

    # Last kb_search aliases (fallback when nothing was focused yet)
    last_kb_search_aliases: list[str]

    # Memory registry (ReasoningBank): mem_id -> memory item.
    mem_all_items: list[MemoryItem]  # unique, in first-seen order
    mem_id_to_item: dict[str, MemoryItem]

    # Focused memory ids (cited inline as mem:<id> or opened via mem_get).
    mem_focus_ids: list[str]
    mem_focus_seen: set[str]

    # Last mem_search results (fallback when nothing was focused yet)
    last_mem_search_ids: list[str]


def _merge_focus_aliases(rt: _RuntimeState, aliases: list[str]) -> None:
    for a in aliases:
        alias = (a or "").strip()
        if not alias:
            continue
        if alias in rt.kb_focus_seen:
            continue
        rt.kb_focus_seen.add(alias)
        rt.kb_focus_aliases.append(alias)


def _merge_focus_mem_ids(rt: _RuntimeState, mem_ids: list[str]) -> None:
    for mid in mem_ids:
        mem_id = (mid or "").strip()
        if not mem_id:
            continue
        if mem_id in rt.mem_focus_seen:
            continue
        rt.mem_focus_seen.add(mem_id)
        rt.mem_focus_ids.append(mem_id)


class RecapEngine:
    """Domain-adapted ReCAP engine.

    Primitive actions:
      - kb_search
      - kb_get
      - kb_list
      - generate_recipes

    Composite subtasks ("task") specify an explicit role (orchestrator/mof_expert/tio2_expert),
    but all execution happens in one shared conversation history, consistent with the paper.
    """

    def run(self, ctx: Any, *, user_request: str) -> tuple[dict[str, Any], dict[str, str], list[str]]:
        ctx.check_cancelled()
        if ctx.llm is None:
            raise RecapError("LLM not configured.")
        if ctx.kbs is None:
            raise RecapError("KB not configured.")
        if getattr(ctx, "config", None) is None:
            raise RecapError("App config missing from AgentContext.")

        cfg = ctx.config
        system_prompt = "\n\n".join(
            [
                cfg.prompts.system_base.strip(),
                cfg.priors.system_description_md.strip(),
                cfg.priors.microenvironment_tio2_md.strip(),
                cfg.priors.microenvironment_mof_md.strip(),
            ]
        ).strip()

        root = Node(task_name="Generate catalyst recipe recommendations.", role="orchestrator")

        # Shared conversation history (system is supplied per call).
        history: list[dict[str, Any]] = []
        history.append(
            {
                "role": "user",
                "content": (
                    "User request:\n"
                    f"{user_request}\n\n"
                    f"recipes_per_run={ctx.recipes_per_run}\n"
                    "You must retrieve evidence before generate_recipes (kb_search for literature and/or mem_search for memories)."
                ),
            }
        )

        rt = _RuntimeState(
            state=RecapState.DOWN,
            node_ptr=root,
            depth=0,
            steps=0,
            latest_obs="",
            remaining_subtasks=[],
            done_task_name="",
            done_task_result="",
            previous_stage_task_name="",
            previous_stage_think="",
            kb_alias_map={},
            kb_ref_to_alias={},
            kb_all_aliased_chunks=[],
            kb_alias_to_chunk={},
            kb_next_index=1,
            kb_focus_aliases=[],
            kb_focus_seen=set(),
            last_kb_search_aliases=[],
            mem_all_items=[],
            mem_id_to_item={},
            mem_focus_ids=[],
            mem_focus_seen=set(),
            last_mem_search_ids=[],
        )

        def _resolve_mem_tokens(mem_tokens: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
            """Resolve mem tokens (full UUIDs or hex prefixes) into full mem_ids in the run registry.

            Returns:
              - resolved_mem_ids: list[str] full mem_ids (de-duplicated, first-seen order)
              - invalid_tokens: list[str] tokens that match nothing
              - ambiguous: dict[prefix] -> list[matching_full_ids]
            """
            resolved: list[str] = []
            seen: set[str] = set()
            invalid: list[str] = []
            ambiguous: dict[str, list[str]] = {}

            if not mem_tokens:
                return resolved, invalid, ambiguous

            known_ids: list[str] = list(rt.mem_id_to_item.keys())
            known_ids_lower = {mid.lower(): mid for mid in known_ids}

            for tok_raw in mem_tokens:
                tok = (tok_raw or "").strip()
                if not tok:
                    continue
                tok_norm = tok.lower()

                # Full UUID token path.
                if "-" in tok_norm:
                    mid = known_ids_lower.get(tok_norm)
                    if mid is None:
                        invalid.append(tok_raw)
                        continue
                    if mid not in seen:
                        seen.add(mid)
                        resolved.append(mid)
                    continue

                # Prefix token path (hex without hyphens).
                matches = [mid for mid in known_ids if mid.lower().startswith(tok_norm)]
                if not matches:
                    invalid.append(tok_raw)
                    continue
                if len(matches) > 1:
                    ambiguous[tok_raw] = matches
                    continue
                mid = matches[0]
                if mid not in seen:
                    seen.add(mid)
                    resolved.append(mid)

            return resolved, invalid, ambiguous

        def _validate_and_resolve_final_output(
            obj: dict[str, Any],
            *,
            context: str,
        ) -> tuple[dict[str, str], list[str], list[str], list[str]]:
            """Validate a final recipes JSON object and resolve citations/memories.

            Returns:
              - citations: alias -> canonical kb ref (kb:...)
              - resolved_mem_ids: list[str] full mem_ids cited
              - used_aliases: list[str] aliases found in output (order-preserving, de-duplicated)
              - mem_tokens: list[str] mem tokens found in output (UUIDs or prefixes)

            Raises RecapError if validation fails.
            """
            recipes = obj.get("recipes")
            if not isinstance(recipes, list) or len(recipes) != int(ctx.recipes_per_run):
                raise RecapError(f"Invalid recipe count. Expected exactly {ctx.recipes_per_run}.")

            # Validate per-recipe citation presence (KB alias [C1] or memory mem:<id>).
            missing_citations = 0
            for r in recipes:
                if not isinstance(r, dict):
                    continue
                rationale = str(r.get("rationale") or "")
                if not extract_citation_aliases(rationale) and not extract_memory_ids(rationale):
                    missing_citations += 1
            if missing_citations:
                raise RecapError(
                    "Each recipe rationale must include at least one inline citation. "
                    "Use either a KB alias like [C2] or a memory id like mem:<uuid>."
                )

            text_dump = json.dumps(obj, ensure_ascii=False)
            used_aliases = extract_citation_aliases(text_dump)
            mem_tokens = extract_memory_ids(text_dump)

            if not used_aliases and not mem_tokens:
                raise RecapError("No citations found in final output (KB alias [C#] or mem:<id>).")

            # Resolve KB aliases.
            try:
                citations = resolve_aliases(used_aliases, rt.kb_alias_map)
            except KeyError as e:
                raise RecapError(
                    f"Unknown citation alias in output: {e}. "
                    "Only cite aliases that exist in the run evidence registry (see index / kb_list)."
                ) from e

            # Resolve + validate memory tokens.
            resolved_mem_ids, invalid_tokens, ambiguous = _resolve_mem_tokens(mem_tokens)
            if invalid_tokens:
                raise RecapError(
                    "Unknown mem:<id> cited in output. You may only cite mem:<id> values that exist in the run memory "
                    "registry (use mem_search first). Unknown: "
                    f"{invalid_tokens}"
                )
            if ambiguous:
                # Avoid dumping huge lists; show only a few candidates per token.
                preview = {k: v[:5] for k, v in ambiguous.items()}
                raise RecapError(
                    "Ambiguous mem:<prefix> cited in output. Use a longer prefix or the full UUID. "
                    f"Ambiguous: {preview}"
                )

            archived: list[str] = []
            for mid in resolved_mem_ids:
                it = rt.mem_id_to_item.get(mid)
                if it is not None and it.status != "active":
                    archived.append(mid)
            if archived:
                raise RecapError(
                    "Archived mem:<id> cited in output. Do not cite archived memories. "
                    f"Archived: {archived}"
                )

            return citations, resolved_mem_ids, used_aliases, mem_tokens

        while True:
            ctx.check_cancelled()

            role_instruction = cfg.roles.get(rt.node_ptr.role, "")

            # Build the next prompt (user message) based on state.
            if rt.state == RecapState.DOWN:
                prompt = render_template(
                    cfg.prompts.down_prompt_template,
                    {
                        "task_name": rt.node_ptr.task_name,
                        "role": rt.node_ptr.role,
                        "role_instruction": role_instruction,
                        "user_request": user_request,
                        "recipes_per_run": ctx.recipes_per_run,
                    },
                )
            elif rt.state == RecapState.ACTION_TAKEN:
                prompt = render_template(
                    cfg.prompts.action_taken_prompt_template,
                    {
                        "task_name": rt.node_ptr.task_name,
                        "role": rt.node_ptr.role,
                        "role_instruction": role_instruction,
                        "user_request": user_request,
                        "recipes_per_run": ctx.recipes_per_run,
                        "obs": rt.latest_obs,
                        "remaining_subtask_str": _format_subtasks_for_prompt(rt.remaining_subtasks),
                    },
                )
            elif rt.state == RecapState.UP:
                prompt = render_template(
                    cfg.prompts.up_prompt_template,
                    {
                        "task_name": rt.node_ptr.task_name,
                        "role": rt.node_ptr.role,
                        "role_instruction": role_instruction,
                        "user_request": user_request,
                        "recipes_per_run": ctx.recipes_per_run,
                        "done_task_name": rt.done_task_name,
                        "done_task_result": rt.done_task_result,
                        "previous_stage_task_name": rt.previous_stage_task_name,
                        "previous_stage_think": rt.previous_stage_think,
                        "remaining_subtask_str": _format_subtasks_for_prompt(rt.remaining_subtasks),
                    },
                )
            else:
                raise RecapError(f"Unknown state: {rt.state}")

            # Call the LLM with internal retries on JSON parse failure.
            # Important: do NOT commit invalid assistant outputs into the shared history.
            base_messages = [{"role": "system", "content": system_prompt}] + history + [
                {"role": "user", "content": prompt}
            ]
            extra_user_messages: list[dict[str, Any]] = []
            last_parse_error: str | None = None
            raw: Any | None = None
            info: RecapInfo | None = None

            for attempt in range(1, 4):
                ctx.check_cancelled()
                if rt.steps >= int(cfg.recap.max_steps):
                    raise RecapError(f"Exceeded recap.max_steps={cfg.recap.max_steps}")
                rt.steps += 1

                messages = base_messages + extra_user_messages
                ctx.trace(
                    "llm_request",
                    {
                        "ts": _now_ts(),
                        "agent": rt.node_ptr.role,
                        "recap_state": rt.state.value,
                        "task_name": rt.node_ptr.task_name,
                        "model": ctx.llm.model,
                        "enable_thinking": bool(getattr(ctx.llm, "enable_thinking", False)),
                        "temperature": ctx.temperature,
                        "attempt": attempt,
                        "steps": rt.steps,
                        "messages": messages,
                    },
                )
                plan_extra: dict[str, Any] = {}
                if not bool(getattr(ctx.llm, "enable_thinking", False)):
                    plan_extra = {"response_format": _RECAP_RESPONSE_FORMAT}
                raw = ctx.llm.chat_messages(
                    messages=messages,
                    temperature=ctx.temperature,
                    extra=plan_extra,
                )
                ctx.trace(
                    "llm_response",
                    {
                        "ts": _now_ts(),
                        "agent": rt.node_ptr.role,
                        "recap_state": rt.state.value,
                        "task_name": rt.node_ptr.task_name,
                        "attempt": attempt,
                        "steps": rt.steps,
                        "content": raw.content,
                        "reasoning_content": raw.reasoning_content,
                        "raw": raw.raw,
                    },
                )

                try:
                    info = _parse_recap_info(raw.content)
                    last_parse_error = None
                    break
                except JSONExtractionError as e:
                    last_parse_error = str(e)
                    # Ask for a corrected output; keep this retry instruction ephemeral.
                    extra_user_messages = [
                        {
                            "role": "user",
                            "content": (
                                "FORMAT ERROR: Your previous output was not valid ReCAP JSON.\n"
                                f"{e}\n\n"
                                "Return ONLY a single valid JSON object with keys:\n"
                                '- think: string\n'
                                "- subtasks: array of objects (structured subtasks)\n"
                                "- result: string or JSON (REQUIRED when subtasks=[])\n"
                                "No extra text."
                            ),
                        }
                    ]
                    continue

            if info is None or raw is None:
                raise RecapError(
                    f"Failed to obtain valid ReCAP JSON after retries. Last error: {last_parse_error}"
                )

            # Commit only the successful exchange to shared history.
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": raw.content})
            history = _trim_history(history, max_rounds=int(cfg.recap.max_rounds))

            # Any inline citations used in intermediate reasoning/results are treated as "focused"
            # evidence, so the final generation prompt can stay small.
            _merge_focus_aliases(rt, extract_citation_aliases(info.think))
            _merge_focus_aliases(rt, extract_citation_aliases(info.result))
            _merge_focus_mem_ids(rt, extract_memory_ids(info.think))
            _merge_focus_mem_ids(rt, extract_memory_ids(info.result))

            rt.node_ptr.set_info(info)
            ctx.trace(
                "recap_info",
                {
                    "ts": _now_ts(),
                    "agent": rt.node_ptr.role,
                    "recap_state": rt.state.value,
                    "task_name": rt.node_ptr.task_name,
                    "think": info.think,
                    "subtasks": info.subtasks,
                    "result": info.result,
                    "depth": rt.depth,
                    "steps": rt.steps,
                },
            )

            # Decide next step based on the first subtask (plan-ahead decomposition).
            if not info.subtasks:
                if rt.node_ptr.parent is not None and not info.result.strip():
                    # Enforce a structured "done deliverable" so UP-stage integration does not
                    # rely on the model re-reading the entire shared conversation history.
                    rt.latest_obs = (
                        "ERROR: Task ended with empty subtasks but without a `result`.\n"
                        "When subtasks=[], you MUST include a non-empty `result` summarizing the deliverable "
                        "(and key conclusions / constraints / citations if applicable)."
                    )
                    rt.remaining_subtasks = []
                    rt.state = RecapState.ACTION_TAKEN
                    continue

                # Task done; backtrack to parent (or error if root ends without final generation).
                if rt.node_ptr.parent is None:
                    # Fallback: some models output the final recipes JSON in `result` instead of calling
                    # generate_recipes. Accept it if it validates; otherwise, keep strict behavior.
                    try:
                        parsed_root = extract_first_json_object(info.result)
                        if not isinstance(parsed_root, dict):
                            raise RecapError("Root `result` is not a JSON object.")
                        citations, resolved_mem_ids, used_aliases, mem_tokens = _validate_and_resolve_final_output(
                            parsed_root,
                            context="root_fallback",
                        )
                        ctx.trace(
                            "final_output_fallback_used",
                            {
                                "ts": _now_ts(),
                                "agent": rt.node_ptr.role,
                                "used_aliases": used_aliases,
                                "resolved_citations": citations,
                                "mem_tokens": mem_tokens,
                                "resolved_mem_ids": resolved_mem_ids,
                            },
                        )
                        return parsed_root, citations, resolved_mem_ids
                    except Exception as e:
                        raise RecapError("Root task ended without generate_recipes.") from e

                rt.done_task_name = rt.node_ptr.task_name
                rt.done_task_result = info.result.strip()
                rt.node_ptr = rt.node_ptr.parent
                rt.depth = max(rt.depth - 1, 0)

                parent_info = rt.node_ptr.get_latest_info()
                rt.previous_stage_task_name = rt.node_ptr.task_name
                rt.previous_stage_think = parent_info.think
                rt.remaining_subtasks = parent_info.subtasks[1:]
                rt.state = RecapState.UP
                continue

            first = info.subtasks[0]
            stype = str(first.get("type") or "").strip()

            if stype == "generate_recipes":
                # Only the root orchestrator is allowed to produce the final output.
                if rt.node_ptr.role != "orchestrator" or rt.node_ptr.parent is not None:
                    rt.latest_obs = (
                        "ERROR: generate_recipes can only be called by the orchestrator at the root task.\n"
                        "If you are an expert node (MOF/TIO2) or a nested subtask, return to the parent "
                        "by finishing your task with subtasks=[] and a `result`, then let the root orchestrator "
                        "call generate_recipes."
                    )
                    rt.remaining_subtasks = info.subtasks[1:]
                    rt.state = RecapState.ACTION_TAKEN
                    continue

                # Final generation primitive.
                if not rt.kb_all_aliased_chunks and not rt.mem_all_items:
                    raise RecapError(
                        "generate_recipes requires prior evidence: run kb_search (KB literature) and/or "
                        "mem_search (ReasoningBank) first."
                    )

                # generate_recipes is a *process*: we provide a compact evidence index, and the model can
                # call kb_get/kb_list as needed to open full chunk text on-demand (instead of dumping all
                # evidence into a single prompt).

                def _build_evidence_index() -> str:
                    total = len(rt.kb_all_aliased_chunks)
                    default_limit = int(cfg.evidence.kb_list_default_limit)
                    max_limit = int(cfg.evidence.kb_list_max_limit)
                    limit = min(max(default_limit, 1), max_limit)

                    focused = [a for a in rt.kb_focus_aliases if a in rt.kb_alias_to_chunk]
                    recent = [a for a in rt.last_kb_search_aliases if a in rt.kb_alias_to_chunk]
                    ordered: list[str] = []
                    seen: set[str] = set()
                    for a in focused + recent:
                        if a in seen:
                            continue
                        seen.add(a)
                        ordered.append(a)
                    for ch in rt.kb_all_aliased_chunks:
                        if ch.alias in seen:
                            continue
                        seen.add(ch.alias)
                        ordered.append(ch.alias)

                    shown = ordered[:limit]
                    lines: list[str] = []
                    lines.append(f"Total chunks in run registry: {total}. Showing {len(shown)}/{total} aliases.")
                    lines.append("Use kb_list to view more, kb_get to open full text by alias.")
                    if total == 0:
                        lines.append("(empty; run kb_search first)")
                    lines.append("")
                    for alias in shown:
                        ch = rt.kb_alias_to_chunk.get(alias)
                        if ch is None:
                            continue
                        lines.append(f"[{ch.alias}] source={ch.source}")
                    return "\n".join(lines).strip()

                def _build_mem_index() -> str:
                    total = len(rt.mem_all_items)
                    default_limit = int(cfg.reasoningbank.mem_list_default_limit)
                    max_limit = int(cfg.reasoningbank.mem_list_max_limit)
                    limit = min(max(default_limit, 1), max_limit)

                    focused = [m for m in rt.mem_focus_ids if m in rt.mem_id_to_item]
                    recent = [m for m in rt.last_mem_search_ids if m in rt.mem_id_to_item]
                    ordered: list[str] = []
                    seen: set[str] = set()
                    for mid in focused + recent:
                        if mid in seen:
                            continue
                        seen.add(mid)
                        ordered.append(mid)
                    for it in rt.mem_all_items:
                        if it.mem_id in seen:
                            continue
                        seen.add(it.mem_id)
                        ordered.append(it.mem_id)

                    shown = ordered[:limit]
                    lines: list[str] = []
                    lines.append(f"Total memories in run registry: {total}. Showing {len(shown)}/{total} mem_ids.")
                    lines.append("Use mem_list to view more, mem_get to open full content by mem_id.")
                    if total == 0:
                        lines.append("(empty; run mem_search first)")
                    lines.append("")
                    for mem_id in shown:
                        it = rt.mem_id_to_item.get(mem_id)
                        if it is None:
                            continue
                        snippet = it.content.replace("\n", " ").strip()
                        if len(snippet) > 160:
                            snippet = snippet[:160] + "…"
                        lines.append(f"mem:{it.mem_id} role={it.role} type={it.type} status={it.status} :: {snippet}")
                    return "\n".join(lines).strip()

                gen_prompt = render_template(
                    cfg.prompts.generate_recipes_prompt_template,
                    {
                        "user_request": user_request,
                        "recipes_per_run": ctx.recipes_per_run,
                        "kb_evidence_index": _build_evidence_index(),
                        "mem_evidence_index": _build_mem_index(),
                    },
                )

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "kb_get",
                            "description": "Fetch the full original chunk text for a citation alias (e.g. C12) from the run evidence registry.",
                            "parameters": {
                                "type": "object",
                                "properties": {"alias": {"type": "string"}},
                                "required": ["alias"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "kb_list",
                            "description": "List available citation aliases (and sources) currently stored in the run evidence registry.",
                            "parameters": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "mem_search",
                            "description": "Search ReasoningBank memories and add results to the run memory registry.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer"},
                                },
                                "required": ["query"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "mem_get",
                            "description": "Fetch the full memory content for a mem_id from the run memory registry.",
                            "parameters": {
                                "type": "object",
                                "properties": {"mem_id": {"type": "string"}},
                                "required": ["mem_id"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "mem_list",
                            "description": "List available mem_ids currently stored in the run memory registry.",
                            "parameters": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                        },
                    },
                ]

                # Safety: cap how many full chunks can be injected via kb_get during generation.
                max_full = int(cfg.evidence.max_full_chunks_in_generate_recipes)
                opened_aliases: set[str] = set()
                max_full_mem = int(cfg.reasoningbank.max_full_memories_in_generate_recipes)
                opened_mem_ids: set[str] = set()

                gen_history: list[dict[str, Any]] = list(history) + [{"role": "user", "content": gen_prompt}]
                format_errors = 0

                for turn in range(1, 21):
                    ctx.check_cancelled()
                    if rt.steps >= int(cfg.recap.max_steps):
                        raise RecapError(f"Exceeded recap.max_steps={cfg.recap.max_steps}")
                    rt.steps += 1

                    gen_messages = [{"role": "system", "content": system_prompt}] + gen_history
                    ctx.trace(
                        "llm_request",
                        {
                            "ts": _now_ts(),
                            "agent": "orchestrator",
                            "recap_state": "generate_recipes",
                            "task_name": rt.node_ptr.task_name,
                            "model": ctx.llm.model,
                            "enable_thinking": bool(getattr(ctx.llm, "enable_thinking", False)),
                            "temperature": ctx.temperature,
                            "turn": turn,
                            "steps": rt.steps,
                            "messages": gen_messages,
                        },
                    )
                    gen_raw = ctx.llm.chat_messages(
                        messages=gen_messages,
                        temperature=ctx.temperature,
                        extra={
                            "tools": tools,
                            "tool_choice": "auto",
                        },
                    )
                    ctx.trace(
                        "llm_response",
                        {
                            "ts": _now_ts(),
                            "agent": "orchestrator",
                            "recap_state": "generate_recipes",
                            "task_name": rt.node_ptr.task_name,
                            "turn": turn,
                            "steps": rt.steps,
                            "content": gen_raw.content,
                            "reasoning_content": gen_raw.reasoning_content,
                            "raw": gen_raw.raw,
                            "tool_calls": gen_raw.tool_calls,
                        },
                    )

                    # Tool call path (preferred for on-demand evidence access).
                    if gen_raw.tool_calls:
                        gen_history.append(
                            {
                                "role": "assistant",
                                "content": gen_raw.content,
                                "tool_calls": gen_raw.tool_calls,
                            }
                        )
                        for tc in gen_raw.tool_calls:
                            tc_id = str(tc.get("id") or f"tool_call_{turn}")
                            fn = tc.get("function") or {}
                            name = str(fn.get("name") or "").strip()
                            args_raw = fn.get("arguments") or ""
                            try:
                                args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw.strip() else {}
                            except Exception:
                                args = {}

                            tool_obs: str
                            if name == "kb_get":
                                alias = str(args.get("alias") or "").strip()
                                if alias.startswith("[") and alias.endswith("]"):
                                    alias = alias[1:-1].strip()

                                stored = rt.kb_alias_to_chunk.get(alias)
                                if stored is None:
                                    tool_obs = (
                                        f"ERROR: Unknown citation alias: {alias!r}.\n"
                                        "You can only kb_get an alias that exists in the run evidence registry."
                                    )
                                elif alias not in opened_aliases and len(opened_aliases) >= max_full:
                                    tool_obs = (
                                        "ERROR: kb_get limit reached for generate_recipes.\n"
                                        f"Already opened {len(opened_aliases)}/{max_full} full chunks; "
                                        "use the evidence you already opened or narrow your needs."
                                    )
                                else:
                                    opened_aliases.add(alias)
                                    _merge_focus_aliases(rt, [stored.alias])
                                    tool_obs = (
                                        "KB get (from run evidence registry):\n"
                                        f"[{stored.alias}] source={stored.source}\n"
                                        f"{stored.content}\n"
                                    ).strip()
                                    ctx.trace(
                                        "kb_get",
                                        {
                                            "ts": _now_ts(),
                                            "agent": "orchestrator",
                                            "context": "generate_recipes",
                                            "alias": stored.alias,
                                            "ref": stored.ref,
                                            "source": stored.source,
                                            "kb_namespace": stored.kb_namespace,
                                            "lightrag_chunk_id": stored.lightrag_chunk_id,
                                        },
                                    )
                            elif name == "kb_list":
                                total = len(rt.kb_all_aliased_chunks)
                                default_limit = int(cfg.evidence.kb_list_default_limit)
                                max_limit = int(cfg.evidence.kb_list_max_limit)
                                try:
                                    limit = int(args.get("limit")) if args.get("limit") is not None else default_limit
                                except Exception:
                                    limit = default_limit
                                if limit < 1:
                                    limit = 1
                                if limit > max_limit:
                                    limit = max_limit

                                shown = rt.kb_all_aliased_chunks[:limit]
                                lines: list[str] = []
                                lines.append(f"KB evidence registry: {total} chunks total.")
                                if total == 0:
                                    lines.append("(empty; run kb_search first)")
                                else:
                                    lines.append(f"Showing {len(shown)}/{total} (limit={limit}).")
                                    lines.append("")
                                    for a in shown:
                                        lines.append(f"[{a.alias}] source={a.source}")
                                tool_obs = "\n".join(lines).strip()
                                ctx.trace(
                                    "kb_list",
                                    {
                                        "ts": _now_ts(),
                                        "agent": "orchestrator",
                                        "context": "generate_recipes",
                                        "total": total,
                                        "limit": limit,
                                        "shown_aliases": [a.alias for a in shown],
                                    },
                                )
                            elif name == "mem_search":
                                if ctx.rb is None:
                                    tool_obs = (
                                        "ERROR: ReasoningBank is not configured.\n"
                                        "mem_search is unavailable in this run."
                                    )
                                else:
                                    query = str(args.get("query") or "").strip()
                                    try:
                                        top_k_arg = int(args.get("top_k")) if args.get("top_k") is not None else None
                                    except Exception:
                                        top_k_arg = None

                                    results: list[dict[str, Any]] = []
                                    if top_k_arg is not None and top_k_arg > 0:
                                        results = ctx.rb.query(query=query, n_results=top_k_arg, status=["active"])
                                    else:
                                        k_role = int(cfg.reasoningbank.k_role)
                                        k_global = int(cfg.reasoningbank.k_global)
                                        role_results = ctx.rb.query(
                                            query=query,
                                            n_results=k_role,
                                            status=["active"],
                                            role=[rt.node_ptr.role],
                                        )
                                        global_results = ctx.rb.query(
                                            query=query,
                                            n_results=k_global,
                                            status=["active"],
                                            role=["global"],
                                        )
                                        results = role_results + global_results

                                    seen: set[str] = set()
                                    mem_ids: list[str] = []
                                    matched_by_id: dict[str, list[dict[str, Any]]] = {}
                                    distance_by_id: dict[str, float | None] = {}
                                    for r in results:
                                        it: MemoryItem = r["item"]
                                        matched = r.get("matched_claims") or []
                                        if isinstance(matched, list) and matched:
                                            matched_by_id.setdefault(it.mem_id, []).extend(
                                                [m for m in matched if isinstance(m, dict)]
                                            )
                                        distance = r.get("distance")
                                        if distance is not None:
                                            try:
                                                d = float(distance)
                                            except Exception:
                                                d = None
                                            prev = distance_by_id.get(it.mem_id)
                                            if prev is None or (d is not None and d < prev):
                                                distance_by_id[it.mem_id] = d
                                        if it.mem_id in seen:
                                            continue
                                        seen.add(it.mem_id)
                                        mem_ids.append(it.mem_id)
                                        if it.mem_id not in rt.mem_id_to_item:
                                            rt.mem_id_to_item[it.mem_id] = it
                                            rt.mem_all_items.append(it)

                                    rt.last_mem_search_ids = mem_ids

                                    lines = [f"MEM search results: {len(mem_ids)} items."]
                                    for mid in mem_ids[: min(len(mem_ids), 8)]:
                                        it = rt.mem_id_to_item.get(mid)
                                        if it is None:
                                            continue
                                        snippet = ""
                                        matched = matched_by_id.get(mid) or []
                                        if matched:
                                            # Show the best-matching claim snippet when available.
                                            best = sorted(
                                                matched,
                                                key=lambda m: float(m.get("distance")) if m.get("distance") is not None else 1e9,
                                            )[0]
                                            claim_id = str(best.get("claim_id") or "").strip()
                                            claim_text = str(best.get("text") or "").replace("\n", " ").strip()
                                            if len(claim_text) > 180:
                                                claim_text = claim_text[:180] + "…"
                                            if claim_id:
                                                snippet = f"match={claim_id} :: {claim_text}"
                                            else:
                                                snippet = f"match :: {claim_text}"
                                        if not snippet:
                                            snippet = it.content.replace("\n", " ").strip()
                                            if len(snippet) > 200:
                                                snippet = snippet[:200] + "…"
                                        lines.append(f"mem:{it.mem_id} role={it.role} type={it.type} :: {snippet}")
                                    if len(mem_ids) > 8:
                                        lines.append("Use mem_list to view more, mem_get to open full content by mem_id.")
                                    tool_obs = "\n".join(lines).strip()

                                    ctx.trace(
                                        "mem_search",
                                        {
                                            "ts": _now_ts(),
                                            "agent": "orchestrator",
                                            "context": "generate_recipes",
                                            "query": query,
                                            "top_k": top_k_arg,
                                            "results": [
                                                {
                                                    "mem_id": rt.mem_id_to_item[m].mem_id,
                                                    "role": rt.mem_id_to_item[m].role,
                                                    "type": rt.mem_id_to_item[m].type,
                                                    "status": rt.mem_id_to_item[m].status,
                                                    "source_run_id": rt.mem_id_to_item[m].source_run_id,
                                                    "distance": distance_by_id.get(m),
                                                    "matched_claims": matched_by_id.get(m, [])[:6],
                                                }
                                                for m in mem_ids
                                                if m in rt.mem_id_to_item
                                            ],
                                        },
                                    )
                            elif name == "mem_get":
                                mem_id = str(args.get("mem_id") or "").strip()
                                if mem_id.startswith("mem:"):
                                    mem_id = mem_id[4:].strip()

                                stored = rt.mem_id_to_item.get(mem_id)
                                if stored is None:
                                    tool_obs = (
                                        f"ERROR: Unknown mem_id: {mem_id!r}.\n"
                                        "You can only mem_get a mem_id that exists in the run memory registry "
                                        "(run mem_search first)."
                                    )
                                elif mem_id not in opened_mem_ids and len(opened_mem_ids) >= max_full_mem:
                                    tool_obs = (
                                        "ERROR: mem_get limit reached for generate_recipes.\n"
                                        f"Already opened {len(opened_mem_ids)}/{max_full_mem} full memories; "
                                        "use the memories you already opened or narrow your needs."
                                    )
                                else:
                                    opened_mem_ids.add(mem_id)
                                    _merge_focus_mem_ids(rt, [stored.mem_id])
                                    tool_obs = (
                                        "MEM get (from run memory registry):\n"
                                        f"mem:{stored.mem_id} role={stored.role} type={stored.type} status={stored.status}\n"
                                        f"{stored.content}\n"
                                    ).strip()
                                    ctx.trace(
                                        "mem_get",
                                        {
                                            "ts": _now_ts(),
                                            "agent": "orchestrator",
                                            "context": "generate_recipes",
                                            "mem_id": stored.mem_id,
                                            "role": stored.role,
                                            "type": stored.type,
                                            "status": stored.status,
                                            "source_run_id": stored.source_run_id,
                                        },
                                    )
                            elif name == "mem_list":
                                total = len(rt.mem_all_items)
                                default_limit = int(cfg.reasoningbank.mem_list_default_limit)
                                max_limit = int(cfg.reasoningbank.mem_list_max_limit)
                                try:
                                    limit = int(args.get("limit")) if args.get("limit") is not None else default_limit
                                except Exception:
                                    limit = default_limit
                                if limit < 1:
                                    limit = 1
                                if limit > max_limit:
                                    limit = max_limit

                                shown = rt.mem_all_items[:limit]
                                lines = [f"Run memory registry: {total} memories total."]
                                if total == 0:
                                    lines.append("(empty; run mem_search first)")
                                else:
                                    lines.append(f"Showing {len(shown)}/{total} (limit={limit}).")
                                    lines.append("")
                                    for it in shown:
                                        snippet = it.content.replace("\n", " ").strip()
                                        if len(snippet) > 120:
                                            snippet = snippet[:120] + "…"
                                        lines.append(f"mem:{it.mem_id} role={it.role} type={it.type} :: {snippet}")
                                tool_obs = "\n".join(lines).strip()
                                ctx.trace(
                                    "mem_list",
                                    {
                                        "ts": _now_ts(),
                                        "agent": "orchestrator",
                                        "context": "generate_recipes",
                                        "total": total,
                                        "limit": limit,
                                        "shown_mem_ids": [it.mem_id for it in shown],
                                    },
                                )
                            else:
                                tool_obs = f"ERROR: Unknown tool name: {name!r}"

                            gen_history.append({"role": "tool", "tool_call_id": tc_id, "content": tool_obs})
                        continue

                    # Final output path (schema-enforced): use response_format=json_schema to guarantee that the
                    # *final* recipes JSON is 100% structured.
                    if rt.steps >= int(cfg.recap.max_steps):
                        raise RecapError(f"Exceeded recap.max_steps={cfg.recap.max_steps}")
                    rt.steps += 1

                    final_messages = [{"role": "system", "content": system_prompt}] + gen_history + [
                        {
                            "role": "user",
                            "content": (
                                "Now return the final answer as a single JSON object ONLY. "
                                "No extra text."
                            ),
                        }
                    ]
                    final_extra: dict[str, Any] = {}
                    if not bool(getattr(ctx.llm, "enable_thinking", False)):
                        final_extra = {
                            "response_format": _recipes_response_format(recipes_per_run=int(ctx.recipes_per_run))
                        }
                    ctx.trace(
                        "llm_request",
                        {
                            "ts": _now_ts(),
                            "agent": "orchestrator",
                            "recap_state": "generate_recipes.final",
                            "task_name": rt.node_ptr.task_name,
                            "model": ctx.llm.model,
                            "enable_thinking": bool(getattr(ctx.llm, "enable_thinking", False)),
                            "temperature": ctx.temperature,
                            "turn": turn,
                            "steps": rt.steps,
                            "messages": final_messages,
                            "extra": final_extra,
                        },
                    )
                    final_raw = ctx.llm.chat_messages(
                        messages=final_messages,
                        temperature=ctx.temperature,
                        extra=final_extra,
                    )
                    ctx.trace(
                        "llm_response",
                        {
                            "ts": _now_ts(),
                            "agent": "orchestrator",
                            "recap_state": "generate_recipes.final",
                            "task_name": rt.node_ptr.task_name,
                            "turn": turn,
                            "steps": rt.steps,
                            "content": final_raw.content,
                            "reasoning_content": final_raw.reasoning_content,
                            "raw": final_raw.raw,
                        },
                    )

                    try:
                        parsed = extract_first_json_object(final_raw.content)
                    except JSONExtractionError as e:
                        format_errors += 1
                        if format_errors >= 3:
                            raise RecapError(f"generate_recipes final output is not valid JSON after retries: {e}")
                        gen_history.append(
                            {
                                "role": "user",
                                "content": (
                                    f"FORMAT ERROR: {e}\n\n"
                                    "Return ONLY a single valid JSON object matching the required schema. No extra text."
                                ),
                            }
                        )
                        continue

                    try:
                        citations, resolved_mem_ids, used_aliases, mem_tokens = _validate_and_resolve_final_output(
                            parsed,
                            context="generate_recipes.final",
                        )
                    except RecapError as e:
                        gen_history.append(
                            {
                                "role": "user",
                                "content": (
                                    f"ERROR: {e}\n\n"
                                    "Fix the JSON so it matches the required schema, and ensure all citations are valid "
                                    "(existing KB aliases / existing mem:<id> in the run registry)."
                                ),
                            }
                        )
                        continue

                    ctx.trace(
                        "citations_resolved",
                        {
                            "ts": _now_ts(),
                            "agent": "orchestrator",
                            "aliases": used_aliases,
                            "resolved": citations,
                        },
                    )
                    ctx.trace(
                        "memories_resolved",
                        {
                            "ts": _now_ts(),
                            "agent": "orchestrator",
                            "mem_tokens": mem_tokens,
                            "mem_ids": resolved_mem_ids,
                            "resolved": [
                                {
                                    "mem_id": mid,
                                    "role": rt.mem_id_to_item[mid].role if mid in rt.mem_id_to_item else None,
                                    "type": rt.mem_id_to_item[mid].type if mid in rt.mem_id_to_item else None,
                                    "source_run_id": rt.mem_id_to_item[mid].source_run_id
                                    if mid in rt.mem_id_to_item
                                    else None,
                                }
                                for mid in resolved_mem_ids
                            ],
                        },
                    )
                    return parsed, citations, resolved_mem_ids

                raise RecapError("generate_recipes exceeded maximum turns without producing a valid final output.")

            if stype == "kb_search":
                kb_name = str(first.get("kb_name") or "").strip()
                query = str(first.get("query") or "").strip()
                top_k = (
                    int(first.get("top_k"))
                    if first.get("top_k") is not None
                    else int(cfg.kb.default_top_k)
                )
                mode = str(first.get("mode") or cfg.kb.default_mode)

                if kb_name == "kb_principles":
                    kb = ctx.kbs.kb_principles
                elif kb_name == "kb_modulation":
                    kb = ctx.kbs.kb_modulation
                else:
                    obs = f"ERROR: Unknown kb_name={kb_name!r}. Valid: kb_principles, kb_modulation."
                    rt.latest_obs = obs
                    rt.remaining_subtasks = info.subtasks[1:]
                    rt.state = RecapState.ACTION_TAKEN
                    continue

                chunks = kb.query_chunks(query, mode=mode, top_k=top_k)

                # Assign GLOBAL aliases, stable across multiple kb_search calls within this run.
                aliased_for_obs: list[AliasedKBChunk] = []
                seen_in_obs: set[str] = set()
                for ch in chunks:
                    ref = ch.ref
                    alias = rt.kb_ref_to_alias.get(ref)
                    if alias is None:
                        alias = f"{cfg.citations.alias_prefix}{rt.kb_next_index}"
                        rt.kb_next_index += 1
                        rt.kb_ref_to_alias[ref] = alias
                        rt.kb_alias_map[alias] = ref

                        stored = AliasedKBChunk(
                            alias=alias,
                            ref=ref,
                            source=ch.source,
                            content=ch.content,
                            kb_namespace=ch.kb_namespace,
                            lightrag_chunk_id=ch.lightrag_chunk_id,
                        )
                        rt.kb_alias_to_chunk[alias] = stored
                        rt.kb_all_aliased_chunks.append(stored)
                    else:
                        stored = rt.kb_alias_to_chunk.get(alias)
                        if stored is None:
                            stored = AliasedKBChunk(
                                alias=alias,
                                ref=ref,
                                source=ch.source,
                                content=ch.content,
                                kb_namespace=ch.kb_namespace,
                                lightrag_chunk_id=ch.lightrag_chunk_id,
                            )
                            rt.kb_alias_to_chunk[alias] = stored
                            # Do NOT append to kb_all_aliased_chunks here: alias already existed.

                    if alias in seen_in_obs:
                        continue
                    seen_in_obs.add(alias)
                    aliased_for_obs.append(stored)

                obs = _format_kb_observation(
                    kb_name=kb_name,
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    aliased=aliased_for_obs,
                )
                rt.node_ptr.set_obs(obs)
                rt.latest_obs = obs
                rt.last_kb_search_aliases = [a.alias for a in aliased_for_obs]

                ctx.trace(
                    "kb_query",
                    {
                        "ts": _now_ts(),
                        "agent": rt.node_ptr.role,
                        "kb_namespace": kb_name,
                        "query": query,
                        "mode": mode,
                        "top_k": top_k,
                        "results": [
                            {
                                "alias": a.alias,
                                "ref": a.ref,
                                "source": a.source,
                                "content": a.content,
                                "kb_namespace": a.kb_namespace,
                                "lightrag_chunk_id": a.lightrag_chunk_id,
                            }
                            for a in aliased_for_obs
                        ],
                    },
                )

                rt.remaining_subtasks = info.subtasks[1:]
                rt.state = RecapState.ACTION_TAKEN
                continue

            if stype == "kb_get":
                alias = str(first.get("alias") or "").strip()
                stored = rt.kb_alias_to_chunk.get(alias)
                if stored is None:
                    rt.latest_obs = (
                        f"ERROR: Unknown citation alias: {alias!r}.\n"
                        "You can only kb_get an alias that was returned by a prior kb_search in this run."
                    )
                else:
                    _merge_focus_aliases(rt, [stored.alias])
                    rt.latest_obs = (
                        "KB get (from run evidence registry):\n"
                        f"[{stored.alias}] source={stored.source}\n"
                        f"{stored.content}\n"
                    ).strip()
                    ctx.trace(
                        "kb_get",
                        {
                            "ts": _now_ts(),
                            "agent": rt.node_ptr.role,
                            "alias": stored.alias,
                            "ref": stored.ref,
                            "source": stored.source,
                            "kb_namespace": stored.kb_namespace,
                            "lightrag_chunk_id": stored.lightrag_chunk_id,
                        },
                    )

                rt.remaining_subtasks = info.subtasks[1:]
                rt.state = RecapState.ACTION_TAKEN
                continue

            if stype == "kb_list":
                total = len(rt.kb_all_aliased_chunks)
                default_limit = int(cfg.evidence.kb_list_default_limit)
                max_limit = int(cfg.evidence.kb_list_max_limit)
                limit_raw = first.get("limit")
                try:
                    limit = int(limit_raw) if limit_raw is not None else default_limit
                except Exception:
                    limit = default_limit
                if limit < 1:
                    limit = 1
                if limit > max_limit:
                    limit = max_limit

                shown = rt.kb_all_aliased_chunks[:limit]
                lines: list[str] = []
                lines.append(f"KB evidence registry: {total} chunks total.")
                if total == 0:
                    lines.append("(empty; run kb_search first)")
                else:
                    lines.append(f"Showing {len(shown)}/{total} (limit={limit}).")
                    lines.append("")
                    for a in shown:
                        lines.append(f"[{a.alias}] source={a.source}")

                rt.latest_obs = "\n".join(lines).strip()
                ctx.trace(
                    "kb_list",
                    {
                        "ts": _now_ts(),
                        "agent": rt.node_ptr.role,
                        "total": total,
                        "limit": limit,
                        "shown_aliases": [a.alias for a in shown],
                    },
                )

                rt.remaining_subtasks = info.subtasks[1:]
                rt.state = RecapState.ACTION_TAKEN
                continue

            if stype == "mem_search":
                query = str(first.get("query") or "").strip()
                top_k_raw = first.get("top_k")
                try:
                    top_k = int(top_k_raw) if top_k_raw is not None else None
                except Exception:
                    top_k = None
                role = str(first.get("role") or "").strip() or None
                status = str(first.get("status") or "active").strip() or "active"
                mem_type = str(first.get("mem_type") or "").strip() or None

                if ctx.rb is None:
                    rt.latest_obs = "ERROR: ReasoningBank is not configured (mem_search unavailable)."
                    rt.remaining_subtasks = info.subtasks[1:]
                    rt.state = RecapState.ACTION_TAKEN
                    continue

                # Retrieval strategy:
                # - If role is explicitly provided, search that role.
                # - Otherwise, search current role (k_role) + global (k_global) and merge.
                results: list[dict[str, Any]] = []
                if role:
                    results = ctx.rb.query(
                        query=query,
                        n_results=int(top_k or 5),
                        role=[role],
                        status=[status],
                        type=[mem_type] if mem_type else None,
                    )
                else:
                    k_role = int(cfg.reasoningbank.k_role)
                    k_global = int(cfg.reasoningbank.k_global)
                    role_results = ctx.rb.query(
                        query=query,
                        n_results=int(k_role if top_k is None else top_k),
                        role=[rt.node_ptr.role],
                        status=[status],
                        type=[mem_type] if mem_type else None,
                    )
                    global_results = ctx.rb.query(
                        query=query,
                        n_results=int(k_global if top_k is None else top_k),
                        role=["global"],
                        status=[status],
                        type=[mem_type] if mem_type else None,
                    )
                    results = role_results + global_results

                seen: set[str] = set()
                mem_ids: list[str] = []
                matched_by_id: dict[str, list[dict[str, Any]]] = {}
                distance_by_id: dict[str, float | None] = {}
                for r in results:
                    it: MemoryItem = r["item"]
                    matched = r.get("matched_claims") or []
                    if isinstance(matched, list) and matched:
                        matched_by_id.setdefault(it.mem_id, []).extend([m for m in matched if isinstance(m, dict)])
                    distance = r.get("distance")
                    if distance is not None:
                        try:
                            d = float(distance)
                        except Exception:
                            d = None
                        prev = distance_by_id.get(it.mem_id)
                        if prev is None or (d is not None and d < prev):
                            distance_by_id[it.mem_id] = d
                    if it.mem_id in seen:
                        continue
                    seen.add(it.mem_id)
                    mem_ids.append(it.mem_id)
                    if it.mem_id not in rt.mem_id_to_item:
                        rt.mem_id_to_item[it.mem_id] = it
                        rt.mem_all_items.append(it)

                rt.last_mem_search_ids = mem_ids

                lines: list[str] = []
                lines.append(f"MEM search: {len(mem_ids)} items.")
                for mid in mem_ids[: min(len(mem_ids), 8)]:
                    it = rt.mem_id_to_item.get(mid)
                    if it is None:
                        continue
                    snippet = ""
                    matched = matched_by_id.get(mid) or []
                    if matched:
                        best = sorted(
                            matched,
                            key=lambda m: float(m.get("distance")) if m.get("distance") is not None else 1e9,
                        )[0]
                        claim_id = str(best.get("claim_id") or "").strip()
                        claim_text = str(best.get("text") or "").replace("\n", " ").strip()
                        if len(claim_text) > 200:
                            claim_text = claim_text[:200] + "…"
                        if claim_id:
                            snippet = f"match={claim_id} :: {claim_text}"
                        else:
                            snippet = f"match :: {claim_text}"
                    if not snippet:
                        snippet = it.content.replace("\n", " ").strip()
                        if len(snippet) > 220:
                            snippet = snippet[:220] + "…"
                    lines.append(
                        f"mem:{it.mem_id} role={it.role} type={it.type} status={it.status} source_run_id={it.source_run_id or ''}\n"
                        f"{snippet}"
                    )
                if len(mem_ids) > 8:
                    lines.append("Use mem_list to view more, mem_get to open full content by mem_id.")

                rt.latest_obs = "\n\n".join([l for l in lines if l]).strip()
                ctx.trace(
                    "mem_search",
                    {
                        "ts": _now_ts(),
                        "agent": rt.node_ptr.role,
                        "query": query,
                        "top_k": top_k,
                        "role": role,
                        "status": status,
                        "mem_type": mem_type,
                        "results": [
                            {
                                "mem_id": rt.mem_id_to_item[m].mem_id,
                                "role": rt.mem_id_to_item[m].role,
                                "type": rt.mem_id_to_item[m].type,
                                "status": rt.mem_id_to_item[m].status,
                                "source_run_id": rt.mem_id_to_item[m].source_run_id,
                                "distance": distance_by_id.get(m),
                                "matched_claims": matched_by_id.get(m, [])[:6],
                            }
                            for m in mem_ids
                            if m in rt.mem_id_to_item
                        ],
                    },
                )

                rt.remaining_subtasks = info.subtasks[1:]
                rt.state = RecapState.ACTION_TAKEN
                continue

            if stype == "mem_get":
                mem_id = str(first.get("mem_id") or "").strip()
                if mem_id.startswith("mem:"):
                    mem_id = mem_id[4:].strip()

                stored = rt.mem_id_to_item.get(mem_id)
                if stored is None:
                    rt.latest_obs = (
                        f"ERROR: Unknown mem_id: {mem_id!r}.\n"
                        "You can only mem_get a mem_id that was returned by a prior mem_search in this run."
                    )
                else:
                    _merge_focus_mem_ids(rt, [stored.mem_id])
                    rt.latest_obs = (
                        "MEM get (from run memory registry):\n"
                        f"mem:{stored.mem_id} role={stored.role} type={stored.type} status={stored.status}\n"
                        f"{stored.content}\n"
                    ).strip()
                    ctx.trace(
                        "mem_get",
                        {
                            "ts": _now_ts(),
                            "agent": rt.node_ptr.role,
                            "mem_id": stored.mem_id,
                            "role": stored.role,
                            "type": stored.type,
                            "status": stored.status,
                            "source_run_id": stored.source_run_id,
                        },
                    )

                rt.remaining_subtasks = info.subtasks[1:]
                rt.state = RecapState.ACTION_TAKEN
                continue

            if stype == "mem_list":
                total = len(rt.mem_all_items)
                default_limit = int(cfg.reasoningbank.mem_list_default_limit)
                max_limit = int(cfg.reasoningbank.mem_list_max_limit)
                limit_raw = first.get("limit")
                try:
                    limit = int(limit_raw) if limit_raw is not None else default_limit
                except Exception:
                    limit = default_limit
                if limit < 1:
                    limit = 1
                if limit > max_limit:
                    limit = max_limit

                shown = rt.mem_all_items[:limit]
                lines: list[str] = []
                lines.append(f"Run memory registry: {total} memories total.")
                if total == 0:
                    lines.append("(empty; run mem_search first)")
                else:
                    lines.append(f"Showing {len(shown)}/{total} (limit={limit}).")
                    lines.append("")
                    for it in shown:
                        snippet = it.content.replace("\n", " ").strip()
                        if len(snippet) > 140:
                            snippet = snippet[:140] + "…"
                        lines.append(f"mem:{it.mem_id} role={it.role} type={it.type} :: {snippet}")

                rt.latest_obs = "\n".join(lines).strip()
                ctx.trace(
                    "mem_list",
                    {
                        "ts": _now_ts(),
                        "agent": rt.node_ptr.role,
                        "total": total,
                        "limit": limit,
                        "shown_mem_ids": [it.mem_id for it in shown],
                    },
                )

                rt.remaining_subtasks = info.subtasks[1:]
                rt.state = RecapState.ACTION_TAKEN
                continue

            if stype == "task":
                role = str(first.get("role") or "orchestrator").strip() or "orchestrator"
                task = str(first.get("task") or "").strip()
                if not task:
                    rt.latest_obs = f"ERROR: Empty task in subtask: {json.dumps(first, ensure_ascii=False)}"
                    rt.remaining_subtasks = info.subtasks[1:]
                    rt.state = RecapState.ACTION_TAKEN
                    continue

                if rt.depth + 1 > int(cfg.recap.max_depth):
                    raise RecapError(f"Exceeded recap.max_depth={cfg.recap.max_depth}")

                child = Node(task_name=task, role=role, parent=rt.node_ptr)
                rt.node_ptr.add_child(child)
                rt.node_ptr = child
                rt.depth += 1
                rt.state = RecapState.DOWN
                continue

            # Should not happen (parser validates), but keep a safe fallback.
            rt.latest_obs = (
                "ERROR: Unknown subtask type. Expected one of "
                "['task','kb_search','kb_get','kb_list','mem_search','mem_get','mem_list','generate_recipes'].\n"
                f"Got: {json.dumps(first, ensure_ascii=False)}"
            )
            rt.remaining_subtasks = info.subtasks[1:]
            rt.state = RecapState.ACTION_TAKEN
            continue
