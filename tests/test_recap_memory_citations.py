from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from src.agents.types import AgentContext
from src.config.load_config import load_app_config
from src.llm.openai_compat import ChatCompletionResult
from src.recap.engine import RecapEngine
from src.storage.reasoningbank_store import ReasoningBankStore
from src.storage.sqlite_store import SQLiteStore
from src.utils.cancel import CancellationToken


class _EmptyKB:
    def query_chunks(self, query: str, *, mode: str, top_k: int) -> list[Any]:
        return []


@dataclass(frozen=True)
class _DummyKBs:
    kb_principles: Any
    kb_modulation: Any


class _FakeLLM:
    model = "fake-llm"

    def __init__(self) -> None:
        self._planned_mem_search = False
        self._done_tio2 = False
        self._done_mof = False

    def chat_messages(self, *, messages: list[dict[str, Any]], temperature: float, extra: dict[str, Any] | None = None) -> ChatCompletionResult:  # noqa: D401,E501
        extra = extra or {}

        # ReCAP planning/refinement calls.
        rf = extra.get("response_format") or {}
        schema_name = ((rf.get("json_schema") or {}) if isinstance(rf, dict) else {}).get("name")
        if schema_name == "recap_response":
            # Determine which role we are currently acting as from the injected role instructions.
            # (The engine uses a shared chat history but injects role instructions into the user prompt.)
            prompt = str((messages[-1] or {}).get("content") or "")
            is_tio2 = "Role: tio2_expert" in prompt
            is_mof = "Role: mof_expert" in prompt

            if is_tio2:
                self._done_tio2 = True
                report = {
                    "schema": "tio2_mechanisms_report_v1",
                    "mechanisms": [
                        {"id": i, "impact": "supporting", "justification": "synthetic"} for i in range(1, 8)
                    ],
                    "synthesis": "synthetic",
                }
                content = json.dumps({"think": "", "subtasks": [], "result": report})
                return ChatCompletionResult(content=content, raw={}, tool_calls=[])

            if is_mof:
                self._done_mof = True
                report = {
                    "schema": "mof_roles_report_v1",
                    "roles": [
                        {"id": i, "impact": "minor", "justification": "synthetic"} for i in range(1, 11)
                    ],
                    "synthesis": "synthetic",
                }
                content = json.dumps({"think": "", "subtasks": [], "result": report})
                return ChatCompletionResult(content=content, raw={}, tool_calls=[])

            # Orchestrator: strict gating now requires accepted expert deliverables before generate_recipes.
            subtasks: list[dict[str, Any]]
            if not self._done_tio2:
                subtasks = [{"type": "task", "role": "tio2_expert", "task": "synthetic tio2 report"}]
            elif not self._done_mof:
                subtasks = [{"type": "task", "role": "mof_expert", "task": "synthetic mof report"}]
            elif not self._planned_mem_search:
                self._planned_mem_search = True
                subtasks = [{"type": "mem_search", "query": "synthetic", "top_k": 5}]
            else:
                subtasks = [{"type": "generate_recipes"}]

            content = json.dumps({"think": "", "subtasks": subtasks, "result": ""})
            return ChatCompletionResult(content=content, raw={}, tool_calls=[])

        # generate_recipes tool loop (tool calling).
        if extra.get("tool_choice") == "auto" and extra.get("tools"):
            # We already have memory evidence from the primitive mem_search step; no tools needed.
            return ChatCompletionResult(content="", raw={}, tool_calls=[])

        # Final generate_recipes JSON output (schema enforced).
        if schema_name == "generate_recipes_output":
            mem_ids: list[str] = []
            for m in messages:
                # Ignore the system prompt, which contains an example mem:<uuid>.
                if m.get("role") == "system":
                    continue
                text = str(m.get("content") or "")
                for match in re.finditer(
                    r"mem:(?P<mem_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
                    text,
                ):
                    mem_ids.append(match.group("mem_id"))

            assert mem_ids, "expected at least one real mem:<id> in non-system messages"
            mem_id = mem_ids[-1]

            out = {
                "recipes": [
                    {
                        "M1": "Cu",
                        "M2": "Ag",
                        "atomic_ratio": "1:1",
                        "small_molecule_modifier": "acetic acid (-COOH)",
                        "rationale": f"Use prior experience mem:{mem_id} to bias towards C2H4 selectivity.",
                    }
                ],
                "overall_notes": "",
            }
            return ChatCompletionResult(content=json.dumps(out), raw={}, tool_calls=[])

        raise AssertionError(f"Unexpected LLM call. extra={extra!r}")


def test_recap_engine_can_retrieve_and_cite_memories(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "app.db")
        chroma_dir = os.path.join(td, "chroma")

        monkeypatch.setenv("C2XC_SQLITE_PATH", db_path)
        monkeypatch.setenv("C2XC_RB_CHROMA_DIR", chroma_dir)
        monkeypatch.setenv("C2XC_RB_EMBEDDING_MODE", "hash")

        cfg = load_app_config()
        rb = ReasoningBankStore.from_config(cfg)
        seeded = rb.upsert(
            mem_id=None,
            status="active",
            role="global",
            type="manual_note",
            content="synthetic memory: Cu-Ag may improve C2H4 selectivity.",
            source_run_id=None,
            schema_version=1,
            extra={},
            preserve_created_at=True,
        )

        store = SQLiteStore(db_path)
        try:
            batch = store.create_batch(
                user_request="test",
                n_runs=1,
                recipes_per_run=1,
                config={"dry_run": True},
            )
            run = store.create_run(batch_id=batch.batch_id, run_index=1)

            ctx = AgentContext(
                store=store,
                config=cfg,
                kbs=_DummyKBs(kb_principles=_EmptyKB(), kb_modulation=_EmptyKB()),
                rb=rb,
                llm=_FakeLLM(),  # type: ignore[arg-type]
                cancel=CancellationToken(),
                batch_id=batch.batch_id,
                run_id=run.run_id,
                recipes_per_run=1,
                temperature=0.1,
            )

            recipes_json, citations, mem_ids = RecapEngine().run(ctx, user_request="test request")
            assert citations == {}
            assert mem_ids == [seeded.mem_id]

            rationale = str((recipes_json.get("recipes") or [{}])[0].get("rationale") or "")
            assert f"mem:{seeded.mem_id}" in rationale

            resolved = store.get_latest_event(run_id=run.run_id, event_type="memories_resolved")
            assert resolved is not None
        finally:
            store.close()
