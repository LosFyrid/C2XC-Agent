from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

import pytest

from src.config.load_config import load_app_config
from src.llm.openai_compat import ChatCompletionResult
from src.runtime.reasoningbank_learn import learn_reasoningbank_for_run
from src.storage.reasoningbank_store import ReasoningBankStore
from src.storage.sqlite_store import SQLiteStore


class _FakeRBLearnLLM:
    model = "fake-llm"
    base_url = "http://fake.local"

    def __init__(self, *, store: SQLiteStore, run_id: str, forbidden_event_id: str) -> None:
        self._store = store
        self._run_id = run_id
        self._forbidden_event_id = forbidden_event_id
        self.out_of_bounds_event_id: str | None = None
        self._turn = 0

    def chat_messages(
        self, *, messages: list[dict[str, Any]], temperature: float, extra: dict[str, Any] | None = None
    ) -> ChatCompletionResult:
        extra = extra or {}

        if extra.get("tool_choice") == "auto" and extra.get("tools"):
            if self._turn == 0:
                # Insert an event after the RB learn snapshot cutoff so rb_open_event must reject it.
                time.sleep(0.01)
                self.out_of_bounds_event_id = self._store.append_event(
                    self._run_id,
                    "recap_info",
                    {"ts": time.time(), "agent": "orchestrator", "recap_state": "post_snapshot", "task_name": "test"},
                )
                self._turn += 1
                tool_calls = [
                    {
                        "id": "call_forbidden",
                        "type": "function",
                        "function": {
                            "name": "rb_open_event",
                            "arguments": json.dumps(
                                {
                                    "event_id": self._forbidden_event_id,
                                    "mode": "full",
                                    "reason": "attempt forbidden llm_request",
                                }
                            ),
                        },
                    },
                    {
                        "id": "call_oob",
                        "type": "function",
                        "function": {
                            "name": "rb_open_event",
                            "arguments": json.dumps(
                                {
                                    "event_id": self.out_of_bounds_event_id,
                                    "mode": "full",
                                    "reason": "attempt snapshot out-of-bounds",
                                }
                            ),
                        },
                    },
                ]
                return ChatCompletionResult(content="", raw={}, tool_calls=tool_calls)

            # After one tool round, output a minimal valid extractor result.
            rb_content = "\n".join(
                [
                    "RBMEM_CLAIMS_V1",
                    "TOPIC=test memory (deref tools policy)",
                    "SCOPE=global",
                    "CLAIMS_JSON="
                    + json.dumps(
                        [
                            {
                                "claim_id": "c1",
                                "status": "fact",
                                "facts": {"source_run_ids": [self._run_id]},
                                "inference": {
                                    "summary": "dereference tools should respect facts-only + snapshot cutoff."
                                },
                                "constraint": {"avoid": [], "allow_positive": False},
                                "conditions": [],
                                "limitations": [],
                                "support": {"count": 0, "run_ids": []},
                                "contra": {"count": 0, "run_ids": []},
                            }
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "",
                ]
            )
            out = {
                "items": [
                    {
                        "role": "global",
                        "type": "reasoningbank_item",
                        "content": rb_content,
                        "extra": {"test": True},
                    }
                ]
            }
            return ChatCompletionResult(content=json.dumps(out), raw={}, tool_calls=[])

        raise AssertionError(f"Unexpected LLM call in RB learn. extra={extra!r}")


def test_rb_learn_deref_blocks_llm_logs_and_snapshot_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "app.db")
        chroma_dir = os.path.join(td, "chroma")

        monkeypatch.setenv("C2XC_SQLITE_PATH", db_path)
        monkeypatch.setenv("C2XC_RB_CHROMA_DIR", chroma_dir)
        monkeypatch.setenv("C2XC_RB_EMBEDDING_MODE", "hash")

        cfg = load_app_config()
        rb = ReasoningBankStore.from_config(cfg)

        store = SQLiteStore(db_path)
        try:
            batch = store.create_batch(
                user_request="test",
                n_runs=1,
                recipes_per_run=1,
                config={"dry_run": True},
            )
            run = store.create_run(batch_id=batch.batch_id, run_index=1)

            # Minimal run output + feedback (required by RB learn).
            store.append_event(run.run_id, "final_output", {"recipes_json": {"recipes": []}})

            # Feedback can be product-less for this test; keep it minimal.
            store.upsert_feedback_for_run(
                run_id=run.run_id,
                score=7.0,
                pros="good",
                cons="",
                other="",
                products=[],
            )

            forbidden_event_id = store.append_event(
                run.run_id,
                "llm_request",
                {"ts": time.time(), "model": "fake", "prompt": "do not learn from me"},
            )

            llm = _FakeRBLearnLLM(store=store, run_id=run.run_id, forbidden_event_id=forbidden_event_id)

            learn_reasoningbank_for_run(
                store,
                rb=rb,
                cfg=cfg,
                llm=llm,  # type: ignore[arg-type]
                run_id=run.run_id,
                rb_job_id="rb_job_test",
            )

            assert llm.out_of_bounds_event_id, "expected fake LLM to insert an out-of-bounds event"

            # Validate snapshot cutoff semantics.
            snapshot_row = store.get_latest_event(run_id=run.run_id, event_type="rb_learn_snapshot")
            assert snapshot_row is not None
            snapshot_payload = json.loads(str(snapshot_row["payload_json"]))
            cutoff = float((snapshot_payload.get("snapshot") or {}).get("trace_cutoff_ts") or 0.0)
            assert cutoff > 0

            oob_row = store.get_event(run_id=run.run_id, event_id=str(llm.out_of_bounds_event_id))
            assert oob_row is not None
            assert float(oob_row["created_at"]) > cutoff

            # Validate facts-only policy + error observability.
            opened = store.list_latest_events(
                run_id=run.run_id,
                limit=50,
                event_types=["rb_source_opened"],
                include_payload=True,
            )
            codes = {str((e.get("payload") or {}).get("error_code") or "") for e in opened}
            assert "forbidden_event_type" in codes
            assert "snapshot_out_of_bounds" in codes
        finally:
            store.close()
