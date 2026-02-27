from __future__ import annotations

import json
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

from src.agents.orchestrator import OrchestratorAgent
from src.agents.types import AgentContext
from src.config.load_config import load_app_config
from src.llm.openai_compat import OpenAICompatibleChatClient
from src.runtime.dry_run_simulation import append_dry_run_simulation
from src.runtime.reasoningbank_learn import safe_execute_rb_learn_job
from src.storage.sqlite_store import SQLiteStore, default_db_path
from src.storage.reasoningbank_store import ReasoningBankDependencyError, ReasoningBankError, ReasoningBankStore
from src.tools.kb_registry import KnowledgeBases
from src.utils.cancel import CancelledError, CancellationToken


@dataclass(frozen=True)
class WorkerConfig:
    poll_interval_s: float = 0.5


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterable[None]:
    """Temporarily override os.environ keys (restores on exit)."""
    prev: dict[str, str | None] = {}
    try:
        for k, v in overrides.items():
            prev[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


class RunWorker:
    """Single-threaded background worker that executes queued runs."""

    def __init__(self, *, db_path: str | None = None, config: WorkerConfig | None = None) -> None:
        self._db_path = db_path or default_db_path()
        self._config = config or WorkerConfig()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._orchestrator = OrchestratorAgent()
        self._app_config = load_app_config()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "poll_interval_s": float(self._config.poll_interval_s),
            "db_path": self._db_path,
        }

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="c2xc-run-worker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is None:
            return
        t.join(timeout=timeout_s)

    def _run_loop(self) -> None:
        store = SQLiteStore(self._db_path)
        try:
            while not self._stop.is_set():
                claimed = store.claim_next_queued_run()
                if claimed is None:
                    # No runs queued; try ReasoningBank jobs.
                    rb_claimed = store.claim_next_queued_rb_job()
                    if rb_claimed is None:
                        time.sleep(self._config.poll_interval_s)
                        continue
                    try:
                        self._execute_rb_job(store, rb_claimed)
                    except Exception as e:
                        # Extremely defensive: never crash the worker loop.
                        run_id = str(rb_claimed["run_id"])
                        store.append_event(
                            run_id,
                            "rb_job_failed",
                            {"error": f"rb_job_unhandled_exception: {e}", "traceback": traceback.format_exc()},
                        )
                        store.update_rb_job_status(str(rb_claimed["rb_job_id"]), "failed", error=str(e))
                    continue

                run_id = str(claimed["run_id"])
                batch_id = str(claimed["batch_id"])

                try:
                    self._execute_one(store, run_id=run_id, batch_id=batch_id)
                except Exception as e:
                    # Extremely defensive: never crash the worker loop.
                    store.append_event(
                        run_id,
                        "run_failed",
                        {"error": f"worker_unhandled_exception: {e}", "traceback": traceback.format_exc()},
                    )
                    store.update_run_status(run_id, "failed", error=str(e))

                self._update_batch_terminal_status_if_done(store, batch_id=batch_id)
        finally:
            store.close()

    def _execute_rb_job(self, store: SQLiteStore, job_row: Any) -> None:
        rb_job_id = str(job_row["rb_job_id"])
        run_id = str(job_row["run_id"])
        kind = str(job_row["kind"])

        # Resolve per-run config snapshot so RB learn is reproducible, while still allowing
        # RB-learn-specific overrides (different model/base than the main run).
        config_snapshot: dict[str, Any] = {}
        try:
            run_row = store.get_run(run_id=run_id)
            if run_row is not None:
                batch_row = store.get_batch(batch_id=str(run_row["batch_id"]))
                if batch_row is not None:
                    config_snapshot = json.loads(str(batch_row["config_json"] or "{}"))
        except Exception:
            config_snapshot = {}
        overrides = dict(config_snapshot.get("overrides", {}) or {})

        env_overrides: dict[str, str] = {}

        # Embeddings (used by Chroma collection).
        embedding_model = str(config_snapshot.get("embedding_model") or overrides.get("embedding_model") or "").strip()
        embedding_api_base = str(config_snapshot.get("embedding_api_base") or overrides.get("embedding_api_base") or "").strip()
        embedding_dim = str(config_snapshot.get("embedding_dim") or overrides.get("embedding_dim") or "").strip()
        embedding_send_dimensions = str(
            config_snapshot.get("embedding_send_dimensions") or overrides.get("embedding_send_dimensions") or ""
        ).strip()
        if embedding_model:
            env_overrides["C2XC_EMBEDDING_MODEL"] = embedding_model
        if embedding_api_base:
            env_overrides["C2XC_EMBEDDING_API_BASE"] = embedding_api_base
        if embedding_dim:
            env_overrides["C2XC_EMBEDDING_DIM"] = embedding_dim
        if embedding_send_dimensions:
            env_overrides["C2XC_EMBEDDING_SEND_DIMENSIONS"] = embedding_send_dimensions

        # RB learn LLM config: prefer rb_* overrides, fall back to main run config.
        run_llm_model = str(config_snapshot.get("llm_model") or overrides.get("llm_model") or "").strip()
        run_openai_api_base = str(config_snapshot.get("openai_api_base") or overrides.get("openai_api_base") or "").strip()
        run_llm_reasoning_effort = str(
            config_snapshot.get("llm_reasoning_effort") or overrides.get("llm_reasoning_effort") or ""
        ).strip()
        run_llm_verbosity = str(config_snapshot.get("llm_verbosity") or overrides.get("llm_verbosity") or "").strip()
        rb_llm_model = str(config_snapshot.get("rb_llm_model") or overrides.get("rb_llm_model") or "").strip() or run_llm_model
        rb_openai_api_base = str(config_snapshot.get("rb_openai_api_base") or overrides.get("rb_openai_api_base") or "").strip() or run_openai_api_base
        if rb_llm_model:
            env_overrides["LLM_MODEL"] = rb_llm_model
        if run_llm_reasoning_effort:
            env_overrides["C2XC_LLM_REASONING_EFFORT"] = run_llm_reasoning_effort
        if run_llm_verbosity:
            env_overrides["C2XC_LLM_VERBOSITY"] = run_llm_verbosity
        if rb_openai_api_base:
            env_overrides["OPENAI_API_BASE"] = rb_openai_api_base

        # Job row is already claimed/running. Record a trace event for UI observability.
        store.append_event(
            run_id,
            "rb_job_started",
            {
                "rb_job_id": rb_job_id,
                "kind": kind,
                "llm_model": rb_llm_model,
                "llm_reasoning_effort": run_llm_reasoning_effort,
                "llm_verbosity": run_llm_verbosity,
                "openai_api_base": rb_openai_api_base,
                "embedding_model": embedding_model,
                "embedding_api_base": embedding_api_base,
            },
        )

        if kind != "learn":
            store.append_event(
                run_id,
                "rb_job_failed",
                {"rb_job_id": rb_job_id, "kind": kind, "error": f"Unknown RB job kind: {kind!r}"},
            )
            store.update_rb_job_status(rb_job_id, "failed", error=f"Unknown kind: {kind}")
            return

        with _temporary_env(env_overrides):
            try:
                rb = ReasoningBankStore.from_config(self._app_config)
            except ReasoningBankDependencyError as e:
                store.append_event(
                    run_id,
                    "rb_job_failed",
                    {"rb_job_id": rb_job_id, "kind": kind, "error": str(e), "missing": e.missing},
                )
                store.update_rb_job_status(rb_job_id, "failed", error=str(e))
                return
            except ReasoningBankError as e:
                store.append_event(run_id, "rb_job_failed", {"rb_job_id": rb_job_id, "kind": kind, "error": str(e)})
                store.update_rb_job_status(rb_job_id, "failed", error=str(e))
                return

            # LLM is required unless RB learn is configured for dry-run mode.
            llm: OpenAICompatibleChatClient | None
            try:
                llm = OpenAICompatibleChatClient()
            except Exception:
                llm = None

            delta_id = safe_execute_rb_learn_job(
                store,
                rb=rb,
                cfg=self._app_config,
                llm=llm,
                run_id=run_id,
                rb_job_id=rb_job_id,
            )
            if delta_id is None:
                store.update_rb_job_status(rb_job_id, "failed", error="rb_learn_failed")
                return

            store.append_event(
                run_id,
                "rb_job_completed",
                {"rb_job_id": rb_job_id, "kind": kind, "delta_id": delta_id},
            )
            store.update_rb_job_status(rb_job_id, "completed")

    def _execute_one(self, store: SQLiteStore, *, run_id: str, batch_id: str) -> None:
        run_row = store.get_run(run_id=run_id)
        batch_row = store.get_batch(batch_id=batch_id)
        if run_row is None or batch_row is None:
            store.append_event(run_id, "run_failed", {"error": "Missing run/batch record in DB."})
            store.update_run_status(run_id, "failed", error="Missing run/batch record")
            return

        # Cancellation checks before heavy init.
        if store.is_cancel_requested(target_type="batch", target_id=batch_id):
            store.acknowledge_cancel(target_type="batch", target_id=batch_id)
            store.append_event(run_id, "run_canceled", {"reason": "batch_cancel_requested"})
            store.update_run_status(run_id, "canceled")
            return
        if store.is_cancel_requested(target_type="run", target_id=run_id):
            store.acknowledge_cancel(target_type="run", target_id=run_id)
            store.append_event(run_id, "run_canceled", {"reason": "cancel_requested"})
            store.update_run_status(run_id, "canceled")
            return

        user_request = str(batch_row["user_request"])
        recipes_per_run = int(batch_row["recipes_per_run"])

        config_snapshot = json.loads(str(batch_row["config_json"]))
        temperature = float(config_snapshot.get("temperature", 0.7))
        dry_run = bool(config_snapshot.get("dry_run", False))
        overrides = dict(config_snapshot.get("overrides", {}) or {})

        mode = "dry_run" if dry_run else "normal"
        store.append_event(
            run_id,
            "run_started",
            {
                "mode": mode,
                "user_request": user_request,
                "run_index": int(run_row["run_index"]),
                "n_runs": int(batch_row["n_runs"]),
                "recipes_per_run": recipes_per_run,
                "temperature": temperature,
            },
        )

        cancel = CancellationToken()
        try:
            if dry_run:
                recipes_json, citations = append_dry_run_simulation(
                    store,
                    run_id=run_id,
                    user_request=user_request,
                    recipes_per_run=recipes_per_run,
                    temperature=temperature,
                    run_index=int(run_row["run_index"]),
                    n_runs=int(batch_row["n_runs"]),
                    alias_prefix=str(self._app_config.citations.alias_prefix),
                )
                store.append_event(
                    run_id,
                    "final_output",
                    {"recipes_json": recipes_json, "citations": citations, "memory_ids": []},
                )
                store.update_run_status(run_id, "completed")
                return

            env_overrides: dict[str, str] = {}
            kb_principles_dir = str(
                config_snapshot.get("kb_principles_dir") or overrides.get("kb_principles_dir") or ""
            ).strip()
            kb_modulation_dir = str(
                config_snapshot.get("kb_modulation_dir") or overrides.get("kb_modulation_dir") or ""
            ).strip()
            llm_model = str(config_snapshot.get("llm_model") or overrides.get("llm_model") or "").strip()
            llm_reasoning_effort = str(
                config_snapshot.get("llm_reasoning_effort") or overrides.get("llm_reasoning_effort") or ""
            ).strip()
            llm_verbosity = str(config_snapshot.get("llm_verbosity") or overrides.get("llm_verbosity") or "").strip()
            openai_api_base = str(
                config_snapshot.get("openai_api_base") or overrides.get("openai_api_base") or ""
            ).strip()
            embedding_model = str(config_snapshot.get("embedding_model") or overrides.get("embedding_model") or "").strip()
            embedding_api_base = str(
                config_snapshot.get("embedding_api_base") or overrides.get("embedding_api_base") or ""
            ).strip()
            embedding_dim = str(config_snapshot.get("embedding_dim") or overrides.get("embedding_dim") or "").strip()
            embedding_send_dimensions = str(
                config_snapshot.get("embedding_send_dimensions") or overrides.get("embedding_send_dimensions") or ""
            ).strip()
            if kb_principles_dir:
                env_overrides["LIGHTRAG_KB_PRINCIPLES_DIR"] = kb_principles_dir
            if kb_modulation_dir:
                env_overrides["LIGHTRAG_KB_MODULATION_DIR"] = kb_modulation_dir
            if llm_model:
                env_overrides["LLM_MODEL"] = llm_model
            if llm_reasoning_effort:
                env_overrides["C2XC_LLM_REASONING_EFFORT"] = llm_reasoning_effort
            if llm_verbosity:
                env_overrides["C2XC_LLM_VERBOSITY"] = llm_verbosity
            if openai_api_base:
                env_overrides["OPENAI_API_BASE"] = openai_api_base
            if embedding_model:
                env_overrides["C2XC_EMBEDDING_MODEL"] = embedding_model
            if embedding_api_base:
                env_overrides["C2XC_EMBEDDING_API_BASE"] = embedding_api_base
            if embedding_dim:
                env_overrides["C2XC_EMBEDDING_DIM"] = embedding_dim
            if embedding_send_dimensions:
                env_overrides["C2XC_EMBEDDING_SEND_DIMENSIONS"] = embedding_send_dimensions

            with _temporary_env(env_overrides):
                kbs = KnowledgeBases.from_env()
                llm = OpenAICompatibleChatClient()
                rb = None
                try:
                    rb = ReasoningBankStore.from_config(self._app_config)
                except Exception as e:
                    # RB is optional for normal runs; if unavailable, proceed with KB-only.
                    store.append_event(run_id, "rb_unavailable", {"error": str(e)})

                ctx = AgentContext(
                    store=store,
                    config=self._app_config,
                    kbs=kbs,
                    rb=rb,
                    llm=llm,
                    cancel=cancel,
                    batch_id=batch_id,
                    run_id=run_id,
                    recipes_per_run=recipes_per_run,
                    temperature=temperature,
                )

                result = self._orchestrator.run(ctx, user_request=user_request)
                store.append_event(
                    run_id,
                    "final_output",
                    {
                        "recipes_json": result.recipes_json,
                        "citations": result.citations,
                        "memory_ids": result.memory_ids,
                    },
                )
                store.update_run_status(run_id, "completed")
        except CancelledError:
            cancel.request_cancel()
            store.append_event(run_id, "run_canceled", {"reason": "cancel_requested"})
            store.update_run_status(run_id, "canceled")
        except Exception as e:
            store.append_event(run_id, "run_failed", {"error": str(e), "traceback": traceback.format_exc()})
            store.update_run_status(run_id, "failed", error=str(e))

    def _update_batch_terminal_status_if_done(self, store: SQLiteStore, *, batch_id: str) -> None:
        runs = store.list_runs_for_batch_rows(batch_id=batch_id)
        if not runs:
            return
        statuses = [str(r["status"]) for r in runs]
        if any(s in {"queued", "running"} for s in statuses):
            return

        if any(s == "failed" for s in statuses):
            store.update_batch_status(batch_id, "failed")
        elif any(s == "canceled" for s in statuses):
            store.update_batch_status(batch_id, "canceled")
        else:
            store.update_batch_status(batch_id, "completed")
