from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any

from src.agents.orchestrator import OrchestratorAgent
from src.agents.types import AgentContext
from src.config.load_config import default_config_path, load_app_config
from src.llm.openai_compat import OpenAICompatibleChatClient
from src.runtime.dry_run_simulation import append_dry_run_simulation
from src.storage.reasoningbank_store import ReasoningBankStore
from src.storage.sqlite_store import SQLiteStore
from src.tools.kb_registry import KnowledgeBases
from src.utils.cancel import CancelledError, CancellationToken


def _clamp_int(name: str, value: int, *, min_v: int, max_v: int) -> int:
    if value < min_v or value > max_v:
        raise SystemExit(f"{name} must be in [{min_v}..{max_v}], got {value}")
    return value


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C2XC-Agent batch (agent-first CLI).")
    parser.add_argument(
        "--request",
        default="",
        help="Optional user request. If omitted, a sensible default is used.",
    )
    parser.add_argument("--n-runs", type=int, default=1, help="Number of runs to queue (1..5).")
    parser.add_argument(
        "--recipes-per-run",
        type=int,
        default=3,
        help="Number of recipes per run (1..3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=(
            "Legacy sampling temperature (kept for trace/replay metadata). "
            "NOTE: when using reasoning_effort!=none (e.g. GPT-5), temperature is not sent to the API."
        ),
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="SQLite path (default: env C2XC_SQLITE_PATH or data/app.db).",
    )
    parser.add_argument("--kb-principles-dir", default="", help="Override LIGHTRAG_KB_PRINCIPLES_DIR.")
    parser.add_argument("--kb-modulation-dir", default="", help="Override LIGHTRAG_KB_MODULATION_DIR.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call KB/LLM. Writes a placeholder output to SQLite for pipeline testing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    app_config = load_app_config()

    n_runs = _clamp_int("n_runs", int(args.n_runs), min_v=1, max_v=app_config.limits.n_runs_max)
    recipes_per_run = _clamp_int(
        "recipes_per_run",
        int(args.recipes_per_run),
        min_v=1,
        max_v=app_config.limits.recipes_per_run_max,
    )
    temperature = float(args.temperature)

    if args.kb_principles_dir:
        os.environ["LIGHTRAG_KB_PRINCIPLES_DIR"] = args.kb_principles_dir
    if args.kb_modulation_dir:
        os.environ["LIGHTRAG_KB_MODULATION_DIR"] = args.kb_modulation_dir

    user_request = (args.request or "").strip()
    if not user_request:
        user_request = (
            "Generate catalyst recipes for photocatalytic CO2 reduction/coupling. "
            "Primary objective: high selectivity and high activity for ethylene (C2H4). "
            "System is M1M2–TiO2 / Zr-BTB with fixed BTB linker; small_molecule_modifier must contain -COOH."
        )

    store = SQLiteStore(args.db_path or None)

    cancel = CancellationToken()
    try:
        config_snapshot: dict[str, Any] = {
            "config_path": str(default_config_path()),
            "n_runs": n_runs,
            "recipes_per_run": recipes_per_run,
            "temperature": temperature,
            "kb_principles_dir": os.getenv("LIGHTRAG_KB_PRINCIPLES_DIR", ""),
            "kb_modulation_dir": os.getenv("LIGHTRAG_KB_MODULATION_DIR", ""),
            "llm_model": os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "")),
            "llm_reasoning_effort": os.getenv("C2XC_LLM_REASONING_EFFORT", ""),
            "llm_verbosity": os.getenv("C2XC_LLM_VERBOSITY", ""),
            "openai_api_base": os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", "")),
            "rb_llm_model": os.getenv("C2XC_RB_LEARN_LLM_MODEL", ""),
            "rb_openai_api_base": os.getenv("C2XC_RB_LEARN_OPENAI_API_BASE", ""),
            "embedding_model": os.getenv("C2XC_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "")),
            "embedding_api_base": os.getenv(
                "C2XC_EMBEDDING_API_BASE",
                os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", ""))),
            ),
            "embedding_dim": os.getenv("C2XC_EMBEDDING_DIM", os.getenv("EMBEDDING_DIM", "")),
            "embedding_send_dimensions": os.getenv(
                "C2XC_EMBEDDING_SEND_DIMENSIONS",
                os.getenv("EMBEDDING_SEND_DIMENSIONS", ""),
            ),
        }
        batch = store.create_batch(
            user_request=user_request,
            n_runs=n_runs,
            recipes_per_run=recipes_per_run,
            config=config_snapshot,
        )
        store.update_batch_status(batch.batch_id, "running")

        runs = [store.create_run(batch_id=batch.batch_id, run_index=i + 1) for i in range(n_runs)]

        orchestrator = OrchestratorAgent()

        if args.dry_run:
            for r in runs:
                store.update_run_status(r.run_id, "running")
                store.append_event(
                    r.run_id,
                    "run_started",
                    {
                        "mode": "dry_run",
                        "user_request": user_request,
                        "run_index": int(r.run_index),
                        "n_runs": n_runs,
                        "recipes_per_run": recipes_per_run,
                        "temperature": temperature,
                    },
                )
                recipes_json, citations = append_dry_run_simulation(
                    store,
                    run_id=r.run_id,
                    user_request=user_request,
                    recipes_per_run=recipes_per_run,
                    temperature=temperature,
                    run_index=int(r.run_index),
                    n_runs=n_runs,
                    alias_prefix=str(app_config.citations.alias_prefix),
                )
                store.append_event(
                    r.run_id,
                    "final_output",
                    {"recipes_json": recipes_json, "citations": citations, "memory_ids": []},
                )
                store.update_run_status(r.run_id, "completed")
            store.update_batch_status(batch.batch_id, "completed")
            return 0

        # Normal mode: requires KB + LLM.
        try:
            kbs = KnowledgeBases.from_env()
            llm = OpenAICompatibleChatClient()
            rb = None
            try:
                rb = ReasoningBankStore.from_config(app_config)
            except Exception:
                rb = None
        except Exception as e:
            err = f"Initialization failed: {e}"
            for r in runs:
                store.append_event(r.run_id, "run_failed", {"error": err})
                store.update_run_status(r.run_id, "failed", error=err)
            store.update_batch_status(batch.batch_id, "failed", error=err)
            return 1

        any_failed = False
        any_canceled = False

        for r in runs:
            if cancel.cancelled:
                store.update_run_status(r.run_id, "canceled")
                any_canceled = True
                continue

            if store.is_cancel_requested(target_type="batch", target_id=batch.batch_id):
                store.acknowledge_cancel(target_type="batch", target_id=batch.batch_id)
                store.append_event(r.run_id, "run_canceled", {"reason": "batch_cancel_requested"})
                store.update_run_status(r.run_id, "canceled")
                any_canceled = True
                cancel.request_cancel()
                continue

            if store.is_cancel_requested(target_type="run", target_id=r.run_id):
                store.acknowledge_cancel(target_type="run", target_id=r.run_id)
                store.append_event(r.run_id, "run_canceled", {"reason": "cancel_requested"})
                store.update_run_status(r.run_id, "canceled")
                any_canceled = True
                continue

            store.update_run_status(r.run_id, "running")
            store.append_event(
                r.run_id,
                "run_started",
                {
                    "user_request": user_request,
                    "run_index": r.run_index,
                    "n_runs": n_runs,
                    "recipes_per_run": recipes_per_run,
                    "temperature": temperature,
                },
            )

            ctx = AgentContext(
                store=store,
                config=app_config,
                kbs=kbs,
                rb=rb,
                llm=llm,
                cancel=cancel,
                batch_id=batch.batch_id,
                run_id=r.run_id,
                recipes_per_run=recipes_per_run,
                temperature=temperature,
            )

            try:
                result = orchestrator.run(ctx, user_request=user_request)
                store.append_event(
                    r.run_id,
                    "final_output",
                    {
                        "recipes_json": result.recipes_json,
                        "citations": result.citations,
                        "memory_ids": result.memory_ids,
                    },
                )
                store.update_run_status(r.run_id, "completed")
            except CancelledError:
                cancel.request_cancel()
                store.append_event(r.run_id, "run_canceled", {"reason": "cancel_requested"})
                store.update_run_status(r.run_id, "canceled")
                any_canceled = True
            except KeyboardInterrupt:
                cancel.request_cancel()
                store.append_event(r.run_id, "run_canceled", {"reason": "KeyboardInterrupt"})
                store.update_run_status(r.run_id, "canceled")
                any_canceled = True
            except Exception as e:
                any_failed = True
                store.append_event(
                    r.run_id,
                    "run_failed",
                    {"error": str(e), "traceback": traceback.format_exc()},
                )
                store.update_run_status(r.run_id, "failed", error=str(e))

        if any_failed:
            store.update_batch_status(batch.batch_id, "failed")
        elif any_canceled:
            store.update_batch_status(batch.batch_id, "canceled")
        else:
            store.update_batch_status(batch.batch_id, "completed")

        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
