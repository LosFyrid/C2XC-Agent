from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from src.config.load_config import load_app_config
from src.llm.openai_compat import OpenAICompatibleChatClient
from src.reasoningbank.rbmem_claims_v1 import (
    RBMEM_CLAIMS_V1_HEADER,
    RBMemClaimsV1ValidationError,
    canonicalize_rbmem_claims_v1,
    is_rbmem_claims_v1,
    validate_rbmem_claims_v1,
)
from src.storage.reasoningbank_store import MemoryItem, ReasoningBankStore
from src.storage.sqlite_store import SQLiteStore


_KB_ALIAS_RE = re.compile(r"\[C\d+(?:\s*,\s*C\d+)*\]")


def _strip_kb_aliases(text: str) -> str:
    return _KB_ALIAS_RE.sub("", text or "").strip()


def _wrap_legacy_item(item: MemoryItem) -> str:
    legacy = _strip_kb_aliases(item.content or "")
    # Keep this deterministic and safe: one hypothesis claim that preserves the legacy text.
    claims = [
        {
            "claim_id": "c1",
            "status": "hypothesis",
            "facts": {
                "source_run_ids": [item.source_run_id] if item.source_run_id else [],
                "migrated_from": "legacy_text",
            },
            "inference": {"summary": legacy[:800]},
            "constraint": {"avoid": [], "allow_positive": False},
            "conditions": [],
            "limitations": ["migrated_from_legacy_text; needs review"],
            "support": {"count": 0, "run_ids": []},
            "contra": {"count": 0, "run_ids": []},
        }
    ]
    out = "\n".join(
        [
            RBMEM_CLAIMS_V1_HEADER,
            f"TOPIC=Legacy migration (wrapped) mem:{item.mem_id}",
            f"SCOPE={item.role}",
            "CLAIMS_JSON=" + json.dumps(claims, ensure_ascii=False, separators=(",", ":")),
            "",
        ]
    )
    return out


@dataclass(frozen=True)
class _RewriteResult:
    content: str
    attempts: int


def _rewrite_with_llm(
    *,
    llm: OpenAICompatibleChatClient,
    system: str,
    legacy_text: str,
    role: str,
    source_run_id: str | None,
    max_attempts: int = 2,
) -> _RewriteResult:
    cleaned = _strip_kb_aliases(legacy_text)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("Legacy content is empty after cleaning.")

    base_prompt = f"""
You are migrating a legacy ReasoningBank memory into a strict, machine-parseable format.

Requirements (hard):
- Output MUST be a key=value block with this exact first line:
  {RBMEM_CLAIMS_V1_HEADER}
- Include: TOPIC=..., SCOPE=..., CLAIMS_JSON=...
- CLAIMS_JSON MUST be a JSON array on a SINGLE line (minified JSON is OK).
- Max 10 claims.
- Each claim MUST contain at least: claim_id, status, inference, constraint, facts, conditions, limitations, support, contra.
- claim.status must be one of: fact | hypothesis | conclusion
- Do NOT include run-local KB aliases like [C12] or [C5, C16].
- Do NOT include next-step experimental instructions in constraint (no 'next_experiment', no 'try X next run', etc).
- Constraints default to negative constraints (avoid); positive constraints (must/prefer) are allowed ONLY as explicit exceptions
  with allow_positive=true and exception_reason.

Role: {role}
source_run_id: {source_run_id or ""}

Legacy memory text (rewrite into 1-5 high-signal claims; preserve meaning, but make it structured):
{cleaned}

Output ONLY the RBMEM_CLAIMS_V1 block. No code fences, no commentary.
""".strip()

last_err: str | None = None
attempts = 0
prompt = base_prompt
while attempts < int(max_attempts):
    attempts += 1
    raw = llm.chat(system=system, user=prompt, temperature=0.0, extra={})
    candidate = (raw.content or "").strip()
    # Sometimes models add leading/trailing text; keep the block from the header onwards.
    idx = candidate.find(RBMEM_CLAIMS_V1_HEADER)
    if idx != -1:
        candidate = candidate[idx:].strip()

    try:
        validate_rbmem_claims_v1(candidate, max_claims=10, forbid_kb_alias=True)
        canon = canonicalize_rbmem_claims_v1(candidate)
        return _RewriteResult(content=canon, attempts=attempts)
    except RBMemClaimsV1ValidationError as e:
        last_err = "; ".join(e.issues[:8]) if e.issues else str(e)
        prompt = (
            base_prompt
            + "\n\n"
            + "Your previous output was invalid. Fix FORMAT ONLY; keep the same meaning.\n"
            + f"Validation issues: {last_err}\n"
            + "Return ONLY the RBMEM_CLAIMS_V1 block.\n"
        )

raise ValueError(f"Failed to rewrite legacy memory after {attempts} attempts: {last_err or 'unknown error'}")


def _resolve_sqlite_path(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.getenv("C2XC_SQLITE_PATH", "").strip()
    return env or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy ReasoningBank items to RBMEM_CLAIMS_V1.",
    )
    parser.add_argument(
        "--mode",
        choices=["wrap", "rewrite"],
        default="rewrite",
        help="wrap: deterministic 1-claim wrapper; rewrite: use LLM to rewrite into structured claims.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of items to process (0 = no limit).",
    )
    parser.add_argument(
        "--sqlite-path",
        default=None,
        help="Optional SQLite path for updating rb_mem_index (defaults to env C2XC_SQLITE_PATH).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes; just print what would change.",
    )
    args = parser.parse_args()

    cfg = load_app_config()
    rb = ReasoningBankStore.from_config(cfg)

    # Optional: update rb_mem_index for better UI consistency (created_at sorting unaffected).
    sqlite_path = _resolve_sqlite_path(args.sqlite_path)
    store: SQLiteStore | None = SQLiteStore(sqlite_path) if sqlite_path else None

    llm: OpenAICompatibleChatClient | None = None
    if args.mode == "rewrite":
        llm = OpenAICompatibleChatClient()
        # Keep system prompt aligned with runtime environment.
        system = "\n\n".join(
            [
                cfg.prompts.system_base.strip(),
                cfg.priors.system_description_md.strip(),
            ]
        ).strip()
    else:
        system = ""

    try:
        items = rb.list_all(include_content=True)
        processed = 0
        updated = 0
        for it in items:
            if it.type != "reasoningbank_item":
                continue
            if is_rbmem_claims_v1(it.content):
                continue
            processed += 1
            if args.limit and processed > int(args.limit):
                break

            if args.mode == "wrap":
                new_content = _wrap_legacy_item(it)
            else:
                assert llm is not None
                res = _rewrite_with_llm(
                    llm=llm,
                    system=system,
                    legacy_text=it.content,
                    role=it.role,
                    source_run_id=it.source_run_id,
                    max_attempts=2,
                )
                new_content = res.content

            if args.dry_run:
                print(f"[DRY RUN] Would migrate mem:{it.mem_id} (role={it.role}, status={it.status})")
                continue

            # Overwrite in-place to preserve mem_id and created_at.
            after = rb.upsert(
                mem_id=it.mem_id,
                status=it.status,
                role=it.role,
                type=it.type,
                content=new_content,
                source_run_id=it.source_run_id,
                schema_version=max(int(it.schema_version), 2),
                extra={**(it.extra or {}), "migrated_to": "RBMEM_CLAIMS_V1"},
                preserve_created_at=True,
            )
            updated += 1
            if store is not None:
                store.upsert_rb_mem_index(
                    mem_id=after.mem_id,
                    created_at=float(after.created_at),
                    updated_at=float(after.updated_at),
                    status=str(after.status),
                    role=str(after.role),
                    type=str(after.type),
                    source_run_id=str(after.source_run_id or "") or None,
                    schema_version=int(after.schema_version),
                )

        # Ensure claim docs exist for all items (including those we didn't touch).
        if not args.dry_run:
            rb.rebuild_claim_index(include_archived=True)

        print(
            f"Processed legacy items: {processed}. Updated: {updated}. Mode={args.mode} DryRun={bool(args.dry_run)}"
        )
        return 0
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())

