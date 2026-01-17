# Changelog

This project currently has no tagged releases. Version labels below follow commit messages and milestone naming.

## v0.2.2 (2026-01-17)

- Fixed: `/api/v1/runs/{run_id}/modifier_checks` no longer crashes due to `sqlite3.Row` lacking `.get()` (PubChem panel 500).
- Improved: PubChem SMILES parsing now accepts multiple keys (`CanonicalSMILES`/`IsomericSMILES`/`SMILES`/`ConnectivitySMILES`) to avoid missing `canonical_smiles`/`has_cooh`.
- New: Cache-only API endpoint `GET /api/v1/runs/{run_id}/modifier_checks` (does not trigger PubChem; returns 404 if not computed yet).
- Changed: WebUI no longer auto-triggers PubChem requests on run page load; it reads cached modifier checks and provides a "Resolve" button for on-demand compute.
- Improved: Output recipe cards surface `implementation_conditions` and `evidence_summary` for faster review.

## v0.2.1 (2026-01-17)

- Improved: WebUI long-text readability and traceability
  - Trace/JSON viewer: long string fields are expandable (with a rendered view + copy).
  - Evidence viewer: evidence chunks render Markdown + safe HTML (e.g. `<sub>/<sup>`) with a Raw fallback view.
  - JSON Tree: object/array nodes support "Show more"/"Show all" to reveal previously hidden keys/items.
  - Output: recipe rationales and `overall_notes` render Markdown while preserving clickable citation aliases.

## v0.2 (2026-01-17)

- New: Strict RB memory format `RBMEM_CLAIMS_V1` (key=value header + `CLAIMS_JSON`) stored in `MemoryItem.content`.
- New: Claim-level epistemic status (`fact|hypothesis|conclusion`) and support/contra counters (item lifecycle remains `active|archived`).
- New: "Search claims, return items" retrieval (claim docs are derived/index-only; the item remains the truth/citation unit).
- New: RB tooling scripts
  - `scripts/rb/rebuild_claim_index.py` (rebuild derived claim docs from existing items)
  - `scripts/rb/migrate_legacy_rbmem_claims_v1.py` (migration helpers; strategy B is supported by wipe/rewrite)
- New: PubChem modifier checks for WebUI review
  - API: `POST /api/v1/runs/{run_id}/modifier_checks`
  - UI: show CID / CanonicalSMILES / best-effort `has_cooh` signal (non-blocking; for manual inspection)

- Changed: Chroma RB collection now stores `doc_type=item|claim` metadata; claim docs are generated per item (max 10 per item).
- Changed: `mem_search` / `GET /api/v1/memories?query=...` can surface `matched_claims` hints (debug/UI optional).
- Changed: Thinking-mode compatibility
  - When `C2XC_LLM_ENABLE_THINKING=1`, the system no longer sends vendor JSON schema response_format; it relies on prompt + local JSON extraction/validation.
- Improved: More actionable error when RB Chroma collection cannot be opened due to embedding/collection config mismatch (suggests collection rename or wiping the persist dir).
- Improved: RB learn now injects a system-built `facts_digest_json` (trace/output/feedback) to reduce hallucinated "facts" in learned memories.
- Improved: RB learn trace visibility (additional RB-specific events for verdict application / rejection / indexing).

## v0.1.1 (2026-01-15)

- Changed: Increased safety bound `recap.max_steps` from 60 to 100 (fewer premature "Exceeded recap.max_steps" failures).
- Improved: Citation parsing robustness
  - Supports multi-alias brackets like `[C5, C16]`.
  - Avoids confusing JSON arrays/keys for citation brackets.
  - Supports `mem:<uuid>` and `mem:<hex_prefix>` citations (prefix tokens are resolved against the run memory registry).
- Improved: Final output validation now checks per-recipe citation presence and resolves cited memory ids.

## v0.1 (2025-12-30)

- Initial implementation (ReCAP engine + KB integration + ReasoningBank store + WebUI).
