from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


RBMEM_CLAIMS_V1_HEADER = "RBMEM_CLAIMS_V1"

ClaimStatus = Literal["fact", "hypothesis", "conclusion"]


class RBMemClaimsV1Error(ValueError):
    pass


class RBMemClaimsV1ParseError(RBMemClaimsV1Error):
    pass


class RBMemClaimsV1ValidationError(RBMemClaimsV1Error):
    def __init__(self, message: str, *, issues: list[str] | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []


_KB_ALIAS_RE = re.compile(r"\[C\d+(?:\s*,\s*C\d+)*\]")


def contains_kb_alias(text: str) -> bool:
    """Detect run-local KB citation aliases like [C12] or [C5, C16]."""
    return bool(_KB_ALIAS_RE.search(text or ""))


def is_rbmem_claims_v1(content: str) -> bool:
    first = ""
    for line in (content or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    return first == RBMEM_CLAIMS_V1_HEADER


def _is_kv_key_line(line: str) -> bool:
    # Allow keys like TOPIC, SCOPE, CLAIMS_JSON.
    return bool(re.match(r"^[A-Z][A-Z0-9_]*=", line or ""))


def _parse_kv_block(content: str) -> dict[str, str]:
    lines = (content or "").splitlines()
    # Drop leading empty lines.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise RBMemClaimsV1ParseError("Empty content.")

    header = lines[idx].strip()
    if header != RBMEM_CLAIMS_V1_HEADER:
        raise RBMemClaimsV1ParseError(
            f"Invalid RBMEM header: {header!r} (expected {RBMEM_CLAIMS_V1_HEADER!r})."
        )

    idx += 1
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_val_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_val_lines
        if current_key is None:
            return
        fields[current_key] = "\n".join(current_val_lines).strip()
        current_key = None
        current_val_lines = []

    while idx < len(lines):
        raw = lines[idx]
        idx += 1
        if not raw.strip():
            # Keep blank lines inside a multi-line JSON block if any; they will be stripped on join.
            if current_key is not None:
                current_val_lines.append(raw)
            continue

        if _is_kv_key_line(raw):
            flush()
            key, val = raw.split("=", 1)
            current_key = key.strip()
            current_val_lines = [val]
            continue

        if current_key is None:
            raise RBMemClaimsV1ParseError(
                f"Unexpected non key=value line outside a field: {raw!r}"
            )
        current_val_lines.append(raw)

    flush()
    return fields


def _json_loads(value: str, *, what: str) -> Any:
    try:
        return json.loads(value)
    except Exception as e:
        raise RBMemClaimsV1ParseError(f"Invalid JSON for {what}: {e}") from e


@dataclass(frozen=True)
class RBMemClaimV1:
    claim_id: str
    status: ClaimStatus
    facts: dict[str, Any]
    inference: Any
    constraint: dict[str, Any]
    conditions: list[str]
    limitations: list[str]
    support: dict[str, Any]
    contra: dict[str, Any]


@dataclass(frozen=True)
class RBMemClaimsDocV1:
    topic: str | None
    scope: str | None
    claims: list[RBMemClaimV1]
    raw_fields: dict[str, str]


def parse_rbmem_claims_v1(content: str) -> RBMemClaimsDocV1:
    fields = _parse_kv_block(content)
    topic = (fields.get("TOPIC") or "").strip() or None
    scope = (fields.get("SCOPE") or "").strip() or None

    raw_claims = (fields.get("CLAIMS_JSON") or "").strip()
    if not raw_claims:
        raise RBMemClaimsV1ParseError("Missing required field: CLAIMS_JSON.")

    claims_any = _json_loads(raw_claims, what="CLAIMS_JSON")
    if not isinstance(claims_any, list):
        raise RBMemClaimsV1ParseError("CLAIMS_JSON must be a JSON array.")

    claims: list[RBMemClaimV1] = []
    for i, c in enumerate(claims_any):
        if not isinstance(c, dict):
            raise RBMemClaimsV1ParseError(f"CLAIMS_JSON[{i}] must be an object.")
        claim_id = str(c.get("claim_id") or "").strip()
        if not claim_id:
            raise RBMemClaimsV1ParseError(f"CLAIMS_JSON[{i}] missing claim_id.")
        status = str(c.get("status") or "").strip()
        if status not in {"fact", "hypothesis", "conclusion"}:
            raise RBMemClaimsV1ParseError(
                f"CLAIMS_JSON[{i}] invalid status: {status!r} (expected fact|hypothesis|conclusion)."
            )
        facts = c.get("facts")
        if not isinstance(facts, dict):
            facts = {}
        inference = c.get("inference")
        constraint = c.get("constraint")
        if not isinstance(constraint, dict):
            constraint = {}
        conditions = c.get("conditions")
        if not isinstance(conditions, list) or not all(isinstance(x, str) for x in conditions):
            conditions = []
        limitations = c.get("limitations")
        if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
            limitations = []
        support = c.get("support")
        if not isinstance(support, dict):
            support = {}
        contra = c.get("contra")
        if not isinstance(contra, dict):
            contra = {}

        claims.append(
            RBMemClaimV1(
                claim_id=claim_id,
                status=status,  # type: ignore[arg-type]
                facts=facts,
                inference=inference,
                constraint=constraint,
                conditions=conditions,
                limitations=limitations,
                support=support,
                contra=contra,
            )
        )

    return RBMemClaimsDocV1(topic=topic, scope=scope, claims=claims, raw_fields=fields)


def _validate_constraint(*, constraint: dict[str, Any], issues: list[str], path: str) -> None:
    # Hard disallow "next step" fields in constraint. Keep list small to avoid over-constraint.
    banned_keys = {
        "next_experiment",
        "next_step",
        "next_steps",
        "suggest_next",
        "try",
        "test",
        "increase_to",
        "decrease_to",
        "tune",
        "optimize",
    }
    for k in list(constraint.keys()):
        if str(k) in banned_keys:
            issues.append(f"{path}.{k}: next-step/experimental instruction keys are not allowed in constraint.")

    allow_positive = bool(constraint.get("allow_positive") is True)
    has_positive = any(k in constraint for k in ("must", "prefer"))
    if has_positive and not allow_positive:
        issues.append(
            f"{path}: positive constraints require allow_positive=true + exception_reason."
        )
    if has_positive:
        reason = str(constraint.get("exception_reason") or "").strip()
        if not reason:
            issues.append(f"{path}: must/prefer present but exception_reason is empty.")

    # Optional: scan common imperative words in constraint strings (low false-positive domain).
    def scan_str(val: Any, *, where: str) -> None:
        if not isinstance(val, str):
            return
        s = val.lower()
        if any(tok in s for tok in ("next experiment", "next run", "you should", "we should test")):
            issues.append(f"{where}: looks like a next-step instruction (not allowed).")

    for k, v in constraint.items():
        if isinstance(v, str):
            scan_str(v, where=f"{path}.{k}")
        elif isinstance(v, list):
            for j, vv in enumerate(v):
                scan_str(vv, where=f"{path}.{k}[{j}]")


def validate_rbmem_claims_v1(
    content: str,
    *,
    max_claims: int = 10,
    forbid_kb_alias: bool = True,
) -> None:
    """Validate RBMEM_CLAIMS_V1 content and raise if invalid."""
    issues: list[str] = []

    if forbid_kb_alias and contains_kb_alias(content or ""):
        issues.append("content: KB aliases like [C12] are not allowed in RB memories.")

    try:
        doc = parse_rbmem_claims_v1(content)
    except RBMemClaimsV1ParseError as e:
        raise RBMemClaimsV1ValidationError(str(e), issues=issues) from e

    if len(doc.claims) > int(max_claims):
        issues.append(f"CLAIMS_JSON: too many claims ({len(doc.claims)} > {max_claims}).")

    # Claim-level validations.
    for i, claim in enumerate(doc.claims):
        # Retrieval anchor: enforce a non-empty natural-language summary.
        #
        # Why: ReasoningBank semantic retrieval is claim-centric. If inference.summary is empty,
        # the derived claim docs become embedding "shells" that are effectively unsearchable.
        summary = ""
        if isinstance(claim.inference, dict):
            summary = str(claim.inference.get("summary") or "").strip()
        if not summary:
            issues.append(
                f"CLAIMS_JSON[{i}].inference.summary: missing/empty (required; put the claim statement here)."
            )

        if contains_kb_alias(json.dumps(claim.__dict__, ensure_ascii=False)):
            issues.append(f"CLAIMS_JSON[{i}]: contains KB alias tokens like [C*].")
        _validate_constraint(
            constraint=claim.constraint,
            issues=issues,
            path=f"CLAIMS_JSON[{i}].constraint",
        )

        # Guardrail: prefer support/contra shapes, but don't over-enforce.
        # If present, ensure run_ids is a list of strings.
        for key in ("support", "contra"):
            obj = getattr(claim, key)
            run_ids = obj.get("run_ids")
            if run_ids is not None and (not isinstance(run_ids, list) or not all(isinstance(x, str) for x in run_ids)):
                issues.append(f"CLAIMS_JSON[{i}].{key}.run_ids: must be a list of strings.")

    if issues:
        raise RBMemClaimsV1ValidationError("RBMEM_CLAIMS_V1 validation failed.", issues=issues)


def canonicalize_rbmem_claims_v1(content: str) -> str:
    """Return a canonical single-line-json rendering (best-effort)."""
    doc = parse_rbmem_claims_v1(content)
    out_lines: list[str] = [RBMEM_CLAIMS_V1_HEADER]
    if doc.topic:
        out_lines.append(f"TOPIC={doc.topic}")
    if doc.scope:
        out_lines.append(f"SCOPE={doc.scope}")
    claims_json = json.dumps(
        [c.__dict__ for c in doc.claims],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    out_lines.append(f"CLAIMS_JSON={claims_json}")
    return "\n".join(out_lines).strip() + "\n"


def claim_text_projection(claim: RBMemClaimV1, *, max_chars: int = 700) -> str:
    """Project a claim into a short embedding-friendly string."""
    parts: list[str] = []
    status = claim.status
    parts.append(f"status={status}")

    # Prefer a stable text anchor if present inside inference/constraint; otherwise fall back to JSON.
    core: str | None = None
    if isinstance(claim.inference, dict):
        core = str(claim.inference.get("summary") or "").strip() or None
    if not core and isinstance(claim.constraint, dict):
        core = str(claim.constraint.get("summary") or "").strip() or None
    if not core:
        # Keep projection small: do not embed the full facts blob.
        core = json.dumps(
            {
                "inference": claim.inference,
                "constraint": claim.constraint,
                "conditions": claim.conditions,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    parts.append(core)
    if claim.conditions:
        parts.append("conditions: " + "; ".join([c.strip() for c in claim.conditions if c.strip()]))

    text = "\n".join([p for p in parts if p]).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
