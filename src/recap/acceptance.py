from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def _now_ts() -> float:
    return time.time()


_ALLOWED_IMPACTS = {"critical", "supporting", "minor", "negligible", "na"}


@dataclass(frozen=True)
class AcceptanceOutcome:
    accepted: bool
    acceptance_record: dict[str, Any]
    report: dict[str, Any] | None
    repair_message: str | None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _validate_items(
    *,
    items: Any,
    required_ids: list[int],
    item_label: str,
) -> tuple[list[int], list[str]]:
    """Validate a list of mechanism/role dicts.

    Returns:
      - missing_ids
      - errors (human-readable strings)
    """
    errors: list[str] = []
    if not isinstance(items, list):
        return required_ids[:], [f"{item_label} must be an array"]

    required_set = set(required_ids)
    seen: set[int] = set()
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"{item_label}[{idx}] must be an object")
            continue
        item_id = _as_int(it.get("id"))
        if item_id is None:
            errors.append(f"{item_label}[{idx}].id must be an integer")
            continue
        if item_id not in required_set:
            errors.append(f"{item_label}[{idx}].id={item_id} is not in required set {sorted(required_set)}")
            continue
        if item_id in seen:
            errors.append(f"{item_label}[{idx}].id={item_id} is duplicated")
            continue
        seen.add(item_id)

        impact = str(it.get("impact") or "").strip()
        if impact not in _ALLOWED_IMPACTS:
            errors.append(
                f"{item_label}[{idx}].impact must be one of {sorted(_ALLOWED_IMPACTS)}, got {impact!r}"
            )

        justification = str(it.get("justification") or "").strip()
        if not justification:
            errors.append(f"{item_label}[{idx}].justification is required (non-empty)")

    missing = [i for i in required_ids if i not in seen]
    return missing, errors


def _build_repair_message(
    *,
    role: str,
    expected_schema: str,
    item_label: str,
    missing_ids: list[int],
    errors: list[str],
    max_repairs: int,
    attempt_idx: int,
) -> str:
    # Keep this compact but explicit; the goal is to make the repair deterministic.
    lines: list[str] = []
    lines.append("ERROR: Deliverable failed strict acceptance.")
    lines.append(f"role={role} expected_schema={expected_schema}")
    lines.append(f"repair_attempt={attempt_idx}/{max_repairs}")
    lines.append("")
    if missing_ids:
        lines.append(f"Missing required {item_label} ids: {missing_ids}")
    if errors:
        lines.append("Validation errors:")
        for e in errors[:20]:
            lines.append(f"- {e}")
        if len(errors) > 20:
            lines.append(f"(+{len(errors) - 20} more)")
    lines.append("")
    lines.append("Repair instructions:")
    lines.append("- Return subtasks=[] and a `result` that is a SINGLE JSON object (no extra text).")
    lines.append(f"- Top-level `schema` MUST be {expected_schema!r}.")
    lines.append(f"- Provide {item_label} as an array. Include ALL required ids, even if impact is negligible/na.")
    lines.append("- For each item: include `id`, `impact`, and a non-empty `justification`.")
    lines.append("- `impact` may be 'negligible' or 'na' ONLY if you justify why it is negligible/not applicable.")
    lines.append("- You may reuse prior content, but ensure coverage is complete and consistent.")
    return "\n".join(lines).strip()


def validate_expert_deliverable(
    *,
    role: str,
    report_obj: dict[str, Any],
    max_repairs: int,
    attempt_idx: int,
) -> AcceptanceOutcome:
    """Validate an expert's `result` JSON against a strict coverage contract."""
    r = (role or "").strip()
    if r not in {"tio2_expert", "mof_expert"}:
        rec = {
            "schema": "acceptance_record_v1",
            "target_role": r,
            "target_schema": None,
            "accepted": True,
            "checked_at": _now_ts(),
            "notes": "role has no strict acceptance contract",
        }
        return AcceptanceOutcome(accepted=True, acceptance_record=rec, report=report_obj, repair_message=None)

    if r == "tio2_expert":
        expected_schema = "tio2_mechanisms_report_v1"
        item_label = "mechanisms"
        required_ids = list(range(1, 8))
        items = report_obj.get("mechanisms")
    else:
        expected_schema = "mof_roles_report_v1"
        item_label = "roles"
        required_ids = list(range(1, 11))
        items = report_obj.get("roles")

    errors: list[str] = []
    parse_error = report_obj.get("_parse_error")
    if isinstance(parse_error, str) and parse_error.strip():
        errors.append(f"result must be a valid JSON object: {parse_error.strip()}")
    schema_name = str(report_obj.get("schema") or "").strip()
    if schema_name != expected_schema:
        errors.append(f"schema must be {expected_schema!r}, got {schema_name!r}")

    missing_ids, item_errors = _validate_items(items=items, required_ids=required_ids, item_label=item_label)
    errors.extend(item_errors)

    accepted = (not missing_ids) and (not errors)
    rec = {
        "schema": "acceptance_record_v1",
        "target_role": r,
        "target_schema": expected_schema,
        "accepted": bool(accepted),
        "missing_ids": missing_ids,
        "errors": errors,
        "checked_at": _now_ts(),
    }
    if accepted:
        return AcceptanceOutcome(accepted=True, acceptance_record=rec, report=report_obj, repair_message=None)

    msg = _build_repair_message(
        role=r,
        expected_schema=expected_schema,
        item_label=item_label,
        missing_ids=missing_ids,
        errors=errors,
        max_repairs=max_repairs,
        attempt_idx=attempt_idx,
    )
    return AcceptanceOutcome(accepted=False, acceptance_record=rec, report=None, repair_message=msg)
