from __future__ import annotations

from src.recap.acceptance import validate_expert_deliverable


def test_tio2_acceptance_requires_all_7_ids() -> None:
    report = {
        "schema": "tio2_mechanisms_report_v1",
        "mechanisms": [
            {"id": 1, "impact": "critical", "justification": "x"},
            {"id": 2, "impact": "supporting", "justification": "x"},
            # Missing 3..7
        ],
        "synthesis": "x",
    }
    out = validate_expert_deliverable(role="tio2_expert", report_obj=report, max_repairs=3, attempt_idx=1)
    assert out.accepted is False
    assert out.acceptance_record["target_schema"] == "tio2_mechanisms_report_v1"
    assert out.acceptance_record["missing_ids"] == [3, 4, 5, 6, 7]
    assert out.repair_message and "Missing required mechanisms ids" in out.repair_message


def test_mof_acceptance_allows_negligible_and_na_with_justification() -> None:
    report = {
        "schema": "mof_roles_report_v1",
        "roles": [
            {"id": i, "impact": ("negligible" if i % 2 == 0 else "na"), "justification": f"ok {i}"}
            for i in range(1, 11)
        ],
        "synthesis": "x",
    }
    out = validate_expert_deliverable(role="mof_expert", report_obj=report, max_repairs=3, attempt_idx=1)
    assert out.accepted is True
    assert out.repair_message is None


def test_acceptance_flags_invalid_impact() -> None:
    report = {
        "schema": "tio2_mechanisms_report_v1",
        "mechanisms": [
            {"id": i, "impact": "unsupported", "justification": "x"} for i in range(1, 8)
        ],
    }
    out = validate_expert_deliverable(role="tio2_expert", report_obj=report, max_repairs=3, attempt_idx=1)
    assert out.accepted is False
    assert any("impact must be one of" in e for e in out.acceptance_record.get("errors", []))


def test_acceptance_includes_parse_error() -> None:
    report = {"schema": "", "_parse_error": "Invalid JSON"}
    out = validate_expert_deliverable(role="tio2_expert", report_obj=report, max_repairs=3, attempt_idx=1)
    assert out.accepted is False
    assert any("result must be a valid JSON object" in e for e in out.acceptance_record.get("errors", []))

