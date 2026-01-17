import json

import pytest

from src.reasoningbank.rbmem_claims_v1 import (
    RBMEM_CLAIMS_V1_HEADER,
    RBMemClaimsV1ValidationError,
    canonicalize_rbmem_claims_v1,
    parse_rbmem_claims_v1,
    validate_rbmem_claims_v1,
)


def _mk_content(*, claims: list[dict], topic: str = "t", scope: str = "global") -> str:
    return "\n".join(
        [
            RBMEM_CLAIMS_V1_HEADER,
            f"TOPIC={topic}",
            f"SCOPE={scope}",
            f"CLAIMS_JSON={json.dumps(claims, ensure_ascii=False)}",
            "",
        ]
    )


def test_parse_minimal_rbmem_claims_v1() -> None:
    content = _mk_content(
        claims=[
            {
                "claim_id": "c1",
                "status": "hypothesis",
                "facts": {"source_run_ids": ["run_1"]},
                "inference": {"summary": "hypothesis summary"},
                "constraint": {"avoid": ["foo"], "allow_positive": False},
                "conditions": [],
                "limitations": [],
                "support": {"count": 0, "run_ids": []},
                "contra": {"count": 0, "run_ids": []},
            }
        ]
    )
    doc = parse_rbmem_claims_v1(content)
    assert doc.topic == "t"
    assert doc.scope == "global"
    assert len(doc.claims) == 1
    assert doc.claims[0].claim_id == "c1"
    assert doc.claims[0].status == "hypothesis"


def test_validate_rejects_kb_alias_tokens() -> None:
    content = _mk_content(
        claims=[
            {
                "claim_id": "c1",
                "status": "fact",
                "facts": {"source_run_ids": ["run_1"]},
                "inference": {"summary": "supported by [C12]"},
                "constraint": {"avoid": ["foo"], "allow_positive": False},
                "conditions": [],
                "limitations": [],
                "support": {"count": 1, "run_ids": ["run_1"]},
                "contra": {"count": 0, "run_ids": []},
            }
        ]
    )
    with pytest.raises(RBMemClaimsV1ValidationError) as e:
        validate_rbmem_claims_v1(content)
    assert any("KB aliases" in issue for issue in e.value.issues)


def test_validate_claim_count_limit() -> None:
    claims: list[dict] = []
    for i in range(11):
        claims.append(
            {
                "claim_id": f"c{i}",
                "status": "hypothesis",
                "facts": {"source_run_ids": ["run_1"]},
                "inference": {"summary": "x"},
                "constraint": {"avoid": ["foo"], "allow_positive": False},
                "conditions": [],
                "limitations": [],
                "support": {"count": 0, "run_ids": []},
                "contra": {"count": 0, "run_ids": []},
            }
        )
    content = _mk_content(claims=claims)
    with pytest.raises(RBMemClaimsV1ValidationError) as e:
        validate_rbmem_claims_v1(content, max_claims=10)
    assert any("too many claims" in issue for issue in e.value.issues)


def test_validate_positive_constraint_requires_exception() -> None:
    content = _mk_content(
        claims=[
            {
                "claim_id": "c1",
                "status": "hypothesis",
                "facts": {"source_run_ids": ["run_1"]},
                "inference": {"summary": "x"},
                "constraint": {"must": ["bar"]},
                "conditions": [],
                "limitations": [],
                "support": {"count": 0, "run_ids": []},
                "contra": {"count": 0, "run_ids": []},
            }
        ]
    )
    with pytest.raises(RBMemClaimsV1ValidationError) as e:
        validate_rbmem_claims_v1(content)
    assert any("positive constraints" in issue for issue in e.value.issues)


def test_validate_banned_constraint_keys() -> None:
    content = _mk_content(
        claims=[
            {
                "claim_id": "c1",
                "status": "hypothesis",
                "facts": {"source_run_ids": ["run_1"]},
                "inference": {"summary": "x"},
                "constraint": {"avoid": ["foo"], "next_experiment": "try X", "allow_positive": False},
                "conditions": [],
                "limitations": [],
                "support": {"count": 0, "run_ids": []},
                "contra": {"count": 0, "run_ids": []},
            }
        ]
    )
    with pytest.raises(RBMemClaimsV1ValidationError) as e:
        validate_rbmem_claims_v1(content)
    assert any("next-step" in issue for issue in e.value.issues)


def test_canonicalize_roundtrip() -> None:
    content = _mk_content(
        topic="Topic With Spaces",
        claims=[
            {
                "claim_id": "c1",
                "status": "hypothesis",
                "facts": {"source_run_ids": ["run_1"]},
                "inference": {"summary": "x"},
                "constraint": {"avoid": ["foo"], "allow_positive": False},
                "conditions": ["cond"],
                "limitations": [],
                "support": {"count": 0, "run_ids": []},
                "contra": {"count": 0, "run_ids": []},
            }
        ],
    )
    canon = canonicalize_rbmem_claims_v1(content)
    assert canon.endswith("\n")
    doc = parse_rbmem_claims_v1(canon)
    assert doc.topic == "Topic With Spaces"
    assert doc.claims[0].claim_id == "c1"

