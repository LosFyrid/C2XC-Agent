from __future__ import annotations

import json

from src.tools.citation_aliases import extract_citation_aliases, extract_memory_ids


def test_extract_citation_aliases_supports_multi_alias_brackets() -> None:
    text = "Evidence suggests X [C1, C12]; also see [C3]."
    assert extract_citation_aliases(text) == ["C1", "C12", "C3"]


def test_extract_citation_aliases_does_not_confuse_json_arrays_for_citations() -> None:
    payload = {
        "recipes": [
            {
                "M1": "Cu",
                "M2": "Ag",
                "rationale": "Use evidence [C7, C10] and prior mem:72f5543c.",
            }
        ]
    }
    dumped = json.dumps(payload, ensure_ascii=False)
    # Should extract only the citation aliases inside the rationale, not JSON keys like M1/M2.
    assert extract_citation_aliases(dumped) == ["C7", "C10"]


def test_extract_memory_ids_supports_full_uuid_and_prefix() -> None:
    full = "72f5543c-fda9-44d6-8a46-755c60d9af19"
    text = f"Use mem:{full} then mem:72f5543c."
    assert extract_memory_ids(text) == [full, "72f5543c"]


def test_extract_memory_ids_prefers_full_uuid_over_prefix() -> None:
    # Ensure we don't accidentally match the first 8 hex chars of a full UUID.
    full = "72f5543c-fda9-44d6-8a46-755c60d9af19"
    assert extract_memory_ids(f"mem:{full}") == [full]

