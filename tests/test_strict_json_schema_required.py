from __future__ import annotations

from typing import Any

from src.recap.engine import _recipes_response_format
from src.runtime.reasoningbank_learn import _extractor_response_format, _merge_response_format


def _assert_openai_strict_required(schema: Any) -> None:
    """Assert the OpenAI strict JSON Schema rule for Structured Outputs.

    When `strict=true`, OpenAI requires that for every object schema with `properties`,
    `required` must be present and must include *every* key in `properties`.
    """
    if isinstance(schema, list):
        for item in schema:
            _assert_openai_strict_required(item)
        return

    if not isinstance(schema, dict):
        return

    props = schema.get("properties")
    if schema.get("type") == "object" and isinstance(props, dict):
        required = schema.get("required")
        assert isinstance(required, list), f"Missing/invalid required for object schema: {schema!r}"
        required_set = {str(x) for x in required}
        props_set = {str(k) for k in props.keys()}
        assert required_set == props_set, f"required must match properties keys. required={required_set}, props={props_set}"

    # Recurse into common schema containers.
    for k in ("properties", "items", "oneOf", "anyOf", "allOf"):
        v = schema.get(k)
        if isinstance(v, dict):
            _assert_openai_strict_required(v)
        elif isinstance(v, list):
            for item in v:
                _assert_openai_strict_required(item)


def test_generate_recipes_schema_is_openai_strict_required() -> None:
    rf = _recipes_response_format(recipes_per_run=1)
    schema = (rf.get("json_schema") or {}).get("schema")
    _assert_openai_strict_required(schema)


def test_rb_extract_items_schema_is_openai_strict_required() -> None:
    rf = _extractor_response_format()
    schema = (rf.get("json_schema") or {}).get("schema")
    _assert_openai_strict_required(schema)


def test_rb_merge_result_schema_is_openai_strict_required() -> None:
    rf = _merge_response_format()
    schema = (rf.get("json_schema") or {}).get("schema")
    _assert_openai_strict_required(schema)

