from __future__ import annotations

import re
from dataclasses import dataclass

from .lightrag_kb import KBChunk


# Alias tokens look like:
# - [C1]
# - [C12]
# - [C5, C16] (comma/semicolon separated lists are common in LLM outputs)
# Prefix is one-or-more uppercase letters; suffix is one-or-more digits.
# Only match bracket groups that *look like* citations, i.e. after optional whitespace the
# first char is NOT a JSON container starter (`{` or `"`). This avoids accidentally treating
# JSON arrays like `["recipes": [...]]` as citation brackets and extracting tokens like `M1`/`M2`.
_BRACKET_RE = re.compile(r"\[\s*(?P<body>(?![{\"])[^\]]+)\]")
_ALIAS_IN_BRACKET_RE = re.compile(r"(?<![A-Za-z0-9])(?P<alias>[A-Z]+\d+)(?![A-Za-z0-9])")

# Memory tokens:
# - mem:<uuid> (canonical)
# - mem:<hex_prefix> (allowed; must be resolved against the run memory registry)
#
# NOTE: We must avoid matching the first 8 hex chars of a full UUID (because a UUID has a '-'
# right after the first 8 chars). The `(?!-)` guard prevents that.
_MEM_TOKEN_RE = re.compile(
    r"\bmem:(?P<mem_id>"
    r"(?:"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|"
    r"[0-9a-fA-F]{8,32}"
    r")"
    r")(?!-)\b"
)


@dataclass(frozen=True)
class AliasedKBChunk:
    """KB chunk with a short alias for LLM-friendly citation.

    Example:
        alias="C1"
        ref="kb:kb_modulation__chunk-<md5>"
    """

    alias: str
    ref: str
    source: str
    content: str
    kb_namespace: str
    lightrag_chunk_id: str | None


def alias_kb_chunks(
    chunks: list[KBChunk], *, prefix: str = "C"
) -> tuple[list[AliasedKBChunk], dict[str, str]]:
    """Assign stable-in-order aliases [C1], [C2]... for a batch of retrieved chunks.

    Returns:
      - aliased list (same order as input)
      - alias_map: alias -> canonical ref (kb:...)
    """
    aliased: list[AliasedKBChunk] = []
    alias_map: dict[str, str] = {}

    for idx, ch in enumerate(chunks, start=1):
        alias = f"{prefix}{idx}"
        aliased_chunk = AliasedKBChunk(
            alias=alias,
            ref=ch.ref,
            source=ch.source,
            content=ch.content,
            kb_namespace=ch.kb_namespace,
            lightrag_chunk_id=ch.lightrag_chunk_id,
        )
        aliased.append(aliased_chunk)
        alias_map[alias] = ch.ref

    return aliased, alias_map


def extract_citation_aliases(text: str) -> list[str]:
    """Extract KB citation aliases from LLM output.

    Supports both:
    - single-alias bracket tokens: [C1]
    - multi-alias bracket tokens: [C5, C16]

    Returns aliases in first-seen order, de-duplicated.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _BRACKET_RE.finditer(text or ""):
        body = (m.group("body") or "").strip()
        if not body:
            continue
        for m2 in _ALIAS_IN_BRACKET_RE.finditer(body):
            alias = m2.group("alias")
            if alias in seen:
                continue
            seen.add(alias)
            ordered.append(alias)
    return ordered


def extract_memory_ids(text: str) -> list[str]:
    """Extract memory citations like `mem:<uuid>` (or `mem:<hex_prefix>`) from text.

    Returns tokens in first-seen order, de-duplicated.
    - Full UUID tokens are returned as full UUID strings.
    - Prefix tokens are returned as the prefix string (hex, without hyphens).

    These tokens must be resolved against the run memory registry before being treated as
    authoritative mem_ids.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _MEM_TOKEN_RE.finditer(text or ""):
        mem_id = m.group("mem_id")
        if mem_id in seen:
            continue
        seen.add(mem_id)
        ordered.append(mem_id)
    return ordered


def resolve_aliases(
    aliases: list[str], alias_map: dict[str, str]
) -> dict[str, str]:
    """Resolve a list of aliases into canonical refs.

    Raises KeyError if any alias is unknown (program-level validation).
    """
    return {a: alias_map[a] for a in aliases}
