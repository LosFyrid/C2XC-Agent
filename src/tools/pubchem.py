from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PUBCHEM_PUG_VIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


class PubChemError(RuntimeError):
    pass


@dataclass(frozen=True)
class PubChemResolution:
    query: str
    normalized_query: str
    status: str  # resolved|unresolved|error
    cid: int | None
    canonical_smiles: str | None
    inchikey: str | None
    has_cooh: bool | None
    error: str | None = None


def _normalize_name(name: str) -> str:
    """Best-effort normalization of the recipe-provided modifier string for PubChem name lookup."""
    s = (name or "").strip()
    if not s:
        return ""
    # Drop parenthetical annotations like "( -COOH )".
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    # Drop common suffix annotations.
    s = re.sub(r"\s*[-–—]\s*cooh\s*$", "", s, flags=re.IGNORECASE).strip()
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _http_get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise PubChemError(f"HTTP {e.code} for {url}. {body[:200]}".strip()) from e
    except urllib.error.URLError as e:
        raise PubChemError(f"Network error for {url}: {e}") from e

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise PubChemError(f"Invalid JSON from PubChem: {e}") from e
    return obj if isinstance(obj, dict) else {}


def _has_carboxylic_acid_smiles(smiles: str) -> bool:
    """Heuristic COOH detection from SMILES without RDKit.

    Goal: detect carboxylic acid (-C(=O)OH), not esters.
    This is best-effort and intentionally conservative.
    """
    s = (smiles or "").strip()
    if not s:
        return False

    # Common acid form: O=C(O)...
    if "O=C(O)" in s:
        return True

    # Alternative acid form: ...C(=O)O  (terminal O, not followed by atom)
    if re.search(r"C\(=O\)O(?![A-Za-z0-9\\[])", s):
        return True

    return False


def resolve_pubchem(name: str, *, timeout_s: float = 8.0) -> PubChemResolution:
    query = (name or "").strip()
    normalized = _normalize_name(query)
    if not normalized:
        return PubChemResolution(
            query=query,
            normalized_query=normalized,
            status="unresolved",
            cid=None,
            canonical_smiles=None,
            inchikey=None,
            has_cooh=None,
            error="empty_modifier",
        )

    try:
        encoded = urllib.parse.quote(normalized, safe="")
        url_cids = f"{_PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
        obj = _http_get_json(url_cids, timeout_s=timeout_s)
        cids = (((obj.get("IdentifierList") or {}) if isinstance(obj.get("IdentifierList"), dict) else {}).get("CID"))
        cid_list = cids if isinstance(cids, list) else []
        cid = int(cid_list[0]) if cid_list else None
        if cid is None:
            return PubChemResolution(
                query=query,
                normalized_query=normalized,
                status="unresolved",
                cid=None,
                canonical_smiles=None,
                inchikey=None,
                has_cooh=None,
                error="no_cid",
            )

        url_props = f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/CanonicalSMILES,InChIKey/JSON"
        props = _http_get_json(url_props, timeout_s=timeout_s)
        table = props.get("PropertyTable")
        properties = (table.get("Properties") if isinstance(table, dict) else None) if table is not None else None
        first = properties[0] if isinstance(properties, list) and properties else {}
        # PubChem's PUG REST has historically returned different SMILES keys depending on
        # endpoint behavior / requested property names. Be liberal in what we accept.
        # We still expose it as `canonical_smiles` in our API contract.
        smiles_any = (
            first.get("CanonicalSMILES")
            or first.get("IsomericSMILES")
            or first.get("SMILES")
            or first.get("ConnectivitySMILES")
        )
        smiles = str(smiles_any or "").strip() or None
        inchikey = str(first.get("InChIKey") or "").strip() or None
        has_cooh = _has_carboxylic_acid_smiles(smiles) if smiles else None

        return PubChemResolution(
            query=query,
            normalized_query=normalized,
            status="resolved",
            cid=cid,
            canonical_smiles=smiles,
            inchikey=inchikey,
            has_cooh=has_cooh,
        )
    except Exception as e:
        return PubChemResolution(
            query=query,
            normalized_query=normalized,
            status="error",
            cid=None,
            canonical_smiles=None,
            inchikey=None,
            has_cooh=None,
            error=str(e),
        )


# --- Agent-facing numeric evidence helpers (PubChem "facts" tool) ---


@dataclass(frozen=True)
class PubChemEvidence:
    """A best-effort PubChem evidence payload suitable for tracing/citation.

    Note: this is intentionally "dirty" (raw-ish) because many properties depend on
    measurement conditions and may have multiple, conflicting sources.
    """

    status: str  # ok|unresolved|error
    query: str
    cid: int | None
    op: str  # resolve|property_table|pug_view_toc|pug_view_section
    heading: str | None
    properties: list[str] | None
    extracted: dict[str, Any] | list[Any] | None
    raw_json: dict[str, Any] | None
    raw_truncated: bool
    error: str | None = None


def _sanitize_json(
    obj: Any,
    *,
    max_depth: int = 6,
    max_list: int = 80,
    max_str: int = 2000,
    _depth: int = 0,
) -> tuple[Any, bool]:
    """Return a JSON-serializable, size-bounded copy of obj.

    We keep this conservative: it preserves structure enough for debugging while
    preventing runaway event payload sizes in SQLite trace storage.
    """
    if _depth >= max_depth:
        return {"_truncated": True, "_reason": "max_depth"}, True

    if obj is None or isinstance(obj, (bool, int, float)):
        return obj, False

    if isinstance(obj, str):
        s = obj
        if len(s) > max_str:
            return s[:max_str] + "…", True
        return s, False

    if isinstance(obj, list):
        out: list[Any] = []
        truncated = False
        for it in obj[:max_list]:
            v, t = _sanitize_json(it, max_depth=max_depth, max_list=max_list, max_str=max_str, _depth=_depth + 1)
            out.append(v)
            truncated = truncated or t
        if len(obj) > max_list:
            out.append({"_truncated": True, "_reason": "max_list", "_skipped": len(obj) - max_list})
            truncated = True
        return out, truncated

    if isinstance(obj, dict):
        out2: dict[str, Any] = {}
        truncated = False
        # Deterministic ordering to keep traces stable-ish.
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            key = str(k)
            v, t = _sanitize_json(
                obj.get(k),
                max_depth=max_depth,
                max_list=max_list,
                max_str=max_str,
                _depth=_depth + 1,
            )
            out2[key] = v
            truncated = truncated or t
        return out2, truncated

    # Fallback for non-JSON types.
    return str(obj), True


def _collect_pubchem_strings(obj: Any, *, limit: int = 120) -> list[str]:
    """Best-effort extraction of human-readable strings from a PUG-View JSON blob."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s2 = (s or "").strip()
        if not s2:
            return
        if s2 in seen:
            return
        seen.add(s2)
        out.append(s2)

    def _walk(x: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(x, dict):
            # Common PUG-View idioms
            if isinstance(x.get("String"), str):
                _add(x["String"])
            swm = x.get("StringWithMarkup")
            if isinstance(swm, list):
                for it in swm:
                    if isinstance(it, dict) and isinstance(it.get("String"), str):
                        _add(str(it.get("String") or ""))
                        if len(out) >= limit:
                            return
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, list):
            for it in x:
                _walk(it)
                if len(out) >= limit:
                    return
            return

    _walk(obj)
    return out[:limit]


def resolve_cids(query: str, *, timeout_s: float = 8.0, max_cids: int = 8) -> list[int]:
    """Resolve a best-effort CID list from a compound name string."""
    normalized = _normalize_name(query or "")
    if not normalized:
        return []
    encoded = urllib.parse.quote(normalized, safe="")
    url = f"{_PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
    obj = _http_get_json(url, timeout_s=timeout_s)
    cids = (((obj.get("IdentifierList") or {}) if isinstance(obj.get("IdentifierList"), dict) else {}).get("CID"))
    cid_list = cids if isinstance(cids, list) else []
    out: list[int] = []
    for c in cid_list[: max(1, int(max_cids))]:
        try:
            out.append(int(c))
        except Exception:
            continue
    return out


def fetch_property_table(
    cid: int,
    *,
    properties: list[str],
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    prop_list = ",".join([p.strip() for p in (properties or []) if str(p or "").strip()])
    if not prop_list:
        prop_list = "MolecularWeight,ExactMass"
    url = f"{_PUBCHEM_BASE}/compound/cid/{int(cid)}/property/{prop_list}/JSON"
    return _http_get_json(url, timeout_s=timeout_s)


def fetch_pug_view_section(
    cid: int,
    *,
    heading: str,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    h = (heading or "").strip()
    if not h:
        raise PubChemError("heading is required for pug_view_section")
    encoded = urllib.parse.quote(h, safe="")
    url = f"{_PUBCHEM_PUG_VIEW_BASE}/data/compound/{int(cid)}/JSON?heading={encoded}"
    return _http_get_json(url, timeout_s=timeout_s)


def fetch_pug_view_record(cid: int, *, timeout_s: float = 8.0) -> dict[str, Any]:
    """Fetch the full PUG-View record (large). Prefer fetch_pug_view_section when possible."""
    url = f"{_PUBCHEM_PUG_VIEW_BASE}/data/compound/{int(cid)}/JSON"
    return _http_get_json(url, timeout_s=timeout_s)


def fetch_pubchem_evidence(
    *,
    query: str = "",
    cid: int | None = None,
    op: str,
    heading: str | None = None,
    properties: list[str] | None = None,
    timeout_s: float = 8.0,
) -> PubChemEvidence:
    """Fetch PubChem data for agent reasoning and return a trace-friendly evidence payload.

    Design goals:
    - Do NOT over-normalize: return raw-ish evidence so the model can interpret condition dependence.
    - Always be robust: never raise; return status=error with details.
    - Include a bounded raw JSON snapshot for auditability.
    """
    q = (query or "").strip()
    op_norm = str(op or "").strip()
    try:
        if cid is None:
            cids = resolve_cids(q, timeout_s=timeout_s)
            cid = int(cids[0]) if cids else None
        if cid is None:
            return PubChemEvidence(
                status="unresolved",
                query=q,
                cid=None,
                op=op_norm,
                heading=(heading or "").strip() or None,
                properties=[p for p in (properties or []) if str(p or "").strip()] or None,
                extracted=None,
                raw_json=None,
                raw_truncated=False,
                error="no_cid",
            )

        if op_norm == "resolve":
            cids2 = resolve_cids(q, timeout_s=timeout_s)
            extracted = {"cids": cids2}
            raw, trunc = _sanitize_json({"cids": cids2}, max_depth=3, max_list=50, max_str=400)
            return PubChemEvidence(
                status="ok",
                query=q,
                cid=int(cid),
                op=op_norm,
                heading=None,
                properties=None,
                extracted=extracted,
                raw_json=raw if isinstance(raw, dict) else {"raw": raw},
                raw_truncated=bool(trunc),
            )

        if op_norm == "property_table":
            props = [str(p).strip() for p in (properties or []) if str(p or "").strip()]
            raw_obj = fetch_property_table(int(cid), properties=props or ["MolecularWeight", "ExactMass"], timeout_s=timeout_s)
            # Extract the first row of values (best-effort).
            extracted: dict[str, Any] = {"cid": int(cid)}
            table = raw_obj.get("PropertyTable")
            properties_rows = (table.get("Properties") if isinstance(table, dict) else None) if table is not None else None
            first = properties_rows[0] if isinstance(properties_rows, list) and properties_rows else {}
            if isinstance(first, dict):
                for k, v in first.items():
                    if k == "CID":
                        continue
                    extracted[str(k)] = v
            raw_s, trunc = _sanitize_json(raw_obj)
            return PubChemEvidence(
                status="ok",
                query=q,
                cid=int(cid),
                op=op_norm,
                heading=None,
                properties=props or None,
                extracted=extracted,
                raw_json=raw_s if isinstance(raw_s, dict) else {"raw": raw_s},
                raw_truncated=bool(trunc),
            )

        if op_norm == "pug_view_toc":
            raw_obj = fetch_pug_view_record(int(cid), timeout_s=timeout_s)
            # Headings are nested; extract a compact list.
            headings: list[str] = []
            try:
                record = raw_obj.get("Record") if isinstance(raw_obj, dict) else None
                sections = record.get("Section") if isinstance(record, dict) else None
                stack = sections[:] if isinstance(sections, list) else []
                while stack:
                    sec = stack.pop(0)
                    if not isinstance(sec, dict):
                        continue
                    toc = str(sec.get("TOCHeading") or "").strip()
                    if toc and toc not in headings:
                        headings.append(toc)
                    children = sec.get("Section")
                    if isinstance(children, list):
                        stack.extend(children)
            except Exception:
                headings = []
            extracted = {"headings": headings[:200]}
            raw_s, trunc = _sanitize_json(raw_obj)
            return PubChemEvidence(
                status="ok",
                query=q,
                cid=int(cid),
                op=op_norm,
                heading=None,
                properties=None,
                extracted=extracted,
                raw_json=raw_s if isinstance(raw_s, dict) else {"raw": raw_s},
                raw_truncated=bool(trunc),
            )

        if op_norm == "pug_view_section":
            h = (heading or "").strip()
            raw_obj = fetch_pug_view_section(int(cid), heading=h, timeout_s=timeout_s)
            strings = _collect_pubchem_strings(raw_obj, limit=120)
            extracted = {"snippets": strings[:120]}
            raw_s, trunc = _sanitize_json(raw_obj)
            return PubChemEvidence(
                status="ok",
                query=q,
                cid=int(cid),
                op=op_norm,
                heading=h or None,
                properties=None,
                extracted=extracted,
                raw_json=raw_s if isinstance(raw_s, dict) else {"raw": raw_s},
                raw_truncated=bool(trunc),
            )

        return PubChemEvidence(
            status="error",
            query=q,
            cid=int(cid),
            op=op_norm,
            heading=(heading or "").strip() or None,
            properties=[p for p in (properties or []) if str(p or "").strip()] or None,
            extracted=None,
            raw_json=None,
            raw_truncated=False,
            error=f"unknown_op:{op_norm}",
        )
    except Exception as e:
        return PubChemEvidence(
            status="error",
            query=q,
            cid=int(cid) if cid is not None else None,
            op=op_norm,
            heading=(heading or "").strip() or None,
            properties=[p for p in (properties or []) if str(p or "").strip()] or None,
            extracted=None,
            raw_json=None,
            raw_truncated=False,
            error=str(e),
        )
