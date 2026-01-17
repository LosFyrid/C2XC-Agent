from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


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
