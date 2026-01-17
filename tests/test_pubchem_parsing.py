from __future__ import annotations

import pytest

import src.tools.pubchem as pubchem


def test_resolve_pubchem_accepts_connectivity_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_http_get_json(url: str, *, timeout_s: float) -> dict:
        if "/cids/JSON" in url:
            return {"IdentifierList": {"CID": [243]}}
        if "/property/" in url:
            # PubChem currently returns ConnectivitySMILES for CanonicalSMILES requests.
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 243,
                            "ConnectivitySMILES": "C1=CC=C(C=C1)C(=O)O",
                            "InChIKey": "WPYMKLBDIGXBTP-UHFFFAOYSA-N",
                        }
                    ]
                }
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(pubchem, "_http_get_json", _fake_http_get_json)

    res = pubchem.resolve_pubchem("benzoic acid (-COOH)")
    assert res.status == "resolved"
    assert res.cid == 243
    assert res.canonical_smiles == "C1=CC=C(C=C1)C(=O)O"
    assert res.has_cooh is True


def test_resolve_pubchem_accepts_smiles_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_http_get_json(url: str, *, timeout_s: float) -> dict:
        if "/cids/JSON" in url:
            return {"IdentifierList": {"CID": [702]}}
        if "/property/" in url:
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 702,
                            "SMILES": "CCO",
                            "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                        }
                    ]
                }
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(pubchem, "_http_get_json", _fake_http_get_json)

    res = pubchem.resolve_pubchem("ethanol")
    assert res.status == "resolved"
    assert res.cid == 702
    assert res.canonical_smiles == "CCO"
    assert res.has_cooh is False


def test_resolve_pubchem_prefers_canonical_smiles_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_http_get_json(url: str, *, timeout_s: float) -> dict:
        if "/cids/JSON" in url:
            return {"IdentifierList": {"CID": [1]}}
        if "/property/" in url:
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 1,
                            "CanonicalSMILES": "O=C(O)c1ccccc1",
                            "ConnectivitySMILES": "C1=CC=C(C=C1)C(=O)O",
                            "InChIKey": "X",
                        }
                    ]
                }
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(pubchem, "_http_get_json", _fake_http_get_json)

    res = pubchem.resolve_pubchem("benzoic acid")
    assert res.status == "resolved"
    assert res.canonical_smiles == "O=C(O)c1ccccc1"

