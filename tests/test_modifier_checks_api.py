from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.storage.sqlite_store import SQLiteStore
from src.tools.pubchem import PubChemResolution


def _seed_run_with_final_output(*, db_path: str, modifier: str) -> str:
    store = SQLiteStore(db_path)
    try:
        batch = store.create_batch(
            user_request="test",
            n_runs=1,
            recipes_per_run=1,
            config={"dry_run": True},
        )
        run = store.create_run(batch_id=batch.batch_id, run_index=1)

        store.append_event(
            run.run_id,
            "final_output",
            {
                "recipes_json": {
                    "recipes": [
                        {
                            "M1": "Cu",
                            "M2": "Mo",
                            "atomic_ratio": "1:1",
                            "small_molecule_modifier": modifier,
                            "rationale": "because",
                        }
                    ]
                },
                "citations": {},
                "memory_ids": [],
            },
        )
        return run.run_id
    finally:
        store.close()


def test_modifier_checks_endpoint_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the endpoint does not crash when the latest event row is a sqlite3.Row
    # (sqlite3.Row does not implement .get()) and that the results are cached as a trace event.
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "app.db")
        monkeypatch.setenv("C2XC_SQLITE_PATH", db_path)
        monkeypatch.setenv("C2XC_ENABLE_WORKER", "0")

        modifier = "benzoic acid (-COOH)"
        run_id = _seed_run_with_final_output(db_path=db_path, modifier=modifier)

        calls: dict[str, int] = {"n": 0}

        def _fake_resolve_pubchem(name: str, *, timeout_s: float = 8.0) -> PubChemResolution:
            calls["n"] += 1
            base = PubChemResolution(
                query=name,
                normalized_query="benzoic acid",
                status="resolved",
                cid=243,
                canonical_smiles="O=C(O)c1ccccc1",
                inchikey="WPYMKLBDIGXBTP-UHFFFAOYSA-N",
                has_cooh=True,
                error=None,
            )
            # Ensure we still reflect the specific `name` passed in (the router uses `res.query`).
            return replace(base, query=name)

        # The router imports `resolve_pubchem` directly, so patch the module attribute.
        import src.api.routers.runs as runs_router  # noqa: PLC0415

        monkeypatch.setattr(runs_router, "resolve_pubchem", _fake_resolve_pubchem)

        app = create_app()
        with TestClient(app) as client:
            # GET should not compute; cache is missing before the first POST.
            resp0 = client.get(f"/api/v1/runs/{run_id}/modifier_checks")
            assert resp0.status_code == 404
            env0 = resp0.json()
            assert env0["error"]["code"] == "not_found"

            resp1 = client.post(f"/api/v1/runs/{run_id}/modifier_checks", json={"force": False})
            assert resp1.status_code == 200
            payload1 = resp1.json()
            assert payload1["run_id"] == run_id
            assert isinstance(payload1["items"], list)
            assert len(payload1["items"]) == 1
            assert payload1["items"][0]["query"] == modifier
            assert payload1["items"][0]["status"] == "resolved"
            assert payload1["items"][0]["cid"] == 243
            assert payload1["items"][0]["has_cooh"] is True

            # After compute, GET should return the cached data and should not hit PubChem again.
            resp_get = client.get(f"/api/v1/runs/{run_id}/modifier_checks")
            assert resp_get.status_code == 200
            payload_get = resp_get.json()
            assert payload_get["items"][0]["query"] == modifier
            assert calls["n"] == 1

            # Second call should hit the cached trace event and not call PubChem again.
            resp2 = client.post(f"/api/v1/runs/{run_id}/modifier_checks", json={"force": False})
            assert resp2.status_code == 200
            payload2 = resp2.json()
            assert payload2["items"][0]["query"] == modifier
            assert calls["n"] == 1
