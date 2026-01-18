from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 5


_ALIAS_RE = re.compile(r"^(?P<prefix>[A-Z]+)(?P<num>\d+)$")


def _alias_sort_key(alias: str) -> tuple[int, str]:
    """Return a stable sort key for citation aliases like 'C12'.

    We primarily sort by numeric suffix to avoid lexicographic issues (C10 vs C2).
    """
    s = (alias or "").strip()
    m = _ALIAS_RE.match(s)
    if not m:
        # Put unknown formats at the end, but keep deterministic order.
        return (1_000_000_000, s)
    try:
        return (int(m.group("num")), s)
    except Exception:
        return (1_000_000_000, s)


def _utc_ts() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def default_db_path() -> str:
    return os.getenv("C2XC_SQLITE_PATH", "data/app.db")


@dataclass(frozen=True)
class BatchRecord:
    batch_id: str
    created_at: float
    user_request: str
    n_runs: int
    recipes_per_run: int
    status: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    batch_id: str
    run_index: int
    created_at: float
    status: str


@dataclass(frozen=True)
class ProductRecord:
    product_id: str
    created_at: float
    updated_at: float
    name: str
    status: str


@dataclass(frozen=True)
class ProductPresetRecord:
    preset_id: str
    created_at: float
    updated_at: float
    name: str
    status: str


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    run_id: str
    created_at: float
    updated_at: float
    score: float | None
    pros: str
    cons: str
    other: str


@dataclass(frozen=True)
class RBJobRecord:
    rb_job_id: str
    run_id: str
    kind: str
    created_at: float
    status: str


@dataclass(frozen=True)
class RBDeltaRecord:
    delta_id: str
    run_id: str
    created_at: float
    status: str


class SQLiteStore:
    """SQLite-backed store for runs and trace events.

    Design goals (project constraints):
    - Single-user, single-instance, no multi-worker deployment assumptions.
    - Extensible: most details go into `events.payload_json` instead of hard-coded columns.
    - Trace must be program-recorded and replayable.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or default_db_path()).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")

        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self, *, mode: str = "IMMEDIATE") -> Iterable[None]:
        """Context manager for an explicit SQLite transaction.

        `BEGIN IMMEDIATE` is preferred for idempotent create flows because it
        enforces single-writer semantics across concurrent HTTP requests.
        """
        self._conn.execute(f"BEGIN {mode};")
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )

        # Base schema (v1): batches/runs/events/cancel_requests.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS batches (
              batch_id TEXT PRIMARY KEY,
              created_at REAL NOT NULL,
              started_at REAL,
              ended_at REAL,
              user_request TEXT NOT NULL,
              n_runs INTEGER NOT NULL,
              recipes_per_run INTEGER NOT NULL,
              status TEXT NOT NULL,
              config_json TEXT NOT NULL,
              error TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              batch_id TEXT NOT NULL,
              run_index INTEGER NOT NULL,
              created_at REAL NOT NULL,
              started_at REAL,
              ended_at REAL,
              status TEXT NOT NULL,
              error TEXT,
              FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cancel_requests (
              cancel_id TEXT PRIMARY KEY,
              created_at REAL NOT NULL,
              target_type TEXT NOT NULL,
              target_id TEXT NOT NULL,
              status TEXT NOT NULL,
              reason TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id, run_index);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cancel_target ON cancel_requests(target_type, target_id, created_at);"
        )

        # Important: initialize new databases at schema_version=1 (base tables),
        # then run explicit migrations up to SCHEMA_VERSION.
        cur.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?);",
            ("schema_version", "1"),
        )
        self._conn.commit()

        self._migrate_if_needed()

    def _get_schema_version(self) -> int:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?;", ("schema_version",)).fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except Exception:
            return 0

    def _set_schema_version(self, version: int) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
            ("schema_version", str(int(version))),
        )

    def _migrate_if_needed(self) -> None:
        current = self._get_schema_version()
        target = int(SCHEMA_VERSION)
        if current == target:
            return
        if current > target:
            raise RuntimeError(f"DB schema_version={current} is newer than code expects ({target}).")

        # Apply sequential migrations in a single transaction.
        cur = self._conn.cursor()
        cur.execute("BEGIN;")
        try:
            while current < target:
                if current == 1:
                    self._migrate_1_to_2(cur)
                    current = 2
                    self._set_schema_version(current)
                elif current == 2:
                    self._migrate_2_to_3(cur)
                    current = 3
                    self._set_schema_version(current)
                elif current == 3:
                    self._migrate_3_to_4(cur)
                    current = 4
                    self._set_schema_version(current)
                elif current == 4:
                    self._migrate_4_to_5(cur)
                    current = 5
                    self._set_schema_version(current)
                else:
                    raise RuntimeError(f"Missing migration step for schema_version={current} -> {current+1}")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _migrate_1_to_2(self, cur: sqlite3.Cursor) -> None:
        # Idempotency table for POST /batches (API-level retries).
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
              key TEXT PRIMARY KEY,
              created_at REAL NOT NULL,
              request_hash TEXT NOT NULL,
              response_json TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_batches_status_created ON batches(status, created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at);")
        # Stable pagination: disambiguate same-timestamp events by event_id.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_run_ts_id ON events(run_id, created_at, event_id);")

    def _migrate_2_to_3(self, cur: sqlite3.Cursor) -> None:
        # Milestone 2: Feedback + Products + Presets.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
              product_id TEXT PRIMARY KEY,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              name TEXT NOT NULL COLLATE NOCASE,
              status TEXT NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_products_name ON products(name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_products_status_created ON products(status, created_at);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS product_presets (
              preset_id TEXT PRIMARY KEY,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              name TEXT NOT NULL,
              status TEXT NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_product_presets_status_created ON product_presets(status, created_at);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS product_preset_products (
              preset_id TEXT NOT NULL,
              product_id TEXT NOT NULL,
              position INTEGER NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL,
              PRIMARY KEY (preset_id, product_id),
              FOREIGN KEY (preset_id) REFERENCES product_presets(preset_id) ON DELETE CASCADE,
              FOREIGN KEY (product_id) REFERENCES products(product_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_preset_products_preset ON product_preset_products(preset_id, position);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
              feedback_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL UNIQUE,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              score REAL,
              pros TEXT NOT NULL,
              cons TEXT NOT NULL,
              other TEXT NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL,
              FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback(run_id);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_products (
              feedback_product_id TEXT PRIMARY KEY,
              feedback_id TEXT NOT NULL,
              product_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              value REAL NOT NULL,
              fraction REAL NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL,
              FOREIGN KEY (feedback_id) REFERENCES feedback(feedback_id) ON DELETE CASCADE,
              FOREIGN KEY (product_id) REFERENCES products(product_id),
              UNIQUE (feedback_id, product_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_products_feedback ON feedback_products(feedback_id);")

    def _migrate_3_to_4(self, cur: sqlite3.Cursor) -> None:
        # Milestone 3: ReasoningBank jobs + deltas + memory edit logs (Chroma is the source of truth).
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rb_jobs (
              rb_job_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              created_at REAL NOT NULL,
              started_at REAL,
              ended_at REAL,
              status TEXT NOT NULL,
              error TEXT,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL,
              FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_jobs_run_created ON rb_jobs(run_id, created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_jobs_status_created ON rb_jobs(status, created_at);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rb_deltas (
              delta_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              status TEXT NOT NULL,
              rolled_back_at REAL,
              rolled_back_reason TEXT,
              ops_json TEXT NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL,
              FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_deltas_run_created ON rb_deltas(run_id, created_at);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mem_edit_log (
              mem_edit_id TEXT PRIMARY KEY,
              mem_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              actor TEXT NOT NULL,
              reason TEXT,
              before_json TEXT NOT NULL,
              after_json TEXT NOT NULL,
              schema_version INTEGER NOT NULL,
              extra_json TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_edit_log_mem_created ON mem_edit_log(mem_id, created_at);")

    def _migrate_4_to_5(self, cur: sqlite3.Cursor) -> None:
        # Milestone 3 follow-up: fast browse pagination for RB memories.
        #
        # Chroma remains the source of truth for full content and vector search. SQLite stores a lightweight
        # metadata index (no content) to support stable newest-first browsing without O(N) scans.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rb_mem_index (
              mem_id TEXT PRIMARY KEY,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              status TEXT NOT NULL,
              role TEXT NOT NULL,
              type TEXT NOT NULL,
              source_run_id TEXT,
              schema_version INTEGER NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_mem_index_created ON rb_mem_index(created_at, mem_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_mem_index_status_created ON rb_mem_index(status, created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_mem_index_role_created ON rb_mem_index(role, created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rb_mem_index_type_created ON rb_mem_index(type, created_at);")

    # --- Idempotency (API support)
    def get_idempotency(self, key: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT key, created_at, request_hash, response_json FROM idempotency_keys WHERE key = ? LIMIT 1;",
            (key,),
        ).fetchone()

    def put_idempotency(
        self, *, key: str, request_hash: str, response_json: str, commit: bool = True
    ) -> None:
        created_at = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO idempotency_keys(key, created_at, request_hash, response_json)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(key) DO NOTHING;
            """,
            (key, created_at, request_hash, response_json),
        )
        if commit:
            self._conn.commit()

    # --- Batches / Runs
    def get_batch(self, *, batch_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT
              batch_id, created_at, started_at, ended_at, user_request, n_runs, recipes_per_run,
              status, config_json, error
            FROM batches
            WHERE batch_id = ?
            LIMIT 1;
            """,
            (batch_id,),
        ).fetchone()

    def get_run(self, *, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT
              run_id, batch_id, run_index, created_at, started_at, ended_at, status, error
            FROM runs
            WHERE run_id = ?
            LIMIT 1;
            """,
            (run_id,),
        ).fetchone()

    def list_batches_page(
        self,
        *,
        limit: int,
        cursor: tuple[float, str] | None,
        statuses: list[str] | None,
    ) -> dict[str, Any]:
        where = ["1=1"]
        params: list[Any] = []

        if statuses:
            where.append("status IN (%s)" % ",".join(["?"] * len(statuses)))
            params.extend(statuses)

        if cursor is not None:
            created_at, batch_id = cursor
            # Newest-first pagination (DESC).
            where.append("(created_at < ? OR (created_at = ? AND batch_id < ?))")
            params.extend([float(created_at), float(created_at), str(batch_id)])

        where_sql = " AND ".join(where)
        fetch_n = int(limit) + 1

        rows = self._conn.execute(
            f"""
            SELECT
              batch_id, created_at, started_at, ended_at,
              user_request, n_runs, recipes_per_run, status, error
            FROM batches
            WHERE {where_sql}
            ORDER BY created_at DESC, batch_id DESC
            LIMIT ?;
            """,
            (*params, fetch_n),
        ).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [
            {
                "batch_id": r["batch_id"],
                "created_at": float(r["created_at"]),
                "started_at": float(r["started_at"]) if r["started_at"] is not None else None,
                "ended_at": float(r["ended_at"]) if r["ended_at"] is not None else None,
                "user_request": r["user_request"],
                "n_runs": int(r["n_runs"]),
                "recipes_per_run": int(r["recipes_per_run"]),
                "status": r["status"],
                "error": r["error"],
            }
            for r in rows
        ]

        next_cursor: tuple[float, str] | None = None
        if has_more and items:
            last = items[-1]
            next_cursor = (float(last["created_at"]), str(last["batch_id"]))

        return {"items": items, "has_more": has_more, "next_cursor": next_cursor}

    def list_runs_page(
        self,
        *,
        batch_id: str | None,
        limit: int,
        cursor: tuple[float, str] | None,
        statuses: list[str] | None,
    ) -> dict[str, Any]:
        where = ["1=1"]
        params: list[Any] = []

        if batch_id:
            where.append("batch_id = ?")
            params.append(batch_id)

        if statuses:
            where.append("status IN (%s)" % ",".join(["?"] * len(statuses)))
            params.extend(statuses)

        if cursor is not None:
            created_at, run_id = cursor
            where.append("(created_at < ? OR (created_at = ? AND run_id < ?))")
            params.extend([float(created_at), float(created_at), str(run_id)])

        where_sql = " AND ".join(where)
        fetch_n = int(limit) + 1

        rows = self._conn.execute(
            f"""
            SELECT
              run_id, batch_id, run_index, created_at, started_at, ended_at, status, error
            FROM runs
            WHERE {where_sql}
            ORDER BY created_at DESC, run_id DESC
            LIMIT ?;
            """,
            (*params, fetch_n),
        ).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [
            {
                "run_id": r["run_id"],
                "batch_id": r["batch_id"],
                "run_index": int(r["run_index"]),
                "created_at": float(r["created_at"]),
                "started_at": float(r["started_at"]) if r["started_at"] is not None else None,
                "ended_at": float(r["ended_at"]) if r["ended_at"] is not None else None,
                "status": r["status"],
                "error": r["error"],
            }
            for r in rows
        ]

        next_cursor: tuple[float, str] | None = None
        if has_more and items:
            last = items[-1]
            next_cursor = (float(last["created_at"]), str(last["run_id"]))

        return {"items": items, "has_more": has_more, "next_cursor": next_cursor}

    def count_runs_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM runs GROUP BY status ORDER BY status;",
        ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def count_batches_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM batches GROUP BY status ORDER BY status;",
        ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def count_rb_jobs_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM rb_jobs GROUP BY status ORDER BY status;",
        ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def list_runs_for_batch_rows(self, *, batch_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT
              run_id, batch_id, run_index, created_at, started_at, ended_at, status, error
            FROM runs
            WHERE batch_id = ?
            ORDER BY run_index;
            """,
            (batch_id,),
        ).fetchall()

    def create_batch(
        self,
        *,
        user_request: str,
        n_runs: int,
        recipes_per_run: int,
        config: dict[str, Any],
        commit: bool = True,
    ) -> BatchRecord:
        batch_id = _new_id("batch")
        created_at = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO batches(
              batch_id, created_at, user_request, n_runs, recipes_per_run, status, config_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            (
                batch_id,
                created_at,
                user_request,
                n_runs,
                recipes_per_run,
                "queued",
                _json_dumps(config),
            ),
        )
        if commit:
            self._conn.commit()
        return BatchRecord(
            batch_id=batch_id,
            created_at=created_at,
            user_request=user_request,
            n_runs=n_runs,
            recipes_per_run=recipes_per_run,
            status="queued",
        )

    def update_batch_status(self, batch_id: str, status: str, *, error: str | None = None) -> None:
        ts = _utc_ts()
        started_at = ts if status == "running" else None
        ended_at = ts if status in {"completed", "failed", "canceled"} else None

        # Only set started_at if previously null; same for ended_at.
        self._conn.execute(
            """
            UPDATE batches
            SET
              status = ?,
              started_at = COALESCE(started_at, ?),
              ended_at = COALESCE(ended_at, ?),
              error = COALESCE(?, error)
            WHERE batch_id = ?;
            """,
            (status, started_at, ended_at, error, batch_id),
        )
        self._conn.commit()

    def create_run(self, *, batch_id: str, run_index: int, commit: bool = True) -> RunRecord:
        run_id = _new_id("run")
        created_at = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO runs(
              run_id, batch_id, run_index, created_at, status
            ) VALUES(?, ?, ?, ?, ?);
            """,
            (run_id, batch_id, run_index, created_at, "queued"),
        )
        if commit:
            self._conn.commit()
        return RunRecord(
            run_id=run_id,
            batch_id=batch_id,
            run_index=run_index,
            created_at=created_at,
            status="queued",
        )

    def update_run_status(self, run_id: str, status: str, *, error: str | None = None) -> None:
        ts = _utc_ts()
        started_at = ts if status == "running" else None
        ended_at = ts if status in {"completed", "failed", "canceled"} else None

        self._conn.execute(
            """
            UPDATE runs
            SET
              status = ?,
              started_at = COALESCE(started_at, ?),
              ended_at = COALESCE(ended_at, ?),
              error = COALESCE(?, error)
            WHERE run_id = ?;
            """,
            (status, started_at, ended_at, error, run_id),
        )
        self._conn.commit()

    def list_runs_for_batch(self, batch_id: str) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT run_id, batch_id, run_index, created_at, status FROM runs WHERE batch_id = ? ORDER BY run_index;",
            (batch_id,),
        ).fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                batch_id=r["batch_id"],
                run_index=int(r["run_index"]),
                created_at=float(r["created_at"]),
                status=r["status"],
            )
            for r in rows
        ]

    # --- Reconcile (startup safety)
    def reconcile_running_runs(self, *, reason: str = "server_restarted") -> int:
        """Mark any 'running' runs as failed.

        This prevents "stuck running forever" after a process restart.
        Returns the number of runs reconciled.
        """
        ts = _utc_ts()
        rows = self._conn.execute(
            "SELECT run_id FROM runs WHERE status = 'running';",
        ).fetchall()
        if not rows:
            return 0

        run_ids = [r["run_id"] for r in rows]
        for run_id in run_ids:
            self._conn.execute(
                """
                UPDATE runs
                SET
                  status = 'failed',
                  ended_at = COALESCE(ended_at, ?),
                  error = COALESCE(error, ?)
                WHERE run_id = ? AND status = 'running';
                """,
                (ts, reason, run_id),
            )
            # Best-effort: record a trace event for UI debugging.
            self._conn.execute(
                """
                INSERT INTO events(event_id, run_id, created_at, event_type, payload_json)
                VALUES(?, ?, ?, ?, ?);
                """,
                (_new_id("evt"), run_id, ts, "run_failed", _json_dumps({"error": reason})),
            )

        self._conn.commit()
        return len(run_ids)

    # --- Queue helpers (single-writer safe claims)
    def claim_next_queued_run(self) -> sqlite3.Row | None:
        """Atomically claim the next queued run and mark it as running.

        This is designed to be safe even if a future deployment accidentally
        starts multiple workers in the same SQLite DB.
        """
        with self.transaction(mode="IMMEDIATE"):
            row = self._conn.execute(
                """
                SELECT run_id, batch_id
                FROM runs
                WHERE status = 'queued'
                ORDER BY created_at ASC, run_id ASC
                LIMIT 1;
                """
            ).fetchone()
            if row is None:
                return None

            ts = _utc_ts()
            run_id = str(row["run_id"])
            batch_id = str(row["batch_id"])

            updated = self._conn.execute(
                """
                UPDATE runs
                SET
                  status = 'running',
                  started_at = COALESCE(started_at, ?)
                WHERE run_id = ? AND status = 'queued';
                """,
                (ts, run_id),
            )
            if updated.rowcount != 1:
                return None

            # Ensure the batch is marked as running (only sets started_at if null).
            self._conn.execute(
                """
                UPDATE batches
                SET
                  status = 'running',
                  started_at = COALESCE(started_at, ?)
                WHERE batch_id = ? AND status = 'queued';
                """,
                (ts, batch_id),
            )

            claimed = self.get_run(run_id=run_id)
            return claimed

    # --- Events (trace)
    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> str:
        event_id = _new_id("evt")
        created_at = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO events(event_id, run_id, created_at, event_type, payload_json)
            VALUES(?, ?, ?, ?, ?);
            """,
            (event_id, run_id, created_at, event_type, _json_dumps(payload)),
        )
        self._conn.commit()
        return event_id

    def iter_events(self, run_id: str) -> Iterable[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT created_at, event_type, payload_json FROM events WHERE run_id = ? ORDER BY created_at;",
            (run_id,),
        )
        for r in rows:
            yield {
                "created_at": float(r["created_at"]),
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]),
            }

    def get_event(self, *, run_id: str, event_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT event_id, run_id, created_at, event_type, payload_json
            FROM events
            WHERE run_id = ? AND event_id = ?
            LIMIT 1;
            """,
            (run_id, event_id),
        ).fetchone()

    def get_latest_event(self, *, run_id: str, event_type: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT event_id, run_id, created_at, event_type, payload_json
            FROM events
            WHERE run_id = ? AND event_type = ?
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1;
            """,
            (run_id, event_type),
        ).fetchone()

    def count_event_types_for_run(self, *, run_id: str, until: float | None = None) -> dict[str, int]:
        rid = (run_id or "").strip()
        if not rid:
            return {}
        where = ["run_id = ?"]
        params: list[Any] = [rid]
        if until is not None:
            where.append("created_at <= ?")
            params.append(float(until))
        where_sql = " AND ".join(where)
        rows = self._conn.execute(
            f"SELECT event_type, COUNT(*) AS n FROM events WHERE {where_sql} GROUP BY event_type ORDER BY event_type;",
            params,
        ).fetchall()
        return {str(r["event_type"]): int(r["n"]) for r in rows}

    def list_latest_events(
        self,
        *,
        run_id: str,
        limit: int,
        event_types: list[str] | None,
        include_payload: bool,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        rid = (run_id or "").strip()
        if not rid:
            return []

        where = ["run_id = ?"]
        params: list[Any] = [rid]
        if event_types:
            where.append("event_type IN (%s)" % ",".join(["?"] * len(event_types)))
            params.extend([str(t) for t in event_types])
        if since is not None:
            where.append("created_at >= ?")
            params.append(float(since))
        if until is not None:
            where.append("created_at <= ?")
            params.append(float(until))
        where_sql = " AND ".join(where)

        if include_payload:
            sql = (
                "SELECT event_id, run_id, created_at, event_type, payload_json "
                "FROM events "
                f"WHERE {where_sql} "
                "ORDER BY created_at DESC, event_id DESC "
                "LIMIT ?"
            )
        else:
            sql = (
                "SELECT event_id, run_id, created_at, event_type "
                "FROM events "
                f"WHERE {where_sql} "
                "ORDER BY created_at DESC, event_id DESC "
                "LIMIT ?"
            )

        rows = self._conn.execute(sql, (*params, int(limit))).fetchall()

        items: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {
                "event_id": str(r["event_id"]),
                "run_id": str(r["run_id"]),
                "created_at": float(r["created_at"]),
                "event_type": str(r["event_type"]),
            }
            if include_payload:
                try:
                    item["payload"] = json.loads(str(r["payload_json"]))
                except Exception:
                    item["payload"] = {}
            items.append(item)
        return items

    def list_events_page(
        self,
        *,
        run_id: str,
        limit: int,
        cursor: tuple[float, str] | None,
        event_types: list[str] | None,
        include_payload: bool,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[str, Any]:
        # Cursor-based pagination, stable ordering by (created_at, event_id).
        where = ["run_id = ?"]
        params: list[Any] = [run_id]

        if event_types:
            where.append("event_type IN (%s)" % ",".join(["?"] * len(event_types)))
            params.extend(event_types)

        if since is not None:
            where.append("created_at >= ?")
            params.append(float(since))
        if until is not None:
            where.append("created_at <= ?")
            params.append(float(until))

        if cursor is not None:
            created_at, event_id = cursor
            # Fetch strictly after the cursor to avoid duplicates.
            where.append("(created_at > ? OR (created_at = ? AND event_id > ?))")
            params.extend([float(created_at), float(created_at), str(event_id)])

        where_sql = " AND ".join(where)
        # Fetch one extra row to determine has_more.
        fetch_n = int(limit) + 1

        if include_payload:
            sql = (
                "SELECT event_id, run_id, created_at, event_type, payload_json "
                "FROM events "
                f"WHERE {where_sql} "
                "ORDER BY created_at ASC, event_id ASC "
                "LIMIT ?"
            )
        else:
            sql = (
                "SELECT event_id, run_id, created_at, event_type "
                "FROM events "
                f"WHERE {where_sql} "
                "ORDER BY created_at ASC, event_id ASC "
                "LIMIT ?"
            )

        rows = self._conn.execute(sql, (*params, fetch_n)).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {
                "event_id": r["event_id"],
                "run_id": r["run_id"],
                "created_at": float(r["created_at"]),
                "event_type": r["event_type"],
            }
            if include_payload:
                item["payload"] = json.loads(r["payload_json"])
            items.append(item)

        next_cursor: tuple[float, str] | None = None
        if has_more and items:
            last = items[-1]
            next_cursor = (float(last["created_at"]), str(last["event_id"]))

        return {"items": items, "has_more": has_more, "next_cursor": next_cursor}

    # --- Evidence (KB alias -> chunk mapping; aggregated from trace)
    def _collect_run_evidence(self, *, run_id: str) -> dict[str, dict[str, Any]]:
        """Aggregate per-run evidence from trace events.

        Evidence is stored inside trace events (append-only). For UI performance,
        we expose a read API that aggregates alias -> evidence information.

        Currently supported evidence event types:
        - kb_query (LightRAG chunk evidence; aliases like C1)
        - pubchem_query (PubChem evidence; aliases like P1)
        """
        rows = self._conn.execute(
            """
            SELECT created_at, event_id, payload_json
            FROM events
            WHERE run_id = ? AND event_type IN ('kb_query', 'pubchem_query')
            ORDER BY created_at ASC, event_id ASC;
            """,
            (run_id,),
        ).fetchall()

        evidence: dict[str, dict[str, Any]] = {}
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:
                continue

            results = payload.get("results") or []
            if not isinstance(results, list):
                continue

            created_at = float(r["created_at"])
            for item in results:
                if not isinstance(item, dict):
                    continue
                alias = str(item.get("alias") or "").strip()
                if alias.startswith("[") and alias.endswith("]"):
                    alias = alias[1:-1].strip()
                if not alias or alias in evidence:
                    continue

                evidence[alias] = {
                    "alias": alias,
                    "ref": str(item.get("ref") or "").strip(),
                    "source": str(item.get("source") or "").strip(),
                    "content": str(item.get("content") or ""),
                    "kb_namespace": str(item.get("kb_namespace") or "").strip()
                    or str(payload.get("kb_namespace") or "").strip(),
                    "lightrag_chunk_id": str(item.get("lightrag_chunk_id") or "").strip() or None,
                    "created_at": created_at,
                }

        return evidence

    def list_evidence_page(
        self,
        *,
        run_id: str,
        limit: int,
        cursor: tuple[float, str] | None,
        include_content: bool,
    ) -> dict[str, Any]:
        """List evidence items for a run.

        Cursor semantics:
        - `cursor` is (created_at, alias) as returned by the previous page.
        - We paginate in alias numeric order (C1, C2, ...), not lexicographic,
          to keep UI behavior intuitive.
        """
        evidence_map = self._collect_run_evidence(run_id=run_id)
        items = list(evidence_map.values())
        items.sort(key=lambda e: _alias_sort_key(str(e.get("alias") or "")))

        if cursor is not None:
            _cursor_created_at, cursor_alias = cursor
            cursor_key = _alias_sort_key(str(cursor_alias))
            items = [e for e in items if _alias_sort_key(str(e.get("alias") or "")) > cursor_key]

        fetch_n = int(limit) + 1
        page_items = items[:fetch_n]

        has_more = len(page_items) > limit
        if has_more:
            page_items = page_items[:limit]

        if not include_content:
            for e in page_items:
                e.pop("content", None)

        next_cursor: tuple[float, str] | None = None
        if has_more and page_items:
            last = page_items[-1]
            next_cursor = (float(last["created_at"]), str(last["alias"]))

        return {"items": page_items, "has_more": has_more, "next_cursor": next_cursor}

    def get_evidence_item(self, *, run_id: str, alias: str) -> dict[str, Any] | None:
        alias = (alias or "").strip()
        if not alias:
            return None
        evidence_map = self._collect_run_evidence(run_id=run_id)
        return evidence_map.get(alias)

    # --- ReasoningBank (Milestone 3): jobs + deltas + edit logs
    def create_rb_job(
        self,
        *,
        run_id: str,
        kind: str = "learn",
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> RBJobRecord:
        rid = (run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required.")
        if self.get_run(run_id=rid) is None:
            raise ValueError("Run not found.")
        cleaned_kind = (kind or "").strip() or "learn"

        rb_job_id = _new_id("rbjob")
        ts = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO rb_jobs(
              rb_job_id, run_id, kind, created_at, status, schema_version, extra_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            (
                rb_job_id,
                rid,
                cleaned_kind,
                ts,
                "queued",
                int(schema_version),
                _json_dumps(extra or {}),
            ),
        )
        if commit:
            self._conn.commit()
        return RBJobRecord(rb_job_id=rb_job_id, run_id=rid, kind=cleaned_kind, created_at=ts, status="queued")

    def get_rb_job(self, *, rb_job_id: str) -> sqlite3.Row | None:
        jid = (rb_job_id or "").strip()
        if not jid:
            return None
        return self._conn.execute(
            """
            SELECT rb_job_id, run_id, kind, created_at, started_at, ended_at, status, error, schema_version, extra_json
            FROM rb_jobs
            WHERE rb_job_id = ?
            LIMIT 1;
            """,
            (jid,),
        ).fetchone()

    def get_latest_rb_job_for_run(
        self,
        *,
        run_id: str,
        kind: str | None = None,
        statuses: list[str] | None = None,
    ) -> sqlite3.Row | None:
        rid = (run_id or "").strip()
        if not rid:
            return None
        where = ["run_id = ?"]
        params: list[Any] = [rid]
        if kind is not None:
            where.append("kind = ?")
            params.append(str(kind))
        if statuses:
            where.append("status IN (%s)" % ",".join(["?"] * len(statuses)))
            params.extend([str(s) for s in statuses])
        where_sql = " AND ".join(where)
        return self._conn.execute(
            f"""
            SELECT rb_job_id, run_id, kind, created_at, started_at, ended_at, status, error, schema_version, extra_json
            FROM rb_jobs
            WHERE {where_sql}
            ORDER BY created_at DESC, rb_job_id DESC
            LIMIT 1;
            """,
            params,
        ).fetchone()

    def list_rb_jobs_for_run(
        self,
        *,
        run_id: str,
        limit: int = 50,
        kind: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        rid = (run_id or "").strip()
        if not rid:
            return []

        where = ["run_id = ?"]
        params: list[Any] = [rid]
        if kind is not None:
            where.append("kind = ?")
            params.append(str(kind))
        if statuses:
            where.append("status IN (%s)" % ",".join(["?"] * len(statuses)))
            params.extend([str(s) for s in statuses])
        where_sql = " AND ".join(where)

        rows = self._conn.execute(
            f"""
            SELECT rb_job_id, run_id, kind, created_at, started_at, ended_at, status, error, schema_version, extra_json
            FROM rb_jobs
            WHERE {where_sql}
            ORDER BY created_at DESC, rb_job_id DESC
            LIMIT ?;
            """,
            (*params, int(limit)),
        ).fetchall()

        items: list[dict[str, Any]] = []
        for r in rows:
            try:
                extra = json.loads(str(r["extra_json"] or "{}"))
                if not isinstance(extra, dict):
                    extra = {}
            except Exception:
                extra = {}
            items.append(
                {
                    "rb_job_id": str(r["rb_job_id"]),
                    "run_id": str(r["run_id"]),
                    "kind": str(r["kind"]),
                    "created_at": float(r["created_at"]),
                    "started_at": float(r["started_at"]) if r["started_at"] is not None else None,
                    "ended_at": float(r["ended_at"]) if r["ended_at"] is not None else None,
                    "status": str(r["status"]),
                    "error": str(r["error"]) if r["error"] is not None else None,
                    "schema_version": int(r["schema_version"]),
                    "extra": extra,
                }
            )
        return items

    def claim_next_queued_rb_job(self) -> sqlite3.Row | None:
        """Atomically claim the next queued RB job and mark it as running."""
        with self.transaction(mode="IMMEDIATE"):
            row = self._conn.execute(
                """
                SELECT rb_job_id, run_id, kind
                FROM rb_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC, rb_job_id ASC
                LIMIT 1;
                """
            ).fetchone()
            if row is None:
                return None

            ts = _utc_ts()
            rb_job_id = str(row["rb_job_id"])

            updated = self._conn.execute(
                """
                UPDATE rb_jobs
                SET
                  status = 'running',
                  started_at = COALESCE(started_at, ?)
                WHERE rb_job_id = ? AND status = 'queued';
                """,
                (ts, rb_job_id),
            )
            if updated.rowcount != 1:
                return None

            return self.get_rb_job(rb_job_id=rb_job_id)

    def update_rb_job_status(self, rb_job_id: str, status: str, *, error: str | None = None) -> None:
        ts = _utc_ts()
        started_at = ts if status == "running" else None
        ended_at = ts if status in {"completed", "failed", "canceled"} else None
        self._conn.execute(
            """
            UPDATE rb_jobs
            SET
              status = ?,
              started_at = COALESCE(started_at, ?),
              ended_at = COALESCE(ended_at, ?),
              error = COALESCE(?, error)
            WHERE rb_job_id = ?;
            """,
            (status, started_at, ended_at, error, rb_job_id),
        )
        self._conn.commit()

    def create_rb_delta(
        self,
        *,
        run_id: str,
        ops: list[dict[str, Any]],
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> RBDeltaRecord:
        rid = (run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required.")
        if self.get_run(run_id=rid) is None:
            raise ValueError("Run not found.")
        if not isinstance(ops, list):
            raise ValueError("ops must be a list.")

        delta_id = _new_id("rbd")
        ts = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO rb_deltas(
              delta_id, run_id, created_at, status, ops_json, schema_version, extra_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            (
                delta_id,
                rid,
                ts,
                "applied",
                _json_dumps(ops),
                int(schema_version),
                _json_dumps(extra or {}),
            ),
        )
        if commit:
            self._conn.commit()
        return RBDeltaRecord(delta_id=delta_id, run_id=rid, created_at=ts, status="applied")

    def get_rb_delta(self, *, delta_id: str) -> sqlite3.Row | None:
        did = (delta_id or "").strip()
        if not did:
            return None
        return self._conn.execute(
            """
            SELECT
              delta_id, run_id, created_at, status, rolled_back_at, rolled_back_reason,
              ops_json, schema_version, extra_json
            FROM rb_deltas
            WHERE delta_id = ?
            LIMIT 1;
            """,
            (did,),
        ).fetchone()

    def list_rb_deltas_for_run(self, *, run_id: str) -> list[dict[str, Any]]:
        rid = (run_id or "").strip()
        if not rid:
            return []
        rows = self._conn.execute(
            """
            SELECT
              delta_id, run_id, created_at, status, rolled_back_at, rolled_back_reason,
              ops_json, schema_version, extra_json
            FROM rb_deltas
            WHERE run_id = ?
            ORDER BY created_at DESC, delta_id DESC;
            """,
            (rid,),
        ).fetchall()
        items: list[dict[str, Any]] = []
        for r in rows:
            try:
                ops = json.loads(str(r["ops_json"] or "[]"))
            except Exception:
                ops = []
            items.append(
                {
                    "delta_id": str(r["delta_id"]),
                    "run_id": str(r["run_id"]),
                    "created_at": float(r["created_at"]),
                    "status": str(r["status"]),
                    "rolled_back_at": float(r["rolled_back_at"]) if r["rolled_back_at"] is not None else None,
                    "rolled_back_reason": str(r["rolled_back_reason"] or "") or None,
                    "ops": ops,
                    "schema_version": int(r["schema_version"]),
                    "extra": json.loads(str(r["extra_json"] or "{}")),
                }
            )
        return items

    def mark_rb_delta_rolled_back(self, *, delta_id: str, reason: str | None = None) -> None:
        ts = _utc_ts()
        did = (delta_id or "").strip()
        if not did:
            raise ValueError("delta_id is required.")
        self._conn.execute(
            """
            UPDATE rb_deltas
            SET
              status = 'rolled_back',
              rolled_back_at = COALESCE(rolled_back_at, ?),
              rolled_back_reason = COALESCE(rolled_back_reason, ?)
            WHERE delta_id = ?;
            """,
            (ts, str(reason or "") or None, did),
        )
        self._conn.commit()

    def append_mem_edit_log(
        self,
        *,
        mem_id: str,
        actor: str,
        reason: str | None,
        before: dict[str, Any],
        after: dict[str, Any],
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> str:
        mid = (mem_id or "").strip()
        if not mid:
            raise ValueError("mem_id is required.")
        act = (actor or "").strip() or "unknown"
        edit_id = _new_id("memedit")
        ts = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO mem_edit_log(
              mem_edit_id, mem_id, created_at, actor, reason, before_json, after_json, schema_version, extra_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                edit_id,
                mid,
                ts,
                act,
                str(reason or "") or None,
                _json_dumps(before),
                _json_dumps(after),
                int(schema_version),
                _json_dumps(extra or {}),
            ),
        )
        if commit:
            self._conn.commit()
        return edit_id

    # --- ReasoningBank memories index (for fast browse pagination)
    def count_rb_mem_index(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM rb_mem_index;").fetchone()
        if row is None:
            return 0
        try:
            return int(row["n"])
        except Exception:
            return 0

    def upsert_rb_mem_index(
        self,
        *,
        mem_id: str,
        created_at: float,
        updated_at: float,
        status: str,
        role: str,
        type: str,  # noqa: A002
        source_run_id: str | None,
        schema_version: int,
        commit: bool = True,
    ) -> None:
        mid = (mem_id or "").strip()
        if not mid:
            raise ValueError("mem_id is required.")
        self._conn.execute(
            """
            INSERT INTO rb_mem_index(
              mem_id, created_at, updated_at, status, role, type, source_run_id, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mem_id) DO UPDATE SET
              created_at = excluded.created_at,
              updated_at = excluded.updated_at,
              status = excluded.status,
              role = excluded.role,
              type = excluded.type,
              source_run_id = excluded.source_run_id,
              schema_version = excluded.schema_version;
            """,
            (
                mid,
                float(created_at),
                float(updated_at),
                str(status),
                str(role),
                str(type),
                str(source_run_id or "") or None,
                int(schema_version),
            ),
        )
        if commit:
            self._conn.commit()

    def upsert_rb_mem_index_many(self, items: list[dict[str, Any]], *, commit: bool = True) -> int:
        """Bulk upsert `rb_mem_index` rows.

        Expected keys per item: mem_id, created_at, updated_at, status, role, type, source_run_id, schema_version.
        Returns number of attempted rows.
        """
        if not items:
            return 0
        rows: list[tuple[Any, ...]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            mid = str(it.get("mem_id") or "").strip()
            if not mid:
                continue
            rows.append(
                (
                    mid,
                    float(it.get("created_at") or 0.0),
                    float(it.get("updated_at") or 0.0),
                    str(it.get("status") or "active"),
                    str(it.get("role") or "global"),
                    str(it.get("type") or "manual_note"),
                    str(it.get("source_run_id") or "") or None,
                    int(it.get("schema_version") or 1),
                )
            )
        if not rows:
            return 0
        self._conn.executemany(
            """
            INSERT INTO rb_mem_index(
              mem_id, created_at, updated_at, status, role, type, source_run_id, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mem_id) DO UPDATE SET
              created_at = excluded.created_at,
              updated_at = excluded.updated_at,
              status = excluded.status,
              role = excluded.role,
              type = excluded.type,
              source_run_id = excluded.source_run_id,
              schema_version = excluded.schema_version;
            """,
            rows,
        )
        if commit:
            self._conn.commit()
        return len(rows)

    def list_rb_mem_index_page(
        self,
        *,
        limit: int,
        cursor: tuple[float, str] | None,
        role: list[str] | None,
        status: list[str] | None,
        type: list[str] | None,  # noqa: A002
    ) -> dict[str, Any]:
        where = ["1=1"]
        params: list[Any] = []

        if role:
            where.append("role IN (%s)" % ",".join(["?"] * len(role)))
            params.extend([str(r) for r in role])
        if status:
            where.append("status IN (%s)" % ",".join(["?"] * len(status)))
            params.extend([str(s) for s in status])
        if type:
            where.append("type IN (%s)" % ",".join(["?"] * len(type)))
            params.extend([str(t) for t in type])

        if cursor is not None:
            created_at, mem_id = cursor
            where.append("(created_at < ? OR (created_at = ? AND mem_id < ?))")
            params.extend([float(created_at), float(created_at), str(mem_id)])

        where_sql = " AND ".join(where)
        fetch_n = int(limit) + 1

        rows = self._conn.execute(
            f"""
            SELECT mem_id, created_at, updated_at, status, role, type, source_run_id, schema_version
            FROM rb_mem_index
            WHERE {where_sql}
            ORDER BY created_at DESC, mem_id DESC
            LIMIT ?;
            """,
            (*params, fetch_n),
        ).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [
            {
                "mem_id": str(r["mem_id"]),
                "created_at": float(r["created_at"]),
                "updated_at": float(r["updated_at"]),
                "status": str(r["status"]),
                "role": str(r["role"]),
                "type": str(r["type"]),
                "source_run_id": str(r["source_run_id"] or "") or None,
                "schema_version": int(r["schema_version"]),
            }
            for r in rows
        ]

        next_cursor: tuple[float, str] | None = None
        if has_more and items:
            last = items[-1]
            next_cursor = (float(last["created_at"]), str(last["mem_id"]))

        return {"items": items, "has_more": has_more, "next_cursor": next_cursor}

    # --- Products / Presets / Feedback (Milestone 2)
    def get_product(self, *, product_id: str) -> sqlite3.Row | None:
        pid = (product_id or "").strip()
        if not pid:
            return None
        return self._conn.execute(
            """
            SELECT product_id, created_at, updated_at, name, status, schema_version, extra_json
            FROM products
            WHERE product_id = ?
            LIMIT 1;
            """,
            (pid,),
        ).fetchone()

    def list_products_page(
        self,
        *,
        limit: int,
        cursor: tuple[float, str] | None,
        statuses: list[str] | None,
    ) -> dict[str, Any]:
        where = ["1=1"]
        params: list[Any] = []

        if statuses:
            where.append("status IN (%s)" % ",".join(["?"] * len(statuses)))
            params.extend([str(s) for s in statuses])

        if cursor is not None:
            created_at, product_id = cursor
            where.append("(created_at < ? OR (created_at = ? AND product_id < ?))")
            params.extend([float(created_at), float(created_at), str(product_id)])

        where_sql = " AND ".join(where)
        fetch_n = int(limit) + 1

        rows = self._conn.execute(
            f"""
            SELECT product_id, created_at, updated_at, name, status, schema_version, extra_json
            FROM products
            WHERE {where_sql}
            ORDER BY created_at DESC, product_id DESC
            LIMIT ?;
            """,
            (*params, fetch_n),
        ).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [
            {
                "product_id": r["product_id"],
                "created_at": float(r["created_at"]),
                "updated_at": float(r["updated_at"]),
                "name": r["name"],
                "status": r["status"],
                "schema_version": int(r["schema_version"]),
                "extra": json.loads(str(r["extra_json"] or "{}")),
            }
            for r in rows
        ]

        next_cursor: tuple[float, str] | None = None
        if has_more and items:
            last = items[-1]
            next_cursor = (float(last["created_at"]), str(last["product_id"]))

        return {"items": items, "has_more": has_more, "next_cursor": next_cursor}

    def create_product(
        self,
        *,
        name: str,
        status: str = "active",
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> ProductRecord:
        cleaned = (name or "").strip()
        if not cleaned:
            raise ValueError("Product name cannot be empty.")
        if status not in {"active", "archived"}:
            raise ValueError(f"Invalid product status: {status!r}")

        product_id = _new_uuid()
        ts = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO products(
              product_id, created_at, updated_at, name, status, schema_version, extra_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            (product_id, ts, ts, cleaned, status, int(schema_version), _json_dumps(extra or {})),
        )
        if commit:
            self._conn.commit()
        return ProductRecord(product_id=product_id, created_at=ts, updated_at=ts, name=cleaned, status=status)

    def update_product(
        self,
        *,
        product_id: str,
        name: str | None = None,
        status: str | None = None,
        schema_version: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        pid = (product_id or "").strip()
        if not pid:
            raise ValueError("product_id is required.")

        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [_utc_ts()]

        if name is not None:
            cleaned = name.strip()
            if not cleaned:
                raise ValueError("Product name cannot be empty.")
            updates.append("name = ?")
            params.append(cleaned)

        if status is not None:
            if status not in {"active", "archived"}:
                raise ValueError(f"Invalid product status: {status!r}")
            updates.append("status = ?")
            params.append(status)

        if schema_version is not None:
            updates.append("schema_version = ?")
            params.append(int(schema_version))

        if extra is not None:
            updates.append("extra_json = ?")
            params.append(_json_dumps(extra))

        if len(updates) == 1:
            # Nothing to update besides updated_at; keep it as a no-op.
            return

        params.append(pid)
        self._conn.execute(
            f"""
            UPDATE products
            SET {", ".join(updates)}
            WHERE product_id = ?;
            """,
            params,
        )
        self._conn.commit()

    def get_product_preset(self, *, preset_id: str) -> sqlite3.Row | None:
        pid = (preset_id or "").strip()
        if not pid:
            return None
        return self._conn.execute(
            """
            SELECT preset_id, created_at, updated_at, name, status, schema_version, extra_json
            FROM product_presets
            WHERE preset_id = ?
            LIMIT 1;
            """,
            (pid,),
        ).fetchone()

    def get_product_preset_item(self, *, preset_id: str) -> dict[str, Any] | None:
        row = self.get_product_preset(preset_id=preset_id)
        if row is None:
            return None
        pid = str(row["preset_id"])
        product_ids = [
            str(x["product_id"])
            for x in self._conn.execute(
                """
                SELECT product_id
                FROM product_preset_products
                WHERE preset_id = ?
                ORDER BY position ASC, product_id ASC;
                """,
                (pid,),
            ).fetchall()
        ]
        return {
            "preset_id": pid,
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "name": row["name"],
            "status": row["status"],
            "product_ids": product_ids,
            "schema_version": int(row["schema_version"]),
            "extra": json.loads(str(row["extra_json"] or "{}")),
        }

    def list_product_presets_page(
        self,
        *,
        limit: int,
        cursor: tuple[float, str] | None,
        statuses: list[str] | None,
    ) -> dict[str, Any]:
        where = ["1=1"]
        params: list[Any] = []

        if statuses:
            where.append("status IN (%s)" % ",".join(["?"] * len(statuses)))
            params.extend([str(s) for s in statuses])

        if cursor is not None:
            created_at, preset_id = cursor
            where.append("(created_at < ? OR (created_at = ? AND preset_id < ?))")
            params.extend([float(created_at), float(created_at), str(preset_id)])

        where_sql = " AND ".join(where)
        fetch_n = int(limit) + 1

        rows = self._conn.execute(
            f"""
            SELECT preset_id, created_at, updated_at, name, status, schema_version, extra_json
            FROM product_presets
            WHERE {where_sql}
            ORDER BY created_at DESC, preset_id DESC
            LIMIT ?;
            """,
            (*params, fetch_n),
        ).fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items: list[dict[str, Any]] = []
        for r in rows:
            preset_id = str(r["preset_id"])
            product_ids = [
                str(x["product_id"])
                for x in self._conn.execute(
                    """
                    SELECT product_id
                    FROM product_preset_products
                    WHERE preset_id = ?
                    ORDER BY position ASC, product_id ASC;
                    """,
                    (preset_id,),
                ).fetchall()
            ]
            items.append(
                {
                    "preset_id": preset_id,
                    "created_at": float(r["created_at"]),
                    "updated_at": float(r["updated_at"]),
                    "name": r["name"],
                    "status": r["status"],
                    "product_ids": product_ids,
                    "schema_version": int(r["schema_version"]),
                    "extra": json.loads(str(r["extra_json"] or "{}")),
                }
            )

        next_cursor: tuple[float, str] | None = None
        if has_more and items:
            last = items[-1]
            next_cursor = (float(last["created_at"]), str(last["preset_id"]))

        return {"items": items, "has_more": has_more, "next_cursor": next_cursor}

    def _validate_product_ids_exist(self, product_ids: list[str]) -> None:
        unique = [str(p).strip() for p in product_ids if str(p).strip()]
        if not unique:
            return

        rows = self._conn.execute(
            "SELECT product_id FROM products WHERE product_id IN (%s);" % ",".join(["?"] * len(unique)),
            tuple(unique),
        ).fetchall()
        found = {str(r["product_id"]) for r in rows}
        missing = [pid for pid in unique if pid not in found]
        if missing:
            raise ValueError(f"Unknown product_id(s): {missing!r}")

    def create_product_preset(
        self,
        *,
        name: str,
        product_ids: list[str],
        status: str = "active",
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> ProductPresetRecord:
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            raise ValueError("Preset name cannot be empty.")
        if status not in {"active", "archived"}:
            raise ValueError(f"Invalid preset status: {status!r}")

        normalized = [str(pid).strip() for pid in (product_ids or []) if str(pid).strip()]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate product_id in preset is not allowed.")
        self._validate_product_ids_exist(normalized)

        preset_id = _new_uuid()
        ts = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO product_presets(
              preset_id, created_at, updated_at, name, status, schema_version, extra_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """,
            (preset_id, ts, ts, cleaned_name, status, int(schema_version), _json_dumps(extra or {})),
        )
        for idx, pid in enumerate(normalized):
            self._conn.execute(
                """
                INSERT INTO product_preset_products(
                  preset_id, product_id, position, schema_version, extra_json
                ) VALUES(?, ?, ?, ?, ?);
                """,
                (preset_id, pid, int(idx), 1, _json_dumps({})),
            )
        if commit:
            self._conn.commit()
        return ProductPresetRecord(preset_id=preset_id, created_at=ts, updated_at=ts, name=cleaned_name, status=status)

    def update_product_preset(
        self,
        *,
        preset_id: str,
        name: str | None = None,
        product_ids: list[str] | None = None,
        status: str | None = None,
        schema_version: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        pid = (preset_id or "").strip()
        if not pid:
            raise ValueError("preset_id is required.")

        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [_utc_ts()]

        if name is not None:
            cleaned = name.strip()
            if not cleaned:
                raise ValueError("Preset name cannot be empty.")
            updates.append("name = ?")
            params.append(cleaned)

        if status is not None:
            if status not in {"active", "archived"}:
                raise ValueError(f"Invalid preset status: {status!r}")
            updates.append("status = ?")
            params.append(status)

        if schema_version is not None:
            updates.append("schema_version = ?")
            params.append(int(schema_version))

        if extra is not None:
            updates.append("extra_json = ?")
            params.append(_json_dumps(extra))

        self._conn.execute(
            f"""
            UPDATE product_presets
            SET {", ".join(updates)}
            WHERE preset_id = ?;
            """,
            (*params, pid),
        )

        if product_ids is not None:
            normalized = [str(x).strip() for x in product_ids if str(x).strip()]
            if len(set(normalized)) != len(normalized):
                raise ValueError("Duplicate product_id in preset is not allowed.")
            self._validate_product_ids_exist(normalized)
            self._conn.execute("DELETE FROM product_preset_products WHERE preset_id = ?;", (pid,))
            for idx, product_id in enumerate(normalized):
                self._conn.execute(
                    """
                    INSERT INTO product_preset_products(
                      preset_id, product_id, position, schema_version, extra_json
                    ) VALUES(?, ?, ?, ?, ?);
                    """,
                    (pid, product_id, int(idx), 1, _json_dumps({})),
                )

        self._conn.commit()

    def get_feedback_for_run(self, *, run_id: str) -> dict[str, Any] | None:
        rid = (run_id or "").strip()
        if not rid:
            return None

        row = self._conn.execute(
            """
            SELECT feedback_id, run_id, created_at, updated_at, score, pros, cons, other, schema_version, extra_json
            FROM feedback
            WHERE run_id = ?
            LIMIT 1;
            """,
            (rid,),
        ).fetchone()
        if row is None:
            return None

        feedback_id = str(row["feedback_id"])
        products_rows = self._conn.execute(
            """
            SELECT
              fp.feedback_product_id,
              fp.product_id,
              p.name AS product_name,
              p.status AS product_status,
              fp.value,
              fp.fraction
            FROM feedback_products fp
            JOIN products p ON p.product_id = fp.product_id
            WHERE fp.feedback_id = ?
            ORDER BY p.name ASC, fp.product_id ASC;
            """,
            (feedback_id,),
        ).fetchall()

        return {
            "feedback": {
                "feedback_id": feedback_id,
                "run_id": str(row["run_id"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
                "score": float(row["score"]) if row["score"] is not None else None,
                "pros": str(row["pros"] or ""),
                "cons": str(row["cons"] or ""),
                "other": str(row["other"] or ""),
                "schema_version": int(row["schema_version"]),
                "extra": json.loads(str(row["extra_json"] or "{}")),
                "products": [
                    {
                        "feedback_product_id": r["feedback_product_id"],
                        "product_id": r["product_id"],
                        "product_name": r["product_name"],
                        "product_status": r["product_status"],
                        "value": float(r["value"]),
                        "fraction": float(r["fraction"]),
                    }
                    for r in products_rows
                ],
            }
        }

    def upsert_feedback_for_run(
        self,
        *,
        run_id: str,
        score: float | None,
        pros: str,
        cons: str,
        other: str,
        products: list[dict[str, Any]],
        schema_version: int = 1,
        extra: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        rid = (run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required.")
        if self.get_run(run_id=rid) is None:
            raise ValueError("Run not found.")

        normalized_products: list[tuple[str, float]] = []
        for item in products or []:
            if not isinstance(item, dict):
                raise ValueError("products must be a list of objects.")
            pid = str(item.get("product_id") or "").strip()
            if not pid:
                raise ValueError("product_id is required for every product row.")
            val = item.get("value")
            if not isinstance(val, (int, float)) or not (val == val) or not (abs(float(val)) < 1e308):
                raise ValueError("value must be a finite number.")
            fval = float(val)
            if fval < 0:
                raise ValueError("value must be >= 0.")
            normalized_products.append((pid, fval))

        product_ids = [pid for pid, _ in normalized_products]
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("Duplicate product_id in feedback is not allowed.")
        self._validate_product_ids_exist(product_ids)

        total = sum(v for _, v in normalized_products)
        fractions: dict[str, float] = {}
        if total > 0:
            for pid, v in normalized_products:
                fractions[pid] = float(v) / float(total)
        else:
            for pid, _v in normalized_products:
                fractions[pid] = 0.0

        ts = _utc_ts()

        with self.transaction(mode="IMMEDIATE"):
            existing = self._conn.execute(
                "SELECT feedback_id, created_at FROM feedback WHERE run_id = ? LIMIT 1;",
                (rid,),
            ).fetchone()
            if existing is None:
                feedback_id = _new_id("feedback")
                created_at = ts
                self._conn.execute(
                    """
                    INSERT INTO feedback(
                      feedback_id, run_id, created_at, updated_at, score, pros, cons, other,
                      schema_version, extra_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        feedback_id,
                        rid,
                        created_at,
                        ts,
                        float(score) if score is not None else None,
                        str(pros or ""),
                        str(cons or ""),
                        str(other or ""),
                        int(schema_version),
                        _json_dumps(extra or {}),
                    ),
                )
            else:
                feedback_id = str(existing["feedback_id"])
                created_at = float(existing["created_at"])
                self._conn.execute(
                    """
                    UPDATE feedback
                    SET
                      updated_at = ?,
                      score = ?,
                      pros = ?,
                      cons = ?,
                      other = ?,
                      schema_version = ?,
                      extra_json = ?
                    WHERE run_id = ?;
                    """,
                    (
                        ts,
                        float(score) if score is not None else None,
                        str(pros or ""),
                        str(cons or ""),
                        str(other or ""),
                        int(schema_version),
                        _json_dumps(extra or {}),
                        rid,
                    ),
                )

            self._conn.execute("DELETE FROM feedback_products WHERE feedback_id = ?;", (feedback_id,))

            for pid, value in normalized_products:
                self._conn.execute(
                    """
                    INSERT INTO feedback_products(
                      feedback_product_id, feedback_id, product_id, created_at, updated_at,
                      value, fraction, schema_version, extra_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        _new_id("fbp"),
                        feedback_id,
                        pid,
                        ts,
                        ts,
                        float(value),
                        float(fractions[pid]),
                        1,
                        _json_dumps({}),
                    ),
                )

        return FeedbackRecord(
            feedback_id=feedback_id,
            run_id=rid,
            created_at=float(created_at),
            updated_at=float(ts),
            score=float(score) if score is not None else None,
            pros=str(pros or ""),
            cons=str(cons or ""),
            other=str(other or ""),
        )

    # --- Cancellation (program-recorded; used by UI/backend to stop runs)
    def request_cancel(self, *, target_type: str, target_id: str, reason: str | None = None) -> str:
        if target_type not in {"batch", "run"}:
            raise ValueError(f"Invalid target_type: {target_type!r}")

        cancel_id = _new_id("cancel")
        created_at = _utc_ts()
        self._conn.execute(
            """
            INSERT INTO cancel_requests(
              cancel_id, created_at, target_type, target_id, status, reason
            ) VALUES(?, ?, ?, ?, ?, ?);
            """,
            (cancel_id, created_at, target_type, target_id, "requested", reason),
        )
        self._conn.commit()
        return cancel_id

    def is_cancel_requested(self, *, target_type: str, target_id: str) -> bool:
        if target_type not in {"batch", "run"}:
            raise ValueError(f"Invalid target_type: {target_type!r}")
        row = self._conn.execute(
            """
            SELECT 1 FROM cancel_requests
            WHERE target_type = ? AND target_id = ? AND status IN ('requested', 'acknowledged')
            LIMIT 1;
            """,
            (target_type, target_id),
        ).fetchone()
        return row is not None

    def acknowledge_cancel(self, *, target_type: str, target_id: str) -> None:
        if target_type not in {"batch", "run"}:
            raise ValueError(f"Invalid target_type: {target_type!r}")
        self._conn.execute(
            """
            UPDATE cancel_requests
            SET status = 'acknowledged'
            WHERE target_type = ? AND target_id = ? AND status = 'requested';
            """,
            (target_type, target_id),
        )
        self._conn.commit()
