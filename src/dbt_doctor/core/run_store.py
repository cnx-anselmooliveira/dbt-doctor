"""Core: DuckDB store that keeps what every dbt-doctor run did, linked by ``run_id``.

A *run* is one campaign (for example "document the lineage of model X"). Every tool of the
documentation flow records into the same run: the batches it served, the scores it received,
the plans it made and the descriptions it wrote. Texts are stored once by hash; scores are
global (a score is valid for a description text and a rubric, whichever run produced it).

DuckDB allows a single writing process, so each operation opens a short connection and waits
a few seconds for the lock. Callers must treat a ``StoreUnavailable`` as non-fatal.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from ..analyzers.score_store import description_hash

SCHEMA_VERSION = 1
DEFAULT_RELATIVE_PATH = Path(".dbt-doctor") / "dbt_doctor.duckdb"


class StoreUnavailable(RuntimeError):
    """The database cannot be opened (locked, unreadable, or from a newer version)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)",
    """CREATE TABLE IF NOT EXISTS manifests (
        manifest_id VARCHAR PRIMARY KEY, path VARCHAR, generated_at VARCHAR,
        invocation_id VARCHAR, project_name VARCHAR, file_mtime DOUBLE, first_seen_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY, started_at TIMESTAMP, ended_at TIMESTAMP, status VARCHAR,
        goal VARCHAR, target_model VARCHAR, manifest_id VARCHAR, rubric_fp VARCHAR,
        params_json VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS events (
        event_id VARCHAR PRIMARY KEY, run_id VARCHAR, seq INTEGER, ts TIMESTAMP, tool VARCHAR,
        model VARCHAR, ok BOOLEAN, manifest_id VARCHAR, summary_json VARCHAR)""",
    "CREATE TABLE IF NOT EXISTS descriptions (hash VARCHAR PRIMARY KEY, text VARCHAR)",
    """CREATE TABLE IF NOT EXISTS snapshots (
        run_id VARCHAR, model VARCHAR, column_name VARCHAR, hash VARCHAR, event_id VARCHAR,
        PRIMARY KEY (run_id, model, column_name))""",
    """CREATE TABLE IF NOT EXISTS scores (
        score_id VARCHAR PRIMARY KEY, run_id VARCHAR, event_id VARCHAR, model VARCHAR,
        column_name VARCHAR, hash VARCHAR, rubric_fp VARCHAR, total DOUBLE, status VARCHAR,
        per_question_json VARCHAR, scored_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS plan_items (
        run_id VARCHAR, event_id VARCHAR, model VARCHAR, column_name VARCHAR, rule VARCHAR,
        action VARCHAR, verdict VARCHAR, reason VARCHAR, parent_model VARCHAR,
        parent_column VARCHAR, parent_status VARCHAR, ts TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS writes (
        write_id VARCHAR PRIMARY KEY, run_id VARCHAR, event_id VARCHAR, model VARCHAR,
        scope VARCHAR, column_name VARCHAR, file_path VARCHAR, action VARCHAR, via VARCHAR,
        reason VARCHAR, old_hash VARCHAR, new_hash VARCHAR, dry_run BOOLEAN, ts TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS lineage_edges (
        run_id VARCHAR, model VARCHAR, column_name VARCHAR, parent_model VARCHAR,
        parent_column VARCHAR, kind VARCHAR, effects VARCHAR, confidence VARCHAR,
        evidence VARCHAR, PRIMARY KEY (run_id, model, column_name))""",
    """CREATE OR REPLACE VIEW v_latest_scores AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY model, lower(column_name), hash, rubric_fp ORDER BY scored_at DESC
            ) AS rn FROM scores
        ) WHERE rn = 1""",
]


class RunStore:
    def __init__(self, path: Path, lock_timeout: float = 3.0) -> None:
        self.path = Path(path)
        self.lock_timeout = lock_timeout
        self._ready = False
        self._lock = threading.RLock()
        self._conn: duckdb.DuckDBPyConnection | None = None  # set while a session is open

    # ── connection ────────────────────────────────────────────────────────

    def _open(self) -> duckdb.DuckDBPyConnection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            ignore = self.path.parent / ".gitignore"
            if not ignore.exists():  # keep the history out of the project's git
                ignore.write_text("*\n", encoding="utf-8")
        except OSError as exc:
            raise StoreUnavailable(f"cannot create {self.path.parent}: {exc}") from exc
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                return duckdb.connect(str(self.path))
            except duckdb.IOException as exc:
                if "lock" in str(exc).lower() and time.monotonic() < deadline:
                    time.sleep(0.1)
                    continue
                raise StoreUnavailable(f"cannot open {self.path.name}: {exc}") from exc
            except duckdb.Error as exc:
                raise StoreUnavailable(f"cannot open {self.path.name}: {exc}") from exc

    @contextmanager
    def connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """A connection: the open session's, or a short one closed on exit."""
        with self._lock:
            if self._conn is not None:
                yield self._conn
                return
            conn = self._open()
            try:
                if not self._ready:
                    self._ensure_schema(conn)
                    self._ready = True
                yield conn
            finally:
                conn.close()

    @contextmanager
    def session(self) -> Iterator[None]:
        """Keep one connection open for a group of operations (one tool call).

        Opening a DuckDB file costs more than the inserts themselves, so a tool call opens it
        once. The write lock is held only while the call runs.
        """
        with self._lock:
            if self._conn is not None:
                yield
                return
            conn = self._open()
            try:
                if not self._ready:
                    self._ensure_schema(conn)
                    self._ready = True
                self._conn = conn
                yield
            finally:
                self._conn = None
                conn.close()

    def _ensure_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        for statement in _SCHEMA:
            conn.execute(statement)
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", [str(SCHEMA_VERSION)])
        elif int(row[0]) > SCHEMA_VERSION:
            raise StoreUnavailable(
                f"{self.path.name} was created by a newer dbt-doctor (schema {row[0]})."
            )
        # Future versions: apply migrations from int(row[0]) up to SCHEMA_VERSION here.

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connection() as conn:
            cur = conn.execute(sql, list(params))
            columns = [d[0] for d in cur.description or []]
            return [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]

    # ── manifests and runs ────────────────────────────────────────────────

    def register_manifest(self, info: dict[str, Any]) -> str:
        key = "|".join(
            str(info.get(k, "")) for k in ("path", "generated_at", "invocation_id", "mtime")
        )
        manifest_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO manifests VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [
                    manifest_id,
                    info.get("path"),
                    info.get("generated_at"),
                    info.get("invocation_id"),
                    info.get("project_name"),
                    info.get("mtime"),
                    utcnow(),
                ],
            )
        return manifest_id

    def create_run(
        self,
        goal: str,
        target_model: str,
        manifest_id: str | None,
        rubric_fp: str,
        params: dict[str, Any],
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or f"run_{utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, NULL, 'open', ?, ?, ?, ?, ?)",
                [run_id, utcnow(), goal, target_model, manifest_id, rubric_fp, json.dumps(params)],
            )
        return run_id

    def close_open_runs(self, except_run: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE runs SET status = 'closed', ended_at = ? "
                "WHERE status = 'open' AND run_id <> ?",
                [utcnow(), except_run or ""],
            )

    def close_run(self, run_id: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE runs SET status = 'closed', ended_at = ? WHERE run_id = ?",
                [utcnow(), run_id],
            )

    def import_hashed_scores(self, run_id: str, by_model: dict[str, list[dict[str, Any]]]) -> int:
        """Insert scores that already carry a description hash (the legacy JSON store)."""
        now = utcnow()
        rows = [
            (
                uuid.uuid4().hex,
                run_id,
                None,
                model,
                e["column"],
                e["hash"],
                e.get("rubric"),
                e.get("total"),
                e.get("status"),
                None,
                now,
            )
            for model, entries in by_model.items()
            for e in entries
            if e.get("hash")
        ]
        if rows:
            with self.connection() as conn:
                conn.executemany(
                    "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
        return len(rows)

    def latest_open_run(self, target_model: str = "") -> dict[str, Any] | None:
        sql = "SELECT * FROM runs WHERE status = 'open'"
        params: list[Any] = []
        if target_model:
            sql += " AND target_model = ?"
            params.append(target_model)
        rows = self.query(sql + " ORDER BY started_at DESC LIMIT 1", params)
        return rows[0] if rows else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", [run_id])
        return rows[0] if rows else None

    def reopen_run(self, run_id: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE runs SET status = 'open', ended_at = NULL WHERE run_id = ?", [run_id]
            )

    # ── events and artifacts ──────────────────────────────────────────────

    def add_event(
        self,
        event_id: str,
        run_id: str,
        tool: str,
        model: str | None,
        ok: bool,
        manifest_id: str | None,
        summary: dict[str, Any],
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO events (event_id, run_id, seq, ts, tool, model, ok, manifest_id, "
                "summary_json) SELECT ?, ?, coalesce(max(seq), 0) + 1, ?, ?, ?, ?, ?, ? "
                "FROM events WHERE run_id = ?",
                [
                    event_id,
                    run_id,
                    utcnow(),
                    tool,
                    model,
                    ok,
                    manifest_id,
                    json.dumps(summary, default=str, ensure_ascii=False),
                    run_id,
                ],
            )

    def add_snapshots(
        self, run_id: str, event_id: str | None, model: str, items: list[tuple[str, str]]
    ) -> None:
        """Record each column's description at first contact with the model in this run."""
        if not items:
            return
        with self.connection() as conn:
            conn.executemany(
                "INSERT INTO descriptions VALUES (?, ?) ON CONFLICT DO NOTHING",
                [(description_hash(t), (t or "").strip()) for _, t in items],
            )
            conn.executemany(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [(run_id, model, c, description_hash(t), event_id) for c, t in items],
            )

    def add_scores(
        self,
        run_id: str,
        event_id: str | None,
        model: str,
        rubric_fp: str,
        items: list[dict[str, Any]],
    ) -> None:
        if not items:
            return
        now = utcnow()
        with self.connection() as conn:
            conn.executemany(
                "INSERT INTO descriptions VALUES (?, ?) ON CONFLICT DO NOTHING",
                [
                    (description_hash(i["description"]), (i["description"] or "").strip())
                    for i in items
                ],
            )
            conn.executemany(
                "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        uuid.uuid4().hex,
                        run_id,
                        event_id,
                        model,
                        i["column"],
                        description_hash(i["description"]),
                        rubric_fp,
                        i["total"],
                        i["status"],
                        json.dumps(i.get("scores")) if i.get("scores") is not None else None,
                        i.get("scored_at") or now,
                    )
                    for i in items
                ],
            )

    def latest_score(
        self, model: str, column: str, description: str, rubric_fp: str
    ) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT hash, rubric_fp AS rubric, total, status, scored_at FROM scores "
            "WHERE model = ? AND lower(column_name) = lower(?) AND hash = ? AND rubric_fp = ? "
            "ORDER BY scored_at DESC LIMIT 1",
            [model, column, description_hash(description), rubric_fp],
        )
        return rows[0] if rows else None

    def add_plan_items(
        self, run_id: str, event_id: str | None, model: str, entries: list[dict[str, Any]]
    ) -> None:
        if not entries:
            return
        now = utcnow()
        with self.connection() as conn:
            conn.executemany(
                "INSERT INTO plan_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        event_id,
                        model,
                        e["column"],
                        e.get("rule"),
                        e.get("action"),
                        e.get("verdict"),
                        e.get("reason"),
                        (e.get("lineage") or {}).get("parent"),
                        (e.get("lineage") or {}).get("parent_column"),
                        e.get("parent_status"),
                        now,
                    )
                    for e in entries
                ],
            )

    def add_writes(self, run_id: str, event_id: str | None, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        now = utcnow()
        with self.connection() as conn:
            conn.executemany(
                "INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        uuid.uuid4().hex,
                        run_id,
                        event_id,
                        r["model"],
                        r.get("scope", "column"),
                        r.get("column"),
                        r.get("file_path"),
                        r["action"],
                        r.get("via"),
                        r.get("reason"),
                        description_hash(r["old"]) if r.get("old") else None,
                        description_hash(r["new"]) if r.get("new") else None,
                        bool(r.get("dry_run")),
                        now,
                    )
                    for r in rows
                ],
            )
            texts = [
                (description_hash(t), t.strip())
                for r in rows
                for t in (r.get("old"), r.get("new"))
                if t
            ]
            conn.executemany("INSERT INTO descriptions VALUES (?, ?) ON CONFLICT DO NOTHING", texts)

    def add_lineage_edges(self, run_id: str, model: str, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return
        with self.connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO lineage_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        model,
                        e["column"],
                        e.get("parent"),
                        e.get("parent_column"),
                        e.get("kind"),
                        json.dumps(e.get("effects") or []),
                        e.get("confidence"),
                        e.get("evidence"),
                    )
                    for e in edges
                ],
            )

    # ── reports ───────────────────────────────────────────────────────────

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query(
            "SELECT r.run_id, r.started_at, r.ended_at, r.status, r.goal, r.target_model, "
            "(SELECT count(*) FROM events e WHERE e.run_id = r.run_id) AS events, "
            "(SELECT count(DISTINCT model) FROM events e "
            " WHERE e.run_id = r.run_id AND model IS NOT NULL) AS models "
            "FROM runs r ORDER BY r.started_at DESC LIMIT ?",
            [max(1, limit)],
        )

    def run_report(self, run_id: str, timeline: int = 15) -> dict[str, Any] | None:
        """Everything one run did, summarised per model."""
        run = self.get_run(run_id)
        if run is None:
            return None
        run["params"] = json.loads(run.pop("params_json") or "{}")
        events = self.query(
            "SELECT tool, count(*) AS n FROM events WHERE run_id = ? GROUP BY tool ORDER BY n DESC",
            [run_id],
        )
        last_events = self.query(
            "SELECT seq, ts, tool, model, ok, summary_json FROM events WHERE run_id = ? "
            "ORDER BY seq DESC LIMIT ?",
            [run_id, timeline],
        )
        for e in last_events:
            e["summary"] = json.loads(e.pop("summary_json") or "{}")

        models: dict[str, dict[str, Any]] = {}

        def entry(model: str) -> dict[str, Any]:
            return models.setdefault(model, {"model": model})

        for row in self.query(
            "SELECT model, count(*) AS n FROM snapshots WHERE run_id = ? GROUP BY model", [run_id]
        ):
            entry(row["model"])["columns_seen"] = row["n"]
        for row in self.query(
            "SELECT model, status, count(*) AS n FROM ("
            " SELECT *, row_number() OVER (PARTITION BY model, lower(column_name) "
            "  ORDER BY scored_at DESC) AS rn FROM scores WHERE run_id = ?) WHERE rn = 1 "
            "GROUP BY model, status",
            [run_id],
        ):
            entry(row["model"]).setdefault("scores", {})[row["status"]] = row["n"]
        for row in self.query(
            "SELECT model, action, count(*) AS n FROM ("
            " SELECT *, row_number() OVER (PARTITION BY model, lower(column_name) "
            "  ORDER BY ts DESC) AS rn FROM plan_items WHERE run_id = ?) WHERE rn = 1 "
            "GROUP BY model, action",
            [run_id],
        ):
            entry(row["model"]).setdefault("plan", {})[row["action"]] = row["n"]
        for row in self.query(
            "SELECT model, action, dry_run, count(*) AS n FROM writes WHERE run_id = ? "
            "GROUP BY model, action, dry_run",
            [run_id],
        ):
            key = f"{row['action']}{' (dry run)' if row['dry_run'] else ''}"
            entry(row["model"]).setdefault("writes", {})[key] = row["n"]

        blocked = self.query(
            "SELECT reason, count(*) AS n FROM ("
            " SELECT *, row_number() OVER (PARTITION BY model, lower(column_name) "
            "  ORDER BY ts DESC) AS rn FROM plan_items WHERE run_id = ?) "
            "WHERE rn = 1 AND action = 'blocked' GROUP BY reason ORDER BY n DESC LIMIT 5",
            [run_id],
        )
        return {
            "run": run,
            "events_by_tool": events,
            "models": sorted(models.values(), key=lambda m: m["model"]),
            "top_blocked_reasons": blocked,
            "latest_events": last_events,
        }
