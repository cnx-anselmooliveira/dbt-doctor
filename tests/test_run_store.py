"""Tests for the DuckDB run store and the recorder that feeds it."""

import subprocess
import sys

import pytest

from dbt_doctor.analyzers.score_store import ScoreStore
from dbt_doctor.core.run_recorder import RunRecorder
from dbt_doctor.core.run_store import SCHEMA_VERSION, RunStore, StoreUnavailable

MANIFEST = {"path": "manifest.json", "mtime": 1.0, "generated_at": "t1", "invocation_id": "i1"}


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / ".dbt-doctor" / "dbt_doctor.duckdb", lock_timeout=0.3)


def make_recorder(store, manifest=None):
    state = {"manifest": manifest or MANIFEST}
    recorder = RunRecorder(
        store, lambda: {"manifest": state["manifest"], "rubric_fp": "fp", "params": {"x": 1}}
    )
    recorder.state = state  # lets a test change the manifest the context reports
    return recorder


# ── store ─────────────────────────────────────────────────────────────────


def test_creates_schema_version_and_a_private_gitignore(store):
    store.query("SELECT 1")
    assert store.query("SELECT value FROM meta WHERE key = 'schema_version'") == [
        {"value": str(SCHEMA_VERSION)}
    ]
    assert (store.path.parent / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert RunStore(store.path).query("SELECT count(*) AS n FROM runs") == [{"n": 0}]  # reopens


def test_a_database_from_a_newer_version_is_refused(store):
    store.query("SELECT 1")
    with store.connection() as conn:
        conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    with pytest.raises(StoreUnavailable, match="newer"):
        RunStore(store.path).query("SELECT 1")


def test_manifest_registration_is_idempotent(store):
    assert store.register_manifest(MANIFEST) == store.register_manifest(MANIFEST)
    assert store.register_manifest({**MANIFEST, "mtime": 2.0}) != store.register_manifest(MANIFEST)
    assert store.query("SELECT count(*) AS n FROM manifests") == [{"n": 2}]


def test_runs_open_close_and_latest_open(store):
    first = store.create_run("g1", "m1", None, "fp", {})
    second = store.create_run("g2", "m2", None, "fp", {})
    assert store.latest_open_run()["run_id"] == second
    assert store.latest_open_run("m1")["run_id"] == first
    store.close_open_runs(except_run=second)
    assert store.get_run(first)["status"] == "closed" and store.get_run(second)["status"] == "open"
    store.close_run(second)
    assert store.latest_open_run() is None


def test_event_sequence_is_per_run(store):
    a, b = store.create_run("a", "", None, "fp", {}), store.create_run("b", "", None, "fp", {})
    for run, event in ((a, "e1"), (a, "e2"), (b, "e3")):
        store.add_event(event, run, "tool", "m", True, None, {"k": 1})
    seqs = {r["event_id"]: r["seq"] for r in store.query("SELECT event_id, seq FROM events")}
    assert seqs == {"e1": 1, "e2": 2, "e3": 1}


def test_snapshots_keep_the_first_description_seen_in_a_run(store):
    run = store.create_run("g", "", None, "fp", {})
    store.add_snapshots(run, None, "m", [("a", "before"), ("b", "")])
    store.add_snapshots(run, None, "m", [("a", "after")])
    rows = store.query(
        "SELECT s.column_name, d.text FROM snapshots s JOIN descriptions d USING (hash) ORDER BY 1"
    )
    assert rows == [{"column_name": "a", "text": "before"}, {"column_name": "b", "text": ""}]


def test_latest_score_matches_text_rubric_and_ignores_case_and_whitespace(store):
    run = store.create_run("g", "", None, "fp", {})
    item = {"column": "Col_A", "description": "Some  text", "total": 40, "status": "ok"}
    store.add_scores(run, None, "m", "fp", [item])
    assert store.latest_score("m", "col_a", "Some text", "fp")["status"] == "ok"
    assert store.latest_score("m", "col_a", "Other text", "fp") is None
    assert store.latest_score("m", "col_a", "Some text", "other-fp") is None
    assert store.latest_score("other", "col_a", "Some text", "fp") is None


def test_the_newest_score_wins_and_history_is_kept(store):
    run = store.create_run("g", "", None, "fp", {})
    store.add_scores(
        run,
        None,
        "m",
        "fp",
        [{"column": "a", "description": "t", "total": 10, "status": "rewrite"}],
    )
    store.add_scores(
        run, None, "m", "fp", [{"column": "a", "description": "t", "total": 42, "status": "ok"}]
    )
    assert store.latest_score("m", "a", "t", "fp")["total"] == 42
    assert store.query("SELECT count(*) AS n FROM scores") == [{"n": 2}]
    assert store.query("SELECT status FROM v_latest_scores") == [{"status": "ok"}]


def test_run_report_summarises_every_artifact(store):
    run = store.create_run("goal", "m", None, "fp", {"flag": True})
    store.add_event("e1", run, "evaluate_model_descriptions", "m", True, None, {"returned": 2})
    store.add_snapshots(run, "e1", "m", [("a", "x"), ("b", "")])
    store.add_scores(
        run, "e1", "m", "fp", [{"column": "a", "description": "x", "total": 40, "status": "ok"}]
    )
    store.add_plan_items(
        run,
        "e1",
        "child",
        [
            {"column": "a", "action": "copy", "verdict": "inconsistent", "reason": "r"},
            {"column": "b", "action": "blocked", "reason": "parent not scored"},
        ],
    )
    store.add_writes(
        run,
        "e1",
        [
            {
                "model": "child",
                "column": "a",
                "action": "changed",
                "old": "o",
                "new": "n",
                "dry_run": False,
            },
            {"model": "child", "column": "b", "action": "skipped", "reason": "x", "dry_run": True},
        ],
    )
    report = store.run_report(run)
    models = {m["model"]: m for m in report["models"]}
    assert report["run"]["params"] == {"flag": True}
    assert report["events_by_tool"] == [{"tool": "evaluate_model_descriptions", "n": 1}]
    assert models["m"]["columns_seen"] == 2 and models["m"]["scores"] == {"ok": 1}
    assert models["child"]["plan"] == {"copy": 1, "blocked": 1}
    assert models["child"]["writes"] == {"changed": 1, "skipped (dry run)": 1}
    assert report["top_blocked_reasons"] == [{"reason": "parent not scored", "n": 1}]
    assert report["latest_events"][0]["summary"] == {"returned": 2}
    assert store.run_report("ghost") is None
    assert store.list_runs()[0]["events"] == 1


def test_lineage_edges_are_replaced_not_duplicated(store):
    run = store.create_run("g", "", None, "fp", {})
    edge = {
        "column": "a",
        "parent": "p",
        "parent_column": "x",
        "kind": "renamed",
        "effects": ["where"],
    }
    store.add_lineage_edges(run, "m", [edge])
    store.add_lineage_edges(run, "m", [{**edge, "kind": "passthrough"}])
    assert store.query("SELECT kind FROM lineage_edges") == [{"kind": "passthrough"}]


def test_a_locked_database_raises_store_unavailable(store):
    store.query("SELECT 1")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import duckdb, sys, time\n"
            f"c = duckdb.connect({str(store.path)!r})\n"
            "print('ready', flush=True)\n"
            "time.sleep(4)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(StoreUnavailable):
            RunStore(store.path, lock_timeout=0.3).query("SELECT 1")
    finally:
        holder.kill()
        holder.wait()


# ── recorder ──────────────────────────────────────────────────────────────


def test_first_event_opens_an_implicit_run_and_events_are_ordered(store):
    recorder = make_recorder(store)
    with recorder.event("t1", "m") as ev:
        ev.summary["a"] = 1
    with recorder.event("t2", "m"):
        pass
    runs = store.query("SELECT goal, status, target_model, manifest_id FROM runs")
    assert (
        len(runs) == 1
        and runs[0]["goal"].startswith("(implicit)")
        and runs[0]["target_model"] == "m"
    )
    assert runs[0]["manifest_id"]
    assert [e["tool"] for e in store.query("SELECT tool FROM events ORDER BY seq")] == ["t1", "t2"]
    assert recorder.info() == {"run_id": recorder.run_id, "recording": "ok"}


def test_a_failing_tool_is_recorded_as_not_ok_and_the_error_propagates(store):
    recorder = make_recorder(store)
    with pytest.raises(ValueError), recorder.event("t", "m"):
        raise ValueError("boom")
    assert store.query("SELECT ok FROM events") == [{"ok": False}]


def test_artifacts_recorded_inside_an_event_are_linked_to_it(store):
    recorder = make_recorder(store)
    with recorder.event("submit", "m") as ev:
        recorder.record_many(
            "m", "fp", [{"column": "a", "description": "t", "total": 40, "status": "ok"}]
        )
        recorder.record_snapshot("m", {"a": "t"})
        event_id = ev.event_id
    assert store.query("SELECT event_id FROM scores") == [{"event_id": event_id}]
    assert store.query("SELECT event_id FROM snapshots") == [{"event_id": event_id}]
    assert recorder.lookup("m", "a", "t", "fp")["status"] == "ok"  # works as a score store


def test_explicit_start_run_records_the_context_and_closes_the_previous_one(store):
    recorder = make_recorder(store)
    first = recorder.start_run("first goal", "model_a")["run_id"]
    second = recorder.start_run("second goal", "model_b")
    assert second["run_id"] != first and second["resumed"] is False
    assert store.get_run(first)["status"] == "closed"
    run = store.get_run(second["run_id"])
    assert (run["goal"], run["target_model"], run["rubric_fp"]) == ("second goal", "model_b", "fp")


def test_resume_continues_the_latest_open_run_and_flags_a_changed_manifest(store):
    first = make_recorder(store)
    run_id = first.start_run("campaign", "model_a")["run_id"]

    resumed = make_recorder(store)  # a new server process
    assert resumed.start_run(target_model="model_a", resume=True) == {
        "resumed": True,
        "manifest_changed": False,
        "run_id": run_id,
        "recording": "ok",
    }
    resumed.state["manifest"] = {**MANIFEST, "mtime": 2.0}  # dbt parse ran in between
    again = make_recorder(store)
    again.state["manifest"] = {**MANIFEST, "mtime": 2.0}
    assert again.start_run(resume=True)["manifest_changed"] is True
    assert make_recorder(store).start_run(target_model="other", resume=True)["resumed"] is False


def test_recording_failures_never_break_the_caller(store, monkeypatch):
    recorder = make_recorder(store)

    def locked():
        raise StoreUnavailable("database is locked")

    monkeypatch.setattr(store, "_open", locked)
    with recorder.event("t", "m") as ev:  # no exception
        ev.summary["x"] = 1
        recorder.record_many(
            "m", "fp", [{"column": "a", "description": "t", "total": 1, "status": "ok"}]
        )
        recorder.record_snapshot("m", {"a": "t"})
    assert recorder.lookup("m", "a", "t", "fp") is None
    assert recorder.run_id is None
    assert recorder.info()["recording"] == "unavailable: database is locked"
    assert recorder.start_run("g")["error"] == "database is locked"


def test_a_database_locked_by_another_process_degrades_gracefully(store):
    store.query("SELECT 1")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import duckdb, time\n"
            f"c = duckdb.connect({str(store.path)!r})\n"
            "print('ready', flush=True)\n"
            "time.sleep(4)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        recorder = make_recorder(RunStore(store.path, lock_timeout=0.3))
        with recorder.event("t", "m"):
            pass
        assert recorder.info()["recording"].startswith("unavailable")
    finally:
        holder.kill()
        holder.wait()


# ── migration of the JSON score store ─────────────────────────────────────


def test_legacy_scores_are_imported_once_and_the_file_is_renamed(store, tmp_path):
    legacy = tmp_path / "target" / "dbt_doctor" / "description_scores.json"
    ScoreStore(legacy).record("raw_orders", "id", "Order identifier.", "fp", 40, "ok")
    ScoreStore(legacy).record("raw_orders", "status", "Order status.", "fp", 10, "rewrite")

    recorder = make_recorder(store)
    assert recorder.import_legacy_scores(legacy) == 2
    assert not legacy.exists() and legacy.with_suffix(".json.migrated").exists()
    assert recorder.lookup("raw_orders", "id", "Order identifier.", "fp")["status"] == "ok"
    assert recorder.lookup("raw_orders", "status", "Order status.", "fp")["status"] == "rewrite"
    assert recorder.import_legacy_scores(legacy) == 0  # nothing left to import
    assert store.query("SELECT status FROM runs WHERE goal LIKE 'import%'") == [
        {"status": "closed"}
    ]


# ── circuit breaker and score cache ───────────────────────────────────────


def test_after_a_failure_the_store_is_skipped_until_the_cooldown_ends(store, monkeypatch):
    recorder = make_recorder(store)
    attempts = []

    def locked():
        attempts.append(1)
        raise StoreUnavailable("database is locked")

    monkeypatch.setattr(store, "_open", locked)
    with recorder.event("t", "m"):
        for _ in range(20):  # a tool call makes many recordings
            recorder.record_snapshot("m", {"a": "t"})
            recorder.lookup("m", "a", "t", "fp")
    assert len(attempts) == 1  # one wait for the lock, not one per recording

    monkeypatch.undo()  # the database is back, and so is the cooldown
    recorder.cooldown = 0.0
    recorder._down_until = 0.0
    with recorder.event("t2", "m"):
        pass
    assert recorder.info()["recording"] == "ok"
    assert store.query("SELECT tool FROM events") == [{"tool": "t2"}]


def test_scores_are_read_once_per_tool_call_and_refreshed_when_new_ones_arrive(store, monkeypatch):
    recorder = make_recorder(store)
    item = {"column": "a", "description": "text", "total": 40, "status": "ok"}
    recorder.record_many("m", "fp", [item])
    reads = []
    original = store.query

    def counting(sql, params=()):
        if "v_latest_scores" in sql:
            reads.append(1)
        return original(sql, params)

    monkeypatch.setattr(store, "query", counting)
    with recorder.event("plan", "m"):
        for _ in range(50):
            assert recorder.lookup("m", "A", "text", "fp")["status"] == "ok"
        assert recorder.lookup("m", "a", "other text", "fp") is None
        assert len(reads) == 1
        recorder.record_many(
            "m", "fp", [{"column": "b", "description": "t2", "total": 5, "status": "rewrite"}]
        )
        assert recorder.lookup("m", "b", "t2", "fp")["status"] == "rewrite"  # sees the new score
        assert len(reads) == 2
    assert recorder.lookup("m", "a", "text", "fp")["status"] == "ok"  # outside a call: direct read


# ── schema migration ──────────────────────────────────────────────────────


def test_a_version_1_database_is_migrated_to_the_current_schema(tmp_path):
    import duckdb

    path = tmp_path / "old.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    conn.execute(
        "CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, started_at TIMESTAMP, ended_at TIMESTAMP, "
        "status VARCHAR, goal VARCHAR, target_model VARCHAR, manifest_id VARCHAR, "
        "rubric_fp VARCHAR, params_json VARCHAR)"
    )
    conn.execute(
        "INSERT INTO runs VALUES ('old_run', now(), NULL, 'open', 'g', '', NULL, 'fp', '{}')"
    )
    conn.close()

    store = RunStore(path)
    assert store.query("SELECT value FROM meta WHERE key = 'schema_version'") == [
        {"value": str(SCHEMA_VERSION)}
    ]
    assert store.get_run("old_run")["goal"] == "g"  # existing data is kept
    store.upsert_work_items("old_run", [{"model": "m", "role": "seed", "status": "needs_test"}])
    assert store.load_work_items("old_run")[0]["status"] == "needs_test"


def test_work_items_keep_their_first_seen_time_and_update_in_place(store):
    run = store.create_run("g", "", None, "fp", {})
    store.upsert_work_items(
        run, [{"model": "m", "role": "seed", "status": "needs_test", "depth": 0}]
    )
    first = store.load_work_items(run)[0]
    store.upsert_work_items(
        run,
        [
            {
                "model": "m",
                "role": "seed",
                "status": "passed",
                "depth": 0,
                "ratio": 1.0,
                "summary": {"a": 1},
            }
        ],
    )
    rows = store.load_work_items(run)
    assert len(rows) == 1 and rows[0]["status"] == "passed" and rows[0]["summary"] == {"a": 1}
    assert rows[0]["first_seen"] == first["first_seen"]
