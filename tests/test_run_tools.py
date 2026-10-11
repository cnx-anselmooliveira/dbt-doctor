"""End-to-end: the flow tools record into the run store through the MCP server."""

import json

import duckdb
import pytest

from dbt_doctor import server
from dbt_doctor.analyzers.description_quality import load_rubric
from dbt_doctor.analyzers.score_store import ScoreStore
from dbt_doctor.core.run_store import StoreUnavailable

from .test_lineage_tools import project, score_model  # noqa: F401  (fixtures and helper)


@pytest.fixture(autouse=True)
def fresh_recorder(monkeypatch):
    monkeypatch.setattr(server, "_recorder", None)
    monkeypatch.setattr(server, "_run_store_enabled", True)
    monkeypatch.setattr(server, "_run_store_path", None)


def db_path(project):  # noqa: F811
    return project / ".dbt-doctor" / "dbt_doctor.duckdb"


def rows(project, sql):  # noqa: F811
    conn = duckdb.connect(str(db_path(project)), read_only=True)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    finally:
        conn.close()


def run_flow(project):  # noqa: F811
    """evaluate+score the parent, plan the child, then write (dry run and real)."""
    score_model("raw_orders")
    plan = json.loads(server.plan_description_updates("silver_orders"))
    preview = json.loads(
        server.update_model_yaml(
            "silver_orders",
            "",
            [{"name": "status_up", "description": "Upper status."}],
            dry_run=True,
        )
    )
    written = json.loads(
        server.update_model_yaml(
            "silver_orders", "", [{"name": "status_up", "description": "Upper status."}]
        )
    )
    return plan, preview, written


def test_every_flow_tool_records_into_one_run(project):  # noqa: F811
    batch = json.loads(server.evaluate_model_descriptions("raw_orders"))
    run_id = batch["run"]["run_id"]
    assert batch["run"]["recording"] == "ok" and run_id
    assert db_path(project).exists()
    assert (db_path(project).parent / ".gitignore").read_text(encoding="utf-8") == "*\n"

    plan, preview, written = run_flow(project)
    assert {plan["run"]["run_id"], preview["run"]["run_id"], written["run"]["run_id"]} == {run_id}

    tools = [e["tool"] for e in rows(project, "SELECT tool FROM events ORDER BY seq")]
    assert tools[0] == "evaluate_model_descriptions"
    assert {"submit_description_scores", "plan_description_updates", "update_model_yaml"} <= set(
        tools
    )
    assert rows(project, "SELECT count(DISTINCT run_id) AS n FROM events") == [{"n": 1}]


def test_scores_plans_edges_and_snapshots_are_recorded(project):  # noqa: F811
    run_flow(project)
    scores = rows(project, "SELECT model, status FROM v_latest_scores ORDER BY column_name")
    assert {s["model"] for s in scores} == {"raw_orders"} and {s["status"] for s in scores} == {
        "ok"
    }
    plan = {
        r["column_name"]: r
        for r in rows(project, "SELECT * FROM plan_items WHERE model = 'silver_orders'")
    }
    assert plan["id"]["action"] == "copy" and plan["id"]["parent_model"] == "raw_orders"
    edges = {
        r["column_name"]: r["kind"]
        for r in rows(project, "SELECT * FROM lineage_edges WHERE model = 'silver_orders'")
    }
    assert edges["status_up"] == "transformed"
    seen = rows(project, "SELECT model, count(*) AS n FROM snapshots GROUP BY model ORDER BY model")
    assert seen == [{"model": "raw_orders", "n": 2}, {"model": "silver_orders", "n": 3}]
    per_question = rows(project, "SELECT per_question_json FROM scores LIMIT 1")[0][
        "per_question_json"
    ]
    assert set(json.loads(per_question)) == {q.id for q in load_rubric().questions}


def test_writes_keep_the_old_and_new_text_and_the_dry_run_flag(project):  # noqa: F811
    run_flow(project)
    writes = rows(
        project,
        "SELECT w.column_name, w.action, w.via, w.dry_run, o.text AS old, n.text AS new "
        "FROM writes w LEFT JOIN descriptions o ON o.hash = w.old_hash "
        "LEFT JOIN descriptions n ON n.hash = w.new_hash ORDER BY w.dry_run DESC, w.column_name",
    )
    assert [(w["column_name"], w["dry_run"], w["via"]) for w in writes] == [
        ("id", True, "copy"),
        ("status_up", True, "supplied"),
        ("id", False, "copy"),
        ("status_up", False, "supplied"),
    ]
    status_up = writes[-1]
    assert status_up["action"] == "changed" and status_up["new"] == "Upper status."
    assert any(w["old"] == "some other wording" for w in writes)  # what the copy replaced


def test_blocked_and_rejected_columns_are_recorded_too(project):  # noqa: F811
    server.update_model_yaml(
        "silver_orders", "", [{"name": "id", "description": "mine"}]
    )  # parent unscored
    score_model("raw_orders")
    server.update_model_yaml(
        "silver_orders", "", [{"name": "status_up", "description": "Order status."}]
    )
    actions = {r["action"]: r for r in rows(project, "SELECT * FROM writes")}
    assert "parent" in actions["skipped"]["reason"] and actions["skipped"]["column_name"] == "id"
    assert actions["rejected"]["column_name"] == "status_up"


def test_run_report_and_list_runs(project):  # noqa: F811
    run_flow(project)
    report = json.loads(server.get_run_report())
    models = {m["model"]: m for m in report["models"]}
    tools = {e["tool"]: e["n"] for e in report["events_by_tool"]}
    assert tools["update_model_yaml"] == 2 and tools["evaluate_model_descriptions"] == 1
    assert models["raw_orders"]["scores"] == {"ok": 2}
    assert models["silver_orders"]["writes"]["changed"] >= 1
    assert report["run"]["status"] == "open"
    listing = json.loads(server.list_runs())
    assert listing["current_run"] == report["run"]["run_id"]
    assert listing["runs"][0]["events"] == sum(tools.values())
    assert "not found" in server.get_run_report("ghost")


def test_start_run_sets_the_context_and_resume_continues_it(project):  # noqa: F811
    started = json.loads(server.start_run("document silver_orders", "silver_orders"))
    batch = json.loads(server.evaluate_model_descriptions("silver_orders"))
    assert batch["run"]["run_id"] == started["run_id"]
    run = rows(project, "SELECT goal, target_model, rubric_fp FROM runs")[0]
    assert (
        run["goal"] == "document silver_orders" and run["rubric_fp"] == load_rubric().fingerprint()
    )

    server._recorder = None  # a new server process
    resumed = json.loads(server.start_run(target_model="silver_orders", resume=True))
    assert resumed["resumed"] is True and resumed["run_id"] == started["run_id"]
    new = json.loads(server.start_run("another campaign"))
    assert new["run_id"] != started["run_id"]
    assert rows(project, f"SELECT status FROM runs WHERE run_id = '{started['run_id']}'") == [
        {"status": "closed"}
    ]


def test_legacy_json_scores_are_migrated_on_first_use(project):  # noqa: F811
    legacy = project / "target" / "dbt_doctor" / "description_scores.json"
    fingerprint = load_rubric().fingerprint()
    ScoreStore(legacy).record("raw_orders", "id", "Order identifier.", fingerprint, 40, "ok")

    plan = json.loads(server.plan_description_updates("silver_orders"))
    entry = next(c for c in plan["columns"] if c["column"] == "id")
    assert entry["action"] == "copy"  # the imported score unblocked the copy
    assert not legacy.exists() and legacy.with_suffix(".json.migrated").exists()


def test_the_flow_keeps_working_when_the_store_is_unavailable(project, monkeypatch):  # noqa: F811
    def locked(self):
        raise StoreUnavailable("database is locked")

    monkeypatch.setattr(server.RunStore, "_open", locked)
    batch = json.loads(server.evaluate_model_descriptions("silver_orders"))
    assert batch["model"] == "silver_orders"
    assert batch["run"]["recording"] == "unavailable: database is locked"
    plan = json.loads(server.plan_description_updates("silver_orders"))
    entry = next(c for c in plan["columns"] if c["column"] == "id")
    assert entry["action"] == "blocked" and entry["parent_status"] == "unscored"
    assert server.list_runs().startswith("❌")  # reading the history needs the store


def test_disabling_the_run_store_falls_back_to_the_json_scores(project, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "_run_store_enabled", False)
    score_model("raw_orders")  # scores go to target/dbt_doctor/description_scores.json
    assert (project / "target" / "dbt_doctor" / "description_scores.json").exists()
    entry = next(
        c
        for c in json.loads(server.plan_description_updates("silver_orders"))["columns"]
        if c["column"] == "id"
    )
    assert entry["action"] == "copy"
    assert "run" not in json.loads(server.evaluate_model_descriptions("raw_orders"))
    assert not db_path(project).exists()
    assert "disabled" in server.start_run("x") and "disabled" in server.list_runs()


def test_a_custom_store_path_is_used(project, tmp_path, monkeypatch):  # noqa: F811
    custom = tmp_path / "elsewhere" / "history.duckdb"
    monkeypatch.setattr(server, "_run_store_path", custom)
    server.evaluate_model_descriptions("raw_orders")
    assert custom.exists() and not db_path(project).exists()
