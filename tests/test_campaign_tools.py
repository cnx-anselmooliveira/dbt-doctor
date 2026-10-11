"""End-to-end: campaign tools through the MCP server, driven by a simulated assistant."""

import json

import duckdb
import pytest

from dbt_doctor import server
from dbt_doctor.core.manifest import DbtManifestParser

from .test_campaign import COPY, ID, chain, manifest_for, model


@pytest.fixture
def project(tmp_path, monkeypatch):
    models = chain() + [
        model(
            "az_join",
            ["az"],
            "select a.id from {{ ref('silver_a') }} a join {{ ref('stg_a') }} b on a.id = b.id",
            {"id": "Something."},
            ["silver_a", "stg_a"],
        ),
        model("view_copy", ["view"], COPY % "gold_a", {"id": ID}, ["gold_a"]),
    ]
    manifest = manifest_for(models)
    for node in manifest["nodes"].values():
        folder = node["fqn"][1]
        node.update(
            alias=node["name"],
            schema=folder,
            database="lake",
            original_file_path=f"models/{folder}/{node['name']}.sql",
            config={"materialized": "table"},
            tags=[],
        )
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(server, "_manifest", DbtManifestParser(str(tmp_path)))
    monkeypatch.setattr(server, "_rubric_path", None)
    monkeypatch.setattr(server, "_connector", None)
    monkeypatch.setattr(server, "_recorder", None)
    monkeypatch.setattr(server, "_run_store_enabled", True)
    monkeypatch.setattr(server, "_run_store_path", None)
    return tmp_path


def tool(fn, *args, **kwargs):
    return json.loads(fn(*args, **kwargs))


def assistant(scorer, rounds=20):
    """Play the client LLM: do every task get_next_work hands out until nothing is left."""
    work = tool(server.get_next_work, limit=10)
    for _ in range(rounds):
        if not work["tasks"]:
            return work
        for task in work["tasks"]:
            offset = 0
            while offset is not None:
                batch = tool(server.evaluate_model_descriptions, task["model"], offset)
                ids = [q["id"] for q in batch["rubric"]["questions"]]
                scores = [
                    {
                        "column": c["name"],
                        "scores": dict.fromkeys(ids, scorer(task["model"], c["name"])),
                    }
                    for c in batch["columns"]
                    if c.get("status") != "missing"
                ]
                server.submit_description_scores(task["model"], scores)
                offset = batch["next_offset"]
        work = tool(server.get_next_work, limit=10)
    raise AssertionError("the campaign did not finish")


def statuses():
    page = tool(server.get_campaign_status, limit=100)
    return {t["model"]: t["status"] for t in page["tables"]}


# ── get_table_lineage ─────────────────────────────────────────────────────


def test_table_lineage_lists_parents_one_level_up_and_every_descendant(project):
    lineage = tool(server.get_table_lineage, "stg_a")
    assert [(p["name"], p["layer"], p["deferred"]) for p in lineage["parents"]] == [
        ("raw_a", "bronze", False)
    ]
    assert lineage["direct_children"] == ["az_join", "silver_a"]
    below = lineage["descendants"]
    assert below["total"] == 4 and below["max_depth"] == 3
    assert [(i["name"], i["depth"], i["sql"]) for i in below["items"]] == [
        ("az_join", 1, "unreadable"),
        ("silver_a", 1, "projection"),
        ("gold_a", 2, "direct"),
        ("view_copy", 3, "direct"),
    ]
    assert below["by_layer"] == {"az": 1, "silver": 1, "gold": 1, "view": 1}


def test_table_lineage_marks_landing_as_deferred_and_pages_and_limits_depth(project):
    lineage = tool(server.get_table_lineage, "raw_a")
    assert [(p["name"], p["deferred"]) for p in lineage["parents"]] == [("landing_a", True)]
    first = tool(server.get_table_lineage, "raw_a", 0, 0, 2)
    assert len(first["descendants"]["items"]) == 2 and first["descendants"]["next_offset"] == 2
    assert tool(server.get_table_lineage, "raw_a", 0, 2, 2)["descendants"]["offset"] == 2
    shallow = tool(server.get_table_lineage, "raw_a", 1)
    assert [i["name"] for i in shallow["descendants"]["items"]] == ["stg_a"]
    assert "not found" in server.get_table_lineage("ghost")


# ── starting a campaign ───────────────────────────────────────────────────


def test_a_bad_request_starts_nothing(project):
    assert "did you mean stg_a" in server.start_campaign(["stg_b"])
    assert server.start_campaign([]).startswith("❌")
    assert "children_scope" in server.start_campaign(["stg_a"], children_scope="everything")
    assert "pass_ratio" in server.start_campaign(["stg_a"], pass_ratio=0)
    assert not (project / ".dbt-doctor" / "dbt_doctor.duckdb").exists()


def test_the_tools_explain_when_there_is_no_campaign(project):
    assert "No campaign yet" in server.get_next_work()
    assert "No campaign yet" in server.get_campaign_status()
    server.start_run("just a run")
    assert "not a campaign" in server.get_next_work()


def test_start_campaign_returns_the_first_tasks_and_records_the_configuration(project):
    started = tool(server.start_campaign, ["stg_a"], "document stg_a", "seeds", 0.8)
    assert started["config"] == {"seeds": ["stg_a"], "children_scope": "seeds", "pass_ratio": 0.8}
    assert [t["model"] for t in started["tasks"]] == ["raw_a"]
    assert started["phase"] == "A_in_progress" and started["run"]["recording"] == "ok"
    conn = duckdb.connect(str(project / ".dbt-doctor" / "dbt_doctor.duckdb"), read_only=True)
    goal, params = conn.execute("SELECT goal, params_json FROM runs").fetchone()
    conn.close()
    assert goal == "document stg_a" and json.loads(params)["campaign"]["pass_ratio"] == 0.8


# ── the whole flow ────────────────────────────────────────────────────────


def test_a_campaign_where_the_parent_passes_ends_with_children_untested(project):
    tool(server.start_campaign, ["stg_a"])
    final = assistant(lambda model, column: 8)
    assert final["phase"] == "A_complete"
    found = statuses()
    assert found["raw_a"] == "passed" and found["stg_a"] == "passed"
    assert {found[n] for n in ("silver_a", "az_join")} == {"untested_child"}  # direct children only
    assert "gold_a" not in found and "view_copy" not in found
    assert "landing_a" not in found  # nothing above a passing parent


def test_a_campaign_where_the_parent_fails_reaches_landing_and_keeps_it_deferred(project):
    tool(server.start_campaign, ["stg_a"])
    final = assistant(lambda model, column: 2)
    found = statuses()
    assert final["phase"] == "A_complete"
    assert found["raw_a"] == "failed_self" and found["stg_a"] == "blocked_by_parent"
    assert found["landing_a"] == "deferred"
    page = tool(server.get_campaign_status, "deferred")
    assert page["matching"] == 1 and page["tables"][0]["reason"] == "landing is never processed"


def test_several_seeds_share_their_work_and_nothing_is_scored_twice(project):
    tool(server.start_campaign, ["stg_a", "silver_a"])
    seen = []

    def scorer(model, column):
        seen.append((model, column))
        return 8

    assistant(scorer)
    assert len(seen) == len(set(seen))  # every column was scored once
    assert statuses()["silver_a"] in ("passed", "blocked_by_parent")


def test_the_campaign_survives_a_restart(project):
    tool(server.start_campaign, ["stg_a"])
    tool(server.get_next_work)
    server._recorder = None  # the server process restarted
    resumed = tool(server.start_run, "", "", True)
    assert resumed["resumed"] is True
    assert statuses()["stg_a"] == "blocked_by_parent"
    final = assistant(lambda model, column: 8)
    assert final["phase"] == "A_complete" and statuses()["stg_a"] == "passed"


def test_skip_work_item_and_the_error_after_repeated_requests_without_progress(project):
    tool(server.start_campaign, ["stg_a"])
    for _ in range(6):
        work = tool(server.get_next_work)  # nobody scores raw_a
    assert statuses()["raw_a"] == "error" and work["tasks"] == []
    assert "not in this campaign" in server.skip_work_item("ghost")
    skipped = tool(server.skip_work_item, "stg_a", "owned elsewhere")
    assert skipped["status"] == "skipped"
    assert statuses()["stg_a"] == "skipped"


def test_the_run_report_summarises_the_campaign(project):
    tool(server.start_campaign, ["stg_a"])
    assistant(lambda model, column: 8)
    report = tool(server.get_run_report)
    assert report["campaign"]["passed"] == 2 and report["campaign"]["untested_child"] == 2
    tools = {e["tool"] for e in report["events_by_tool"]}
    assert {"start_campaign", "get_next_work", "evaluate_model_descriptions"} <= tools


def test_campaigns_need_the_run_store(project, monkeypatch):
    monkeypatch.setattr(server, "_run_store_enabled", False)
    assert "need the run store" in server.start_campaign(["stg_a"])
    assert "need the run store" in server.get_next_work()
    assert "need the run store" in server.get_campaign_status()


def test_a_restarted_server_never_closes_the_open_campaign_by_accident(project):
    started = tool(server.start_campaign, ["stg_a"])
    server._recorder = None  # the server process restarted
    message = server.get_next_work()
    assert "No campaign yet" in message and started["run"]["run_id"] in message
    assert "resume=True" in message
    assert "No campaign yet" in server.skip_work_item("stg_a")
    # Nothing was opened or closed by those calls: the campaign can still be resumed.
    assert tool(server.start_run, "", "", True)["resumed"] is True
    assert tool(server.get_next_work)["tasks"][0]["model"] == "raw_a"
