"""End-to-end tests of the lineage tools and the write barrier through the MCP server."""

import json

import pytest

from dbt_doctor import server
from dbt_doctor.core.manifest import DbtManifestParser

RAW_YAML = """\
version: 2

models:
  - name: raw_orders
    description: Bronze orders
    columns:
      - name: id
        description: Order identifier.
      - name: status
        description: Order status.
"""

SILVER_YAML = """\
version: 2

models:
  - name: silver_orders
    description: Silver orders
    columns:
      - name: id
        description: some other wording
      - name: status
        description: Order status.
      - name: status_up
        description: ""
"""


def node(name, folder, sql, columns, parents=()):
    return {
        "resource_type": "model",
        "name": name,
        "alias": name,
        "schema": folder,
        "database": "lake",
        "fqn": ["proj", folder, name],
        "original_file_path": f"models/{folder}/{name}.sql",
        "raw_code": sql,
        "config": {"materialized": "table"},
        "tags": [],
        "columns": {c: {"description": d} for c, d in columns.items()},
        "depends_on": {"nodes": [f"model.proj.{p}" for p in parents]},
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    manifest = {
        "nodes": {
            "model.proj.raw_orders": node(
                "raw_orders",
                "bronze",
                "select * from {{ source('s', 'orders') }}",
                {"id": "Order identifier.", "status": "Order status."},
            ),
            "model.proj.silver_orders": node(
                "silver_orders",
                "silver",
                "select id, status, upper(status) as status_up from {{ ref('raw_orders') }}",
                {"id": "some other wording", "status": "Order status.", "status_up": ""},
                parents=["raw_orders"],
            ),
        },
        "sources": {},
    }
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for folder, name, text in (
        ("bronze", "raw_orders", RAW_YAML),
        ("silver", "silver_orders", SILVER_YAML),
    ):
        (tmp_path / "models" / folder).mkdir(parents=True)
        (tmp_path / "models" / folder / f"{name}.sql").write_text("select 1")
        (tmp_path / "models" / folder / f"_{name}.yml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(server, "_manifest", DbtManifestParser(str(tmp_path)))
    monkeypatch.setattr(server, "_rubric_path", None)
    monkeypatch.setattr(server, "_connector", None)
    return tmp_path


def score_model(model, total=9, skip=()):
    """Run the evaluation flow for a model, scoring every described column."""
    batch = json.loads(server.evaluate_model_descriptions(model))
    ids = [q["id"] for q in batch["rubric"]["questions"]]
    scores = [
        {"column": c["name"], "scores": dict.fromkeys(ids, total)}
        for c in batch["columns"]
        if c.get("status") != "missing" and c["name"] not in skip
    ]
    return json.loads(server.submit_description_scores(model, scores))


def plan(model):
    return json.loads(server.plan_description_updates(model))


def test_the_evaluation_flow_unblocks_the_copy(project):
    before = {c["column"]: c for c in plan("silver_orders")["columns"]}
    assert before["id"]["action"] == "blocked" and before["id"]["parent_status"] == "unscored"

    score_model("raw_orders")
    after = {c["column"]: c for c in plan("silver_orders")["columns"]}
    assert after["id"]["action"] == "copy" and after["id"]["copy_text"] == "Order identifier."
    assert after["status"]["action"] == "keep"  # already identical
    assert after["status_up"]["action"] == "write_new"  # transformed and empty: needs its own text
    assert plan("silver_orders")["blocked_by"] == []


def test_a_failed_parent_keeps_blocking(project):
    score_model("raw_orders", total=2)  # 10 of 50: below the threshold
    entry = next(c for c in plan("silver_orders")["columns"] if c["column"] == "id")
    assert entry["action"] == "blocked" and entry["parent_status"] == "failed"


def test_update_blocks_then_copies_after_scoring(project):
    silver_yaml = project / "models" / "silver" / "_silver_orders.yml"

    blocked = json.loads(
        server.update_model_yaml("silver_orders", "", [{"name": "id", "description": "mine"}])
    )
    assert blocked["skipped"][0]["column"] == "id"
    assert "some other wording" in silver_yaml.read_text(encoding="utf-8")

    score_model("raw_orders")
    preview = json.loads(
        server.update_model_yaml(
            "silver_orders", "", [{"name": "id", "description": "mine"}], dry_run=True
        )
    )
    assert preview["status"] == "dry_run" and preview["next_step"] is None
    assert preview["copied"][0]["supplied_text_ignored"] is True
    assert "some other wording" in silver_yaml.read_text(encoding="utf-8")  # nothing written

    written = json.loads(
        server.update_model_yaml(
            "silver_orders",
            "",
            [
                {"name": "id", "description": "mine"},
                {"name": "status_up", "description": "Upper-cased status."},
            ],
        )
    )
    content = silver_yaml.read_text(encoding="utf-8")
    assert written["status"] == "written" and "dbt parse" in written["next_step"]
    assert "Order identifier." in content and "mine" not in content
    assert "some other wording" not in content
    assert "Upper-cased status." in content


def test_update_without_overwrite_keeps_existing_text_but_still_enforces(project):
    score_model("raw_orders")
    result = json.loads(
        server.update_model_yaml(
            "silver_orders", "", [], overwrite_descriptions=False, apply_copies=True
        )
    )
    content = (project / "models" / "silver" / "_silver_orders.yml").read_text(encoding="utf-8")
    assert "some other wording" in content  # not replaced
    assert result["copied"][0]["column"] == "id"


def test_stale_copy_of_a_transformed_column_is_rejected(project):
    score_model("raw_orders")
    result = json.loads(
        server.update_model_yaml(
            "silver_orders", "", [{"name": "status_up", "description": "Order status."}]
        )
    )
    assert result["rejected"][0]["column"] == "status_up"
    assert (
        "Order status."
        not in (project / "models" / "silver" / "_silver_orders.yml")
        .read_text(encoding="utf-8")
        .split("status_up")[1]
    )


def test_trace_and_consistency_tools(project):
    trace = json.loads(server.trace_column_lineage("silver_orders", "id"))
    assert [s["model"] for s in trace["chain"]] == ["silver_orders", "raw_orders"]

    transformed = json.loads(server.trace_column_lineage("silver_orders", "status_up"))
    assert len(transformed["chain"]) == 1

    report = json.loads(server.check_description_consistency("silver"))
    assert report["total_models"] == 1 and report["models"][0]["model"] == "silver_orders"
    everything = json.loads(server.check_description_consistency())
    assert [m["model"] for m in everything["models"]] == ["raw_orders", "silver_orders"]


def test_unknown_model_returns_a_readable_error(project):
    assert server.plan_description_updates("ghost").startswith("❌")
    assert server.trace_column_lineage("ghost", "id").startswith("❌")
    assert "not found" in server.update_model_yaml("ghost", "", [])


def test_evaluation_batch_includes_the_parent_context(project):
    batch = json.loads(server.evaluate_model_descriptions("silver_orders"))
    cols = {c["name"]: c for c in batch["columns"]}
    assert cols["id"]["upstream"]["parent"] == "raw_orders"
    assert cols["id"]["upstream"]["parent_description"] == "Order identifier."
    assert cols["status_up"]["upstream"]["built_by"]["expression"] == "upper(status)"
    assert "built_by" in batch["instructions"]
    root = json.loads(server.evaluate_model_descriptions("raw_orders"))
    assert all("upstream" not in c for c in root["columns"])  # nothing above the root layer


def test_evaluation_still_works_if_the_context_cannot_be_built(project, monkeypatch):
    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "_get_planner", broken)
    batch = json.loads(server.evaluate_model_descriptions("silver_orders"))
    assert batch["model"] == "silver_orders"
    assert all("upstream" not in c for c in batch["columns"])
