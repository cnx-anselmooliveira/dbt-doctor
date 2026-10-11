"""Tests for column description lineage (layers, edges, planner and write barrier)."""

import pytest

from dbt_doctor.analyzers.lineage import LayerMap, LineageGraph, LineagePlanner
from dbt_doctor.analyzers.score_store import ScoreStore

FP = "fp1"

COPY_SQL = "select * from {{ ref('%s') }}"


def make_manifest(*models, sources=None):
    """models: dicts with name, sql, columns {name: description}, parents, folders."""
    nodes = {}
    for m in models:
        uid = f"model.proj.{m['name']}"
        nodes[uid] = {
            "resource_type": "model",
            "name": m["name"],
            "fqn": ["proj", *m.get("folders", []), m["name"]],
            "raw_code": m["sql"],
            "columns": {c: {"description": d} for c, d in m["columns"].items()},
            "depends_on": {"nodes": [f"model.proj.{p}" for p in m.get("parents", [])]},
        }
    return {"nodes": nodes, "sources": sources or {}}


def base_models():
    return [
        {
            "name": "landing_orders",
            "sql": "select * from {{ source('s', 'orders') }}",
            "columns": {"id": "landing id"},
            "folders": ["landing"],
        },
        {
            "name": "raw_orders",
            "sql": COPY_SQL % "landing_orders",
            "columns": {
                "id": "Order identifier.",
                "status": "Order status.",
                "amount": "Amount in BRL.",
            },
            "parents": ["landing_orders"],
            "folders": ["bronze"],
        },
        {
            "name": "silver_orders",
            "sql": COPY_SQL % "raw_orders",
            "columns": {"id": "An identifier", "status": "Order status.", "amount": ""},
            "parents": ["raw_orders"],
            "folders": ["silver"],
        },
    ]


@pytest.fixture
def store(tmp_path):
    return ScoreStore(tmp_path / "scores.json")


def planner_for(models, store, sources=None):
    return LineagePlanner(LineageGraph(make_manifest(*models, sources=sources)), store, FP)


def score_ok(store, model, column, text, status="ok"):
    store.record(model, column, text, FP, 40 if status == "ok" else 10, status)


def plan_entry(planner, model, column):
    plan = planner.plan_model(model)
    return next(c for c in plan["columns"] if c["column"] == column)


# ── layers ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, folders, expected",
    [
        ("landing_x", [], "landing"),
        ("raw_x", ["zk", "bronze"], "bronze"),
        ("stg_x", ["zk", "stg"], "stg"),
        ("stg_x", ["zk", "gold"], "stg"),  # the name prefix wins over the folder
        ("aml_dim_x", ["amil", "silver"], "silver"),
        ("something", ["x", "intermediary"], "intermediary"),
        ("view_x", ["view"], "view"),
        ("az_report", ["gold"], "az"),
        ("mystery", ["misc"], "unknown"),
    ],
)
def test_layer_detection(name, folders, expected):
    assert LayerMap().layer_of(name, ["proj", *folders, name]) == expected


def test_layer_order_ranks_and_repeated_labels_keep_first_position():
    layers = LayerMap("landing,bronze,stg,silver,intermediary,stg,gold|view,az")
    assert layers.rank["stg"] == 2  # the repeat does not move it
    assert layers.rank["gold"] == layers.rank["view"]
    assert layers.is_root_or_below("landing") and layers.is_root_or_below("bronze")
    assert not layers.is_root_or_below("stg") and not layers.is_root_or_below("unknown")


def test_unknown_root_layer_is_rejected():
    with pytest.raises(ValueError, match="not in the layer order"):
        LayerMap("a,b", root="bronze")


# ── column edges ──────────────────────────────────────────────────────────


def test_root_layers_have_no_edges():
    graph = LineageGraph(make_manifest(*base_models()))
    assert graph.edge("model.proj.raw_orders", "id") is None
    assert graph.edge("model.proj.landing_orders", "id") is None


def test_copy_model_edges_are_passthrough_with_high_confidence():
    graph = LineageGraph(make_manifest(*base_models()))
    edge = graph.edge("model.proj.silver_orders", "id")
    assert (edge.kind, edge.parent, edge.parent_column) == (
        "passthrough",
        "model.proj.raw_orders",
        "id",
    )
    assert edge.confidence == "high" and edge.effects == []


def test_projection_edges_cover_rename_transform_and_missing_columns():
    models = base_models() + [
        {
            "name": "stg_orders",
            "sql": "select id, status as state, upper(status) as status_up from {{ ref('raw_orders') }} where amount > 0",
            "columns": {"id": "", "state": "", "status_up": "", "ghost": ""},
            "parents": ["raw_orders"],
            "folders": ["stg"],
        }
    ]
    graph = LineageGraph(make_manifest(*models))
    uid = "model.proj.stg_orders"
    renamed = graph.edge(uid, "state")
    assert (renamed.kind, renamed.parent_column, renamed.effects) == (
        "renamed",
        "status",
        ["where"],
    )
    assert graph.edge(uid, "status_up").kind == "transformed"
    assert graph.edge(uid, "id").kind == "passthrough"
    missing = graph.edge(uid, "ghost")
    assert missing.kind == "unknown" and missing.confidence == "low"


def test_join_models_are_unknown_with_low_confidence():
    models = base_models() + [
        {
            "name": "silver_joined",
            "sql": "select a.id from {{ ref('raw_orders') }} a join {{ ref('silver_orders') }} b on a.id = b.id",
            "columns": {"id": ""},
            "parents": ["raw_orders", "silver_orders"],
            "folders": ["silver"],
        }
    ]
    edge = LineageGraph(make_manifest(*models)).edge("model.proj.silver_joined", "id")
    assert edge.kind == "unknown" and edge.confidence == "low"
    assert "join" in edge.evidence


# ── plan: copy rules ──────────────────────────────────────────────────────


def test_unscored_parent_blocks_the_copy(store):
    entry = plan_entry(planner_for(base_models(), store), "silver_orders", "id")
    assert (entry["rule"], entry["action"], entry["parent_status"]) == (
        "must_copy",
        "blocked",
        "unscored",
    )
    assert "evaluate_model_descriptions on raw_orders" in entry["reason"]


def test_failed_parent_blocks_the_copy(store):
    score_ok(store, "raw_orders", "id", "Order identifier.", status="rewrite")
    entry = plan_entry(planner_for(base_models(), store), "silver_orders", "id")
    assert entry["action"] == "blocked" and entry["parent_status"] == "failed"


def test_parent_without_description_blocks_the_copy(store):
    models = base_models()
    models[1]["columns"]["id"] = ""
    entry = plan_entry(planner_for(models, store), "silver_orders", "id")
    assert entry["action"] == "blocked" and entry["parent_status"] == "missing"


def test_passed_parent_makes_a_different_child_description_a_copy(store):
    score_ok(store, "raw_orders", "id", "Order identifier.")
    entry = plan_entry(planner_for(base_models(), store), "silver_orders", "id")
    assert entry["action"] == "copy" and entry["verdict"] == "inconsistent"
    assert entry["copy_text"] == "Order identifier."


def test_identical_description_is_kept_and_whitespace_does_not_matter(store):
    score_ok(store, "raw_orders", "status", "Order status.")
    models = base_models()
    models[2]["columns"]["status"] = "Order   status.\n"
    entry = plan_entry(planner_for(models, store), "silver_orders", "status")
    assert entry["action"] == "keep" and entry["verdict"] == "ok"


def test_empty_child_description_is_copied_when_parent_passed(store):
    score_ok(store, "raw_orders", "amount", "Amount in BRL.")
    entry = plan_entry(planner_for(base_models(), store), "silver_orders", "amount")
    assert entry["action"] == "copy" and entry["verdict"] == "missing"


def test_a_rewritten_parent_description_needs_a_new_score(store):
    score_ok(store, "raw_orders", "id", "Old text that was scored.")
    entry = plan_entry(planner_for(base_models(), store), "silver_orders", "id")
    assert (
        entry["parent_status"] == "unscored"
    )  # the manifest text changed, the score no longer applies


# ── plan: rules that require a different description ──────────────────────


def altering_models(child_columns, sql):
    return base_models() + [
        {
            "name": "stg_filtered",
            "sql": sql,
            "columns": child_columns,
            "parents": ["raw_orders"],
            "folders": ["stg"],
        }
    ]


def test_passthrough_through_a_filter_must_differ_from_the_parent(store):
    sql = "select * from {{ ref('raw_orders') }} where status = 'paid'"
    stale = plan_entry(
        planner_for(altering_models({"status": "Order status."}, sql), store),
        "stg_filtered",
        "status",
    )
    assert (stale["rule"], stale["action"], stale["verdict"]) == (
        "must_differ",
        "write_new",
        "stale_copy",
    )
    own = plan_entry(
        planner_for(altering_models({"status": "Status of paid orders."}, sql), store),
        "stg_filtered",
        "status",
    )
    assert own["action"] == "keep" and own["verdict"] == "ok"
    empty = plan_entry(
        planner_for(altering_models({"status": ""}, sql), store), "stg_filtered", "status"
    )
    assert empty["action"] == "write_new" and empty["verdict"] == "missing"


def test_transformed_column_copied_from_the_parent_is_a_stale_copy(store):
    sql = "select upper(status) as status from {{ ref('raw_orders') }}"
    entry = plan_entry(
        planner_for(altering_models({"status": "Order status."}, sql), store),
        "stg_filtered",
        "status",
    )
    assert entry["rule"] == "must_differ" and entry["verdict"] == "stale_copy"


def test_renamed_passthrough_copies_from_the_source_column(store):
    score_ok(store, "raw_orders", "status", "Order status.")
    sql = "select status as state from {{ ref('raw_orders') }}"
    entry = plan_entry(
        planner_for(altering_models({"state": "Something"}, sql), store), "stg_filtered", "state"
    )
    assert entry["action"] == "copy" and entry["copy_text"] == "Order status."
    assert entry["lineage"]["parent_column"] == "status"


def test_unreadable_lineage_enforces_nothing(store):
    sql = "with t as (select * from {{ ref('raw_orders') }}) select * from t"
    kept = plan_entry(
        planner_for(altering_models({"status": "Order status."}, sql), store),
        "stg_filtered",
        "status",
    )
    assert (kept["rule"], kept["action"], kept["verdict"]) == ("free", "keep", "uncertain")
    empty = plan_entry(
        planner_for(altering_models({"status": ""}, sql), store), "stg_filtered", "status"
    )
    assert empty["action"] == "write_new"


def test_root_layer_is_described_on_its_own(store):
    planner = planner_for(base_models(), store)
    assert plan_entry(planner, "raw_orders", "id")["verdict"] == "root"
    models = base_models()
    models[1]["columns"]["id"] = ""
    assert plan_entry(planner_for(models, store), "raw_orders", "id")["action"] == "write_new"


def test_source_parent_is_described_on_its_own(store):
    models = [
        {
            "name": "silver_from_source",
            "sql": "select * from {{ source('s', 't') }}",
            "columns": {"id": "x"},
            "parents": [],
            "folders": ["silver"],
        }
    ]
    manifest = make_manifest(
        *models, sources={"source.proj.s.t": {"columns": {"id": {"description": "src"}}}}
    )
    manifest["nodes"]["model.proj.silver_from_source"]["depends_on"]["nodes"] = ["source.proj.s.t"]
    planner = LineagePlanner(LineageGraph(manifest), store, FP)
    entry = plan_entry(planner, "silver_from_source", "id")
    assert (entry["rule"], entry["action"], entry["verdict"]) == ("free", "keep", "source_parent")
    assert "blocked" not in {
        e["action"] for e in planner.plan_model("silver_from_source")["columns"]
    }


# ── write barrier ─────────────────────────────────────────────────────────


def test_enforce_copies_identically_and_ignores_the_supplied_text(store):
    score_ok(store, "raw_orders", "id", "Order identifier.")
    planner = planner_for(base_models(), store)
    final, report = planner.enforce(
        "silver_orders", [{"name": "id", "description": "my own words", "tests": ["unique"]}]
    )
    written = {c["name"]: c for c in final}
    assert written["id"]["description"] == "Order identifier."
    assert written["id"]["tests"] == ["unique"]  # other keys are kept
    assert report["copied"][0]["supplied_text_ignored"] is True


def test_enforce_skips_blocked_columns_but_keeps_their_tests(store):
    planner = planner_for(base_models(), store)
    final, report = planner.enforce(
        "silver_orders",
        [{"name": "id", "description": "x", "tests": ["not_null"]}],
        apply_copies=False,
    )
    assert final == [{"name": "id", "tests": ["not_null"]}]
    assert report["skipped"][0]["column"] == "id" and "raw_orders" in report["skipped"][0]["reason"]


def test_enforce_rejects_a_stale_copy_and_accepts_an_own_description(store):
    sql = "select * from {{ ref('raw_orders') }} where status = 'paid'"
    planner = planner_for(altering_models({"status": "Old", "amount": "Old"}, sql), store)
    final, report = planner.enforce(
        "stg_filtered",
        [
            {"name": "status", "description": "Order status."},
            {"name": "amount", "description": "Amount of paid orders in BRL."},
        ],
    )
    by_name = {c["name"]: c for c in final}
    assert "description" not in by_name["status"]
    assert by_name["amount"]["description"] == "Amount of paid orders in BRL."
    assert report["rejected"][0]["column"] == "status"
    assert (
        "filters" in report["rejected"][0]["reason"]
        or "alters rows" in report["rejected"][0]["reason"]
    )


def test_apply_copies_adds_columns_that_were_not_sent(store):
    score_ok(store, "raw_orders", "id", "Order identifier.")
    score_ok(store, "raw_orders", "amount", "Amount in BRL.")
    planner = planner_for(base_models(), store)
    final, report = planner.enforce("silver_orders", [], apply_copies=True)
    assert {c["name"]: c["description"] for c in final} == {
        "id": "Order identifier.",
        "amount": "Amount in BRL.",
    }
    assert {c["column"] for c in report["copied"]} == {"id", "amount"}
    final, _ = planner.enforce("silver_orders", [], apply_copies=False)
    assert final == []


def test_enforce_unknown_model_raises(store):
    with pytest.raises(ValueError, match="not found"):
        planner_for(base_models(), store).enforce("ghost", [])


# ── trace, ordering and reports ───────────────────────────────────────────


def test_trace_follows_passthroughs_down_to_the_root(store):
    chain = planner_for(base_models(), store).trace("silver_orders", "id")
    assert [(s["model"], s["column"]) for s in chain] == [
        ("silver_orders", "id"),
        ("raw_orders", "id"),
    ]
    assert chain[-1]["stop"] == "root layer"


def test_trace_stops_at_a_transformation(store):
    sql = "select upper(status) as status from {{ ref('raw_orders') }}"
    chain = planner_for(altering_models({"status": "x"}, sql), store).trace(
        "stg_filtered", "status"
    )
    assert len(chain) == 1 and "expression" in chain[0]["stop"]


def test_topological_order_puts_parents_first():
    graph = LineageGraph(make_manifest(*reversed(base_models())))
    names = [graph.models[u].name for u in graph.topological_order()]
    assert names.index("landing_orders") < names.index("raw_orders") < names.index("silver_orders")


def test_consistency_report_counts_pages_and_filters_by_layer(store):
    planner = planner_for(base_models(), store)
    full = planner.consistency(limit=2)
    assert full["total_models"] == 3 and full["next_offset"] == 2
    assert [m["model"] for m in full["models"]] == ["landing_orders", "raw_orders"]
    rest = planner.consistency(offset=2, limit=2)
    assert rest["next_offset"] is None and rest["models"][0]["model"] == "silver_orders"
    silver = planner.consistency("silver")
    assert silver["total_models"] == 1
    assert silver["totals"]["blocked"] == 3  # every column: the parent is unscored
    assert silver["models"][0]["blocked_by"] == ["raw_orders"]


def test_plan_model_summarises_actions_and_blockers(store):
    plan = planner_for(base_models(), store).plan_model("silver_orders")
    assert plan["layer"] == "silver" and plan["parents"] == ["raw_orders"]
    assert plan["summary"] == {"blocked": 3}
    assert plan["blocked_by"] == ["raw_orders"]
    assert all(not k.startswith("_") for c in plan["columns"] for k in c)


def test_transformed_column_is_compared_with_the_single_parent_column_it_uses(store):
    sql = "select upper(status) as status_up, concat(status, amount) as mixed from {{ ref('raw_orders') }}"
    planner = planner_for(
        altering_models({"status_up": "Order status.", "mixed": "Order status."}, sql), store
    )
    one_source = plan_entry(planner, "stg_filtered", "status_up")
    assert one_source["lineage"]["parent_column"] == "status"
    assert one_source["verdict"] == "stale_copy"  # same text as raw_orders.status
    # Two parent columns are involved: no single source, so no stale-copy claim.
    assert plan_entry(planner, "stg_filtered", "mixed")["verdict"] == "ok"


CAST_SQL = "select cast(status as varchar) as status_text from {{ ref('raw_orders') }}"


def test_casts_are_not_enforced_by_default(store):
    score_ok(store, "raw_orders", "status", "Order status.")
    own = plan_entry(
        planner_for(altering_models({"status_text": "My own words"}, CAST_SQL), store),
        "stg_filtered",
        "status_text",
    )
    assert (own["rule"], own["action"], own["verdict"]) == ("free", "keep", "cast")
    same = plan_entry(
        planner_for(altering_models({"status_text": "Order status."}, CAST_SQL), store),
        "stg_filtered",
        "status_text",
    )
    assert same["verdict"] == "cast" and same["action"] == "keep"  # not a stale copy
    empty = plan_entry(
        planner_for(altering_models({"status_text": ""}, CAST_SQL), store),
        "stg_filtered",
        "status_text",
    )
    assert empty["action"] == "write_new"


def test_casts_can_be_treated_as_passthrough(store):
    score_ok(store, "raw_orders", "status", "Order status.")
    graph = LineageGraph(make_manifest(*altering_models({"status_text": "My own words"}, CAST_SQL)))
    planner = LineagePlanner(graph, store, FP, casts_are_passthrough=True)
    entry = plan_entry(planner, "stg_filtered", "status_text")
    assert (entry["rule"], entry["action"]) == ("must_copy", "copy")
    assert entry["copy_text"] == "Order status." and entry["lineage"]["parent_column"] == "status"


def test_a_cast_inside_a_filtering_model_still_needs_its_own_text(store):
    sql = "select cast(status as varchar) as status_text from {{ ref('raw_orders') }} where amount > 0"
    entry = plan_entry(
        planner_for(altering_models({"status_text": "Order status."}, sql), store),
        "stg_filtered",
        "status_text",
    )
    assert entry["rule"] == "must_differ" and entry["verdict"] == "stale_copy"


def test_window_in_another_column_does_not_make_a_passthrough_stale(store):
    score_ok(store, "raw_orders", "id", "Order identifier.")
    sql = "select id, row_number() over (order by id) as rn from {{ ref('raw_orders') }}"
    entry = plan_entry(
        planner_for(altering_models({"id": "Order identifier.", "rn": ""}, sql), store),
        "stg_filtered",
        "id",
    )
    assert entry["rule"] == "must_copy" and entry["verdict"] == "ok"


def test_columns_over_a_source_can_be_written_freely(store):
    models = [
        {
            "name": "stg_from_source",
            "sql": "select id, name as customer_name from {{ source('s', 't') }}",
            "columns": {"id": "", "customer_name": "Old"},
            "folders": ["stg"],
        }
    ]
    manifest = make_manifest(
        *models, sources={"source.proj.s.t": {"columns": {"id": {"description": "src"}}}}
    )
    manifest["nodes"]["model.proj.stg_from_source"]["depends_on"]["nodes"] = ["source.proj.s.t"]
    planner = LineagePlanner(LineageGraph(manifest), store, FP)
    assert plan_entry(planner, "stg_from_source", "id")["action"] == "write_new"
    final, report = planner.enforce(
        "stg_from_source",
        [
            {"name": "id", "description": "Own text"},
            {"name": "customer_name", "description": "New"},
        ],
    )
    assert {c["name"]: c["description"] for c in final} == {
        "id": "Own text",
        "customer_name": "New",
    }
    assert report == {"copied": [], "skipped": [], "rejected": []}


# ── upstream context for the evaluation ───────────────────────────────────


def test_upstream_context_describes_how_each_column_is_built(store):
    sql = (
        "select id, status as state, upper(status) as status_up, cast(amount as bigint) as amount_n "
        "from {{ ref('raw_orders') }} where amount > 0"
    )
    columns = {"id": "", "state": "", "status_up": "", "amount_n": "", "ghost": ""}
    context = planner_for(altering_models(columns, sql), store).upstream_context("stg_filtered")
    assert context["id"]["parent"] == "raw_orders" and context["id"]["kind"] == "passthrough"
    assert context["id"]["parent_description"] == "Order identifier."
    assert context["id"]["model_effects"] == ["where"]
    assert (context["state"]["kind"], context["state"]["parent_column"]) == ("renamed", "status")
    assert context["status_up"]["kind"] == "transformed"
    assert context["status_up"]["built_by"] == {
        "model": "stg_filtered",
        "expression": "upper(status)",
    }
    assert context["amount_n"]["kind"] == "cast" and "built_by" not in context["amount_n"]
    assert context["ghost"]["kind"] == "unknown"


def test_upstream_context_finds_the_expression_further_up(store):
    models = base_models() + [
        {
            "name": "stg_upper",
            "sql": "select upper(status) as status from {{ ref('raw_orders') }}",
            "columns": {"status": "Upper status."},
            "parents": ["raw_orders"],
            "folders": ["stg"],
        },
        {
            "name": "silver_copy",
            "sql": "select * from {{ ref('stg_upper') }}",
            "columns": {"status": "Upper status."},
            "parents": ["stg_upper"],
            "folders": ["silver"],
        },
    ]
    info = planner_for(models, store).upstream_context("silver_copy")["status"]
    assert info["kind"] == "passthrough" and info["parent"] == "stg_upper"
    assert info["built_by"] == {"model": "stg_upper", "expression": "upper(status)"}


def test_upstream_context_is_empty_for_root_layers_and_flags_sources(store):
    planner = planner_for(base_models(), store)
    assert planner.upstream_context("raw_orders") == {}
    models = [
        {
            "name": "stg_src",
            "sql": "select id from {{ source('s', 't') }}",
            "columns": {"id": "x"},
            "folders": ["stg"],
        }
    ]
    manifest = make_manifest(
        *models, sources={"source.proj.s.t": {"columns": {"id": {"description": "src text"}}}}
    )
    manifest["nodes"]["model.proj.stg_src"]["depends_on"]["nodes"] = ["source.proj.s.t"]
    info = LineagePlanner(LineageGraph(manifest), store, FP).upstream_context("stg_src")["id"]
    assert info["parent_is_source"] is True and info["parent_description"] == "src text"


def test_long_parent_descriptions_are_truncated(store):
    models = base_models()
    models[1]["columns"]["id"] = "x" * 1000
    info = planner_for(models, store).upstream_context("silver_orders")["id"]
    assert len(info["parent_description"]) == 400


def test_upstream_context_lists_the_chain_so_renames_further_up_are_visible(store):
    models = base_models() + [
        {
            "name": "stg_renamed",
            "sql": "select status as state from {{ ref('raw_orders') }}",
            "columns": {"state": "Order status."},
            "parents": ["raw_orders"],
            "folders": ["stg"],
        },
        {
            "name": "silver_copy2",
            "sql": "select * from {{ ref('stg_renamed') }}",
            "columns": {"state": "Order status."},
            "parents": ["stg_renamed"],
            "folders": ["silver"],
        },
    ]
    info = planner_for(models, store).upstream_context("silver_copy2")["state"]
    assert info["chain"] == [
        {"model": "stg_renamed", "column": "state", "kind": "renamed"},
        {"model": "raw_orders", "column": "status", "kind": "root"},
    ]
