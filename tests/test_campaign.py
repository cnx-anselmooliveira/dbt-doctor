"""Tests for the campaign engine (phase A): table health, the work list and its rules."""

import pytest

from dbt_doctor.analyzers import campaign as camp
from dbt_doctor.analyzers.campaign import (
    CampaignEngine,
    CampaignError,
    table_health,
    validate_tables,
)
from dbt_doctor.analyzers.lineage import LineageGraph, LineagePlanner
from dbt_doctor.analyzers.score_store import ScoreStore
from dbt_doctor.core.run_store import RunStore

FP = "fp1"
ID = "Identifier of the record."
NAME = "Full name of the record."
UPPER = "Upper-cased full name."
COPY = "select * from {{ ref('%s') }}"


def model(name, folders, sql, columns, parents=()):
    return {"name": name, "folders": folders, "sql": sql, "columns": columns, "parents": parents}


def chain():
    return [
        model(
            "landing_a", ["landing"], "select * from {{ source('s', 'a') }}", {"id": "landing id"}
        ),
        model("raw_a", ["bronze"], COPY % "landing_a", {"id": ID, "name": NAME}, ["landing_a"]),
        model("stg_a", ["stg"], COPY % "raw_a", {"id": ID, "name": NAME}, ["raw_a"]),
        model(
            "silver_a",
            ["silver"],
            "select id, upper(name) as name_up from {{ ref('stg_a') }}",
            {"id": ID, "name_up": UPPER},
            ["stg_a"],
        ),
        model("gold_a", ["gold"], COPY % "silver_a", {"id": ID, "name_up": UPPER}, ["silver_a"]),
    ]


def manifest_for(models):
    nodes = {}
    for m in models:
        nodes[f"model.proj.{m['name']}"] = {
            "resource_type": "model",
            "name": m["name"],
            "fqn": ["proj", *m["folders"], m["name"]],
            "raw_code": m["sql"],
            "columns": {c: {"description": d} for c, d in m["columns"].items()},
            "depends_on": {"nodes": [f"model.proj.{p}" for p in m["parents"]]},
        }
    return {"nodes": nodes, "sources": {}}


class Env:
    """A campaign over a synthetic project, with scores kept in a JSON store."""

    def __init__(self, tmp_path, models=None, seeds=("stg_a",), **config):
        self.tmp_path = tmp_path
        self.config = {"pass_ratio": 0.7, "children_scope": "needing_docs", **config}
        self.scores = ScoreStore(tmp_path / "scores.json")
        self.rs = RunStore(tmp_path / "runs.duckdb")
        self.run_id = self.rs.create_run("g", "", None, FP, {"campaign": self.config})
        self.engine = self.build(models or chain())
        self.engine.seed(list(seeds))

    def build(self, models):
        graph = LineageGraph(manifest_for(models))
        planner = LineagePlanner(graph, self.scores, FP)
        self.engine = CampaignEngine(graph, planner, self.rs, self.run_id, self.config)
        return self.engine

    def score(self, name, status="ok", columns=None):
        graph = self.engine.graph
        model_info = graph.models[graph.by_name[name]]
        for column, text in model_info.columns.items():
            if columns is None or column in columns:
                self.scores.record(name, column, text, FP, 40 if status == "ok" else 10, status)

    def advance(self, limit=10):
        return self.engine.advance(limit)

    def items(self):
        return {r["model"]: r for r in self.rs.load_work_items(self.run_id)}

    def statuses(self):
        return {n: i["status"] for n, i in self.items().items()}


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# ── seeds ─────────────────────────────────────────────────────────────────


def test_unknown_tables_are_rejected_with_suggestions():
    graph = LineageGraph(manifest_for(chain()))
    with pytest.raises(CampaignError, match=r"'stg_b' \(did you mean stg_a\?\)"):
        validate_tables(graph, ["stg_a", "stg_b"])
    with pytest.raises(CampaignError, match="at least one"):
        validate_tables(graph, ["", "  "])
    assert validate_tables(graph, [" stg_a ", "stg_a", "raw_a"]) == ["stg_a", "raw_a"]


def test_a_landing_seed_is_kept_in_the_list_but_deferred(tmp_path):
    env = Env(tmp_path, seeds=["landing_a", "raw_a"])
    work = env.advance()
    assert env.statuses()["landing_a"] == camp.DEFERRED
    assert [t["model"] for t in work["tasks"]] == ["raw_a"]


# ── table health ──────────────────────────────────────────────────────────


def test_health_counts_inherited_columns_as_ok(env):
    env.score("raw_a")
    health = table_health(env.engine.planner, "stg_a", 0.7)
    assert health.status == camp.PASSED and health.ratio == 1.0
    assert health.counts == {"ok_inherited": 2}


def test_inheritance_needs_the_parent_to_have_passed_and_the_same_text(tmp_path):
    env = Env(tmp_path)
    health = table_health(env.engine.planner, "stg_a", 0.7)
    assert health.counts == {"waiting_parent": 2} and health.blockers == ["raw_a"]
    env.score("raw_a", "rewrite")
    assert table_health(env.engine.planner, "stg_a", 0.7).counts == {"parent_failed": 2}

    models = chain()
    models[2]["columns"]["name"] = "A different wording."
    env.build(models)
    env.score("raw_a")
    health = table_health(env.engine.planner, "stg_a", 0.7)
    assert health.counts == {"ok_inherited": 1, "needs_copy": 1}
    assert health.status == camp.FAILED_SELF  # 1 of 2 ok is below 70%


def test_own_columns_need_a_score_and_a_table_with_unscored_columns_is_untested(env):
    health = table_health(env.engine.planner, "raw_a", 0.7)
    assert (health.status, health.counts) == (camp.NEEDS_TEST, {"unscored": 2})
    env.score("raw_a", columns=["id"])
    assert table_health(env.engine.planner, "raw_a", 0.7).counts == {"ok_own": 1, "unscored": 1}


def test_the_pass_ratio_decides_between_pass_and_fail(env):
    env.score("raw_a", columns=["id"])
    env.score("raw_a", "rewrite", columns=["name"])
    assert table_health(env.engine.planner, "raw_a", 0.5).status == camp.PASSED
    assert table_health(env.engine.planner, "raw_a", 0.7).status == camp.FAILED_SELF
    assert table_health(env.engine.planner, "raw_a", 1.0).status == camp.FAILED_SELF


def test_empty_and_stale_copy_columns_fail(env):
    models = chain()
    models[3]["columns"] = {"id": ID, "name_up": ""}
    env.build(models)
    env.score("raw_a")
    env.score("stg_a")
    health = table_health(env.engine.planner, "silver_a", 0.7)
    assert health.counts == {"ok_inherited": 1, "missing": 1}
    assert health.status == camp.FAILED_SELF  # 1 of 2 is below 70%

    models = chain()
    models[3]["sql"] = "select id, upper(name) as name from {{ ref('stg_a') }}"
    models[3]["columns"] = {"id": ID, "name": NAME}  # transformed, yet the parent's text
    env.build(models)
    assert table_health(env.engine.planner, "silver_a", 0.7).counts["stale_copy"] == 1


def test_a_table_without_documented_columns_fails(tmp_path):
    models = chain() + [model("az_empty", ["az"], COPY % "silver_a", {}, ["silver_a"])]
    env = Env(tmp_path, models=models, seeds=["az_empty"])
    health = table_health(env.engine.planner, "az_empty", 0.7)
    assert (health.status, health.reason) == (camp.FAILED_SELF, "no documented columns")


# ── the work list ─────────────────────────────────────────────────────────


def test_a_table_blocked_by_an_untested_parent_pulls_the_parent_in(env):
    work = env.advance()
    assert [t["model"] for t in work["tasks"]] == ["raw_a"]
    assert work["tasks"][0]["why"] == "parent of stg_a"
    assert env.statuses()["stg_a"] == camp.BLOCKED_BY_PARENT
    assert env.items()["raw_a"]["role"] == "ancestor" and env.items()["raw_a"]["depth"] == 1
    assert work["phase"] == "A_in_progress" and work["remaining_to_test"] == 1


def test_parents_are_followed_recursively_in_a_single_call(tmp_path):
    env = Env(tmp_path, seeds=["silver_a"])
    assert [t["model"] for t in env.advance()["tasks"]] == ["silver_a"]  # own column unscored
    env.score("silver_a", columns=["name_up"])
    work = env.advance()
    statuses = env.statuses()
    assert statuses["silver_a"] == camp.BLOCKED_BY_PARENT
    assert statuses["stg_a"] == camp.BLOCKED_BY_PARENT
    assert [t["model"] for t in work["tasks"]] == ["raw_a"]  # the first table that can be tested
    assert env.items()["raw_a"]["depth"] == 2


def test_the_walk_stops_at_a_parent_that_passes(env):
    env.advance()
    env.score("raw_a")
    work = env.advance()
    assert work["phase"] == "A_complete" and work["tasks"] == []
    assert env.statuses()["stg_a"] == camp.PASSED and env.statuses()["raw_a"] == camp.PASSED
    assert "landing_a" not in env.items()  # nothing was pulled in above a passing parent


def test_a_failing_bronze_table_keeps_landing_in_the_list_as_deferred(env):
    env.advance()
    env.score("raw_a", "rewrite")
    work = env.advance()
    statuses = env.statuses()
    assert statuses["raw_a"] == camp.FAILED_SELF and statuses["landing_a"] == camp.DEFERRED
    assert statuses["stg_a"] == camp.BLOCKED_BY_PARENT
    assert work["phase"] == "A_complete" and work["tasks"] == []
    assert env.items()["landing_a"]["reason"] == "landing is never processed"


def test_only_the_direct_children_are_listed_and_never_tested(env):
    env.advance()
    items = env.items()
    assert {
        n: (i["role"], i["status"], i["depth"])
        for n, i in items.items()
        if i["role"] == "descendant"
    } == {"silver_a": ("descendant", camp.UNTESTED_CHILD, 1)}
    assert "gold_a" not in items  # a grandchild: it only enters once its parent is updated
    assert all(t["model"] != "silver_a" for t in env.advance()["tasks"])


def test_children_scopes(tmp_path):
    models = chain() + [model("stg_sibling", ["stg"], COPY % "raw_a", {"id": ID}, ["raw_a"])]

    def children(env):
        return {n for n, i in env.items().items() if i["role"] == "descendant"}

    docs = Env(tmp_path / "d", models=models, seeds=["stg_a"])  # needing_docs: the default
    docs.advance()
    docs.score("raw_a", "rewrite")
    docs.advance()
    assert children(docs) == {"silver_a", "stg_sibling"}  # raw_a failed: its children are added too

    seeds = Env(tmp_path / "s", models=models, seeds=["stg_a"], children_scope="seeds")
    seeds.advance()
    seeds.score("raw_a", "rewrite")
    seeds.advance()
    assert children(seeds) == {"silver_a"}  # only the listed tables' children

    none = Env(tmp_path / "n", children_scope="none")
    none.advance()
    assert children(none) == set()


def test_children_of_a_parent_that_passes_are_not_added(tmp_path):
    models = chain() + [model("stg_sibling", ["stg"], COPY % "raw_a", {"id": ID}, ["raw_a"])]
    env = Env(tmp_path, models=models, seeds=["stg_a"])
    env.advance()
    env.score("raw_a")
    env.advance()
    assert "stg_sibling" not in env.items()  # raw_a passed: it is context, not work


def test_a_child_that_becomes_the_parent_of_a_failing_table_is_upgraded(tmp_path):
    models = chain() + [
        model(
            "az_b",
            ["az"],
            "select a.id from {{ ref('silver_a') }} a join {{ ref('stg_a') }} b on a.id = b.id",
            {"id": "Something."},
            ["silver_a", "stg_a"],
        )
    ]
    env = Env(tmp_path, models=models, seeds=["stg_a", "az_b"])
    env.advance()
    assert env.items()["silver_a"]["role"] == "descendant"
    env.score("az_b", "rewrite")
    env.advance()
    item = env.items()["silver_a"]
    assert (item["role"], item["status"]) == ("ancestor", camp.NEEDS_TEST)


def test_nothing_is_tested_twice(env):
    env.advance()
    env.score("raw_a")
    env.advance()
    again = env.advance()
    assert again["tasks"] == [] and again["phase"] == "A_complete"
    # Editing a description invalidates its score: the table is tested again.
    models = chain()
    models[1]["columns"]["id"] = "Rewritten identifier."
    env.build(models)
    work = env.advance()
    assert [t["model"] for t in work["tasks"]] == ["raw_a"]
    assert work["tasks"][0]["columns_without_score"] == 1


def test_advance_is_idempotent_and_the_list_only_grows(env):
    first = env.advance()
    members = set(env.items())
    second = env.advance()
    assert set(env.items()) == members
    assert [t["model"] for t in first["tasks"]] == [t["model"] for t in second["tasks"]]
    assert env.advance()["counts"] == second["counts"]


def test_tasks_are_ordered_parents_first_and_limited(tmp_path):
    models = chain() + [model("raw_b", ["bronze"], COPY % "landing_a", {"id": ID}, ["landing_a"])]
    env = Env(tmp_path, models=models, seeds=["raw_b", "raw_a", "silver_a"])
    names = [t["model"] for t in env.advance()["tasks"]]
    assert names.index("raw_a") < names.index("silver_a")
    assert [t["model"] for t in env.advance(limit=1)["tasks"]] == [names[0]]


def test_asking_again_without_progress_ends_in_an_error(env):
    for _ in range(camp.MAX_ATTEMPTS + 1):
        work = env.advance()
    assert env.statuses()["raw_a"] == camp.ERROR
    assert "no progress" in env.items()["raw_a"]["reason"]
    assert work["tasks"] == [] and work["phase"] == "A_complete"


def test_progress_resets_the_patience(tmp_path):
    env = Env(tmp_path, seeds=["raw_a"])
    for column in ("id", "name"):
        for _ in range(camp.MAX_ATTEMPTS - 1):
            env.advance()
        env.score("raw_a", columns=[column])  # progress between the requests
    env.advance()
    assert env.statuses()["raw_a"] == camp.PASSED


def test_skip_removes_a_table_from_testing_and_expansion(env):
    env.advance()
    result = env.engine.skip("raw_a", "owned by another team")
    assert result["status"] == camp.SKIPPED
    work = env.advance()
    assert work["tasks"] == [] and env.statuses()["raw_a"] == camp.SKIPPED
    with pytest.raises(CampaignError, match="not in this campaign"):
        env.engine.skip("ghost", "")


def test_a_table_that_disappears_from_the_manifest_becomes_an_error(env):
    env.advance()
    env.build([m for m in chain() if m["name"] != "raw_a"])
    env.advance()
    assert env.statuses()["raw_a"] == camp.ERROR
    assert "not found in the manifest" in env.items()["raw_a"]["reason"]


# ── status ────────────────────────────────────────────────────────────────


def test_status_lists_everything_parents_first_with_paging_and_filter(env):
    env.advance()
    env.score("raw_a", "rewrite")
    env.advance()
    status = env.engine.status(limit=3)
    assert status["total"] == 4 and status["phase"] == "A_complete"
    assert [t["model"] for t in status["tables"]][:2] == ["landing_a", "raw_a"]
    assert status["next_offset"] == 3
    assert env.engine.status(offset=3, limit=3)["next_offset"] is None
    assert status["by_role"] == {"seed": 1, "ancestor": 2, "descendant": 1}
    only = env.engine.status(status=camp.UNTESTED_CHILD)
    assert only["matching"] == 1 and {t["model"] for t in only["tables"]} == {"silver_a"}


def test_an_empty_campaign_cannot_advance(tmp_path):
    env = Env(tmp_path)
    env.rs.query("SELECT 1")
    other = env.rs.create_run("other", "", None, FP, {})
    engine = CampaignEngine(env.engine.graph, env.engine.planner, env.rs, other, env.config)
    with pytest.raises(CampaignError, match="no tables"):
        engine.advance()


# ── children are tested only when a direct parent was updated ─────────────


def test_table_checksum_changes_only_when_a_description_does(env):
    graph = env.engine.graph
    base = camp.table_checksum(graph, "raw_a")
    assert base == camp.table_checksum(graph, "raw_a")
    models = chain()
    models[1]["columns"] = {"name": "Full  name of the\nrecord.", "id": ID}  # order, spacing
    other = LineageGraph(manifest_for(models))
    assert camp.table_checksum(other, "raw_a") == base
    models[1]["columns"]["id"] = "A different identifier."
    assert camp.table_checksum(LineageGraph(manifest_for(models)), "raw_a") != base
    assert camp.table_checksum(graph, "stg_a") != "" and len(base) == 16


def test_a_child_is_tested_only_after_a_direct_parent_is_updated(tmp_path):
    env = Env(tmp_path, seeds=["raw_a"])
    env.score("raw_a")
    env.advance()
    assert env.statuses() == {"raw_a": camp.PASSED, "stg_a": camp.UNTESTED_CHILD}
    assert env.advance()["tasks"] == []  # nothing changed: the child stays untested

    models = chain()
    models[1]["columns"]["id"] = "A rewritten identifier."  # raw_a, the parent, was updated
    env.build(models)
    env.score("raw_a")
    env.advance()
    child = env.items()["stg_a"]
    assert child["summary"]["promoted"] is True
    assert child["status"] == camp.FAILED_SELF  # its copy of `id` no longer matches the parent
    assert "pending_seen" not in child["summary"]  # acknowledged: it will not trigger again
    env.advance()
    assert env.items()["stg_a"]["summary"] == child["summary"]  # stable on the next call


def test_a_change_in_an_unrelated_table_promotes_nothing(tmp_path):
    env = Env(tmp_path, seeds=["raw_a"])
    env.score("raw_a")
    env.advance()
    models = chain()
    models[4]["columns"]["id"] = "Edited far away."  # gold_a is nobody's parent here
    env.build(models)
    env.advance()
    assert env.statuses()["stg_a"] == camp.UNTESTED_CHILD


def test_the_list_grows_one_level_at_a_time_as_updates_arrive(tmp_path):
    env = Env(tmp_path, seeds=["raw_a"])
    env.score("raw_a")
    env.advance()
    assert set(env.items()) == {"raw_a", "stg_a"}

    models = chain()
    models[1]["columns"]["id"] = "A rewritten identifier."
    env.build(models)
    env.score("raw_a")
    env.advance()
    # stg_a was promoted and fails, so its own direct children are listed (untested)
    assert env.statuses()["stg_a"] == camp.FAILED_SELF
    assert env.items()["silver_a"]["status"] == camp.UNTESTED_CHILD
    assert env.items()["silver_a"]["depth"] == 2 and "gold_a" not in env.items()

    models[2]["columns"]["id"] = "A rewritten identifier."  # now stg_a is updated too
    env.build(models)
    env.advance()
    assert env.items()["silver_a"]["summary"]["promoted"] is True


def test_a_parent_updated_while_the_child_is_waiting_for_a_test_is_not_lost(tmp_path):
    env = Env(tmp_path, seeds=["raw_a"])
    env.score("raw_a")
    env.advance()
    models = chain()
    models[1]["columns"]["name"] = "Another full name."
    env.build(models)
    env.score("raw_a")
    env.advance()
    seen_then = env.items()["stg_a"]["summary"]["parents_seen"]["raw_a"]
    models[1]["columns"]["name"] = "Yet another full name."
    env.build(models)
    env.score("raw_a")
    env.advance()
    assert env.items()["stg_a"]["summary"]["parents_seen"]["raw_a"] != seen_then
