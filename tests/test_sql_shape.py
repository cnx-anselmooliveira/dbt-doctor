"""Tests for the structural SQL reader (N1) and the score store."""

import json

import pytest

from dbt_doctor.analyzers.score_store import ScoreStore, description_hash, normalize_description
from dbt_doctor.analyzers.sql_shape import analyze_sql

BRONZE_COPY = """
{{ config(materialized='incremental', unique_key='id') }}
select * from {{ ref('landing_x') }}
{% if is_incremental() %}
where updated_at >= (select max(updated_at) from {{ this }})
{% endif %}
"""


def test_incremental_filter_is_not_a_row_altering_effect():
    shape = analyze_sql(BRONZE_COPY)
    assert shape.kind == "copy"
    assert shape.effects == []
    assert shape.uncertain == []
    assert shape.refs == ["landing_x"]
    assert shape.readable


def test_else_branch_of_incremental_block_is_kept():
    sql = "select * from {{ ref('a') }} {% if is_incremental() %} where 1=1 {% else %} limit 5 {% endif %}"
    assert analyze_sql(sql).effects == ["limit"]


@pytest.mark.parametrize(
    "sql, effect",
    [
        ("select * from {{ ref('a') }} where status = 'x'", "where"),
        ("select a, count(*) as n from {{ ref('a') }} group by a", "group_by"),
        ("select distinct a from {{ ref('a') }}", "distinct"),
        ("select a from {{ ref('a') }} union all select a from {{ ref('b') }}", "union"),
        ("select * from {{ ref('a') }} limit 10", "limit"),
        ("select a from {{ ref('a') }} group by a having count(*) > 1", "having"),
    ],
)
def test_row_altering_effects_are_detected(sql, effect):
    assert effect in analyze_sql(sql).effects


@pytest.mark.parametrize(
    "sql, reason",
    [
        ("select a.id from {{ ref('a') }} a join {{ ref('b') }} b on a.id = b.id", "join"),
        ("with t as (select * from {{ ref('a') }}) select * from t", "cte"),
        ("select * from (select * from {{ ref('a') }}) x", "subquery"),
        ("select {{ my_macro() }} as x from {{ ref('a') }}", "jinja_expression"),
        ("{% for c in cols %} select {{ c }} {% endfor %}", "jinja_control"),
    ],
)
def test_ambiguous_models_are_uncertain_not_readable(sql, reason):
    shape = analyze_sql(sql)
    assert reason in shape.uncertain
    assert not shape.readable


def test_keywords_in_strings_and_comments_are_ignored():
    sql = """
    -- where this is not a filter
    select a, 'a where b' as note /* group by */ from {{ ref('a') }}
    """
    shape = analyze_sql(sql)
    assert shape.effects == [] and shape.readable


def test_select_items_are_classified():
    sql = """
    select
        id,
        t.name,
        status as order_status,
        coalesce(amount, 0) as amount_clean,
        cast(created_at as date) as created_date
    from {{ ref('orders') }} t
    """
    shape = analyze_sql(sql)
    items = {i.output: i for i in shape.items}
    assert shape.kind == "projection" and shape.readable
    assert (items["id"].kind, items["id"].source) == ("passthrough", "id")
    assert (items["name"].kind, items["name"].source) == ("passthrough", "name")
    assert (items["order_status"].kind, items["order_status"].source) == ("renamed", "status")
    assert items["amount_clean"].kind == "transformed"
    assert "coalesce(amount, 0)" in items["amount_clean"].expression
    assert items["created_date"].kind == "transformed"


def test_star_with_extra_computed_columns_is_a_projection():
    shape = analyze_sql("select *, upper(name) as name_upper from {{ ref('a') }}")
    assert shape.kind == "projection"
    assert [i.kind for i in shape.items] == ["star", "transformed"]


def test_from_inside_function_is_not_the_from_clause():
    shape = analyze_sql("select extract(year from created_at) as y, id from {{ ref('a') }}")
    assert [i.output for i in shape.items] == ["y", "id"]


def test_empty_and_garbage_input_never_raise():
    assert analyze_sql("").uncertain == ["empty"]
    assert not analyze_sql("this is not sql").readable
    assert not analyze_sql(None).readable  # type: ignore[arg-type]


def test_source_and_ref_with_package_are_recognised():
    assert analyze_sql("select * from {{ source('s', 't') }}").refs == ["s_t"]
    assert analyze_sql("select * from {{ ref('pkg', 'a') }}").refs == ["a"]


# ── score store ───────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    return ScoreStore(tmp_path / "state" / "scores.json")


def test_lookup_returns_the_recorded_score(store):
    store.record("m", "Col_A", "A description", "fp1", 41, "ok")
    entry = store.lookup("m", "col_a", "A description", "fp1")
    assert entry["total"] == 41 and entry["status"] == "ok"


def test_score_is_invalidated_by_text_or_rubric_changes(store):
    store.record("m", "c", "Original text", "fp1", 41, "ok")
    assert store.lookup("m", "c", "Changed text", "fp1") is None
    assert store.lookup("m", "c", "Original text", "fp2") is None
    assert store.lookup("m", "other", "Original text", "fp1") is None
    assert store.lookup("other", "c", "Original text", "fp1") is None


def test_whitespace_and_folded_yaml_style_do_not_invalidate(store):
    store.record("m", "c", "Line one\nline two  ", "fp1", 40, "ok")
    assert store.lookup("m", "c", "Line one line two", "fp1") is not None
    assert normalize_description("  a \n b ") == "a b"
    assert description_hash("a  b") == description_hash("a b")


def test_record_many_writes_once_and_overwrites(store):
    store.record_many(
        "m",
        "fp1",
        [
            {"column": "a", "description": "x", "total": 40, "status": "ok"},
            {"column": "b", "description": "y", "total": 10, "status": "rewrite"},
        ],
    )
    store.record("m", "a", "x", "fp1", 20, "rewrite")
    assert store.lookup("m", "a", "x", "fp1")["status"] == "rewrite"
    assert store.lookup("m", "b", "y", "fp1")["status"] == "rewrite"
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 1


def test_corrupted_or_missing_store_is_treated_as_empty(store):
    assert store.lookup("m", "c", "x", "fp1") is None
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json", encoding="utf-8")
    assert store.lookup("m", "c", "x", "fp1") is None
    store.record("m", "c", "x", "fp1", 40, "ok")  # recovers by rewriting
    assert store.lookup("m", "c", "x", "fp1") is not None


def test_window_functions_do_not_alter_the_rows():
    shape = analyze_sql("select a, row_number() over (partition by a) as rn from {{ ref('a') }}")
    assert shape.effects == [] and shape.readable
    assert {i.output: i.kind for i in shape.items} == {"a": "passthrough", "rn": "transformed"}


@pytest.mark.parametrize(
    "expr, source",
    [
        ("cast(id as bigint) as id_x", "id"),
        ("try_cast(t.id as varchar) as id_x", "id"),
        ("CAST( id AS decimal(10, 2) ) AS id_x", "id"),
    ],
)
def test_pure_type_cast_is_its_own_kind(expr, source):
    shape = analyze_sql(f"select {expr} from {{{{ ref('a') }}}}")
    item = shape.items[0]
    assert (item.kind, item.source, item.output) == ("cast", source, "id_x")


def test_cast_of_an_expression_or_without_alias_is_not_a_pure_cast():
    shape = analyze_sql("select cast(a + b as bigint) as c, cast(d as date) from {{ ref('a') }}")
    assert [i.kind for i in shape.items] == ["transformed", "transformed"]


@pytest.mark.parametrize("target", ["date", "timestamp(3)", "time", "timestamp with time zone"])
def test_cast_to_a_temporal_type_changes_the_format_so_it_is_a_transformation(target):
    shape = analyze_sql(f"select cast(c as {target}) as d from {{{{ ref('a') }}}}")
    assert shape.items[0].kind == "transformed"
