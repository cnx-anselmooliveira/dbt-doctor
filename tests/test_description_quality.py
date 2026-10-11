"""Tests for the description-quality analyzer (rubric, signals, batching, scoring)."""

import json

import pytest

from dbt_doctor.analyzers import description_quality as dq
from dbt_doctor.analyzers.description_quality import (
    DescriptionQualityEvaluator,
    RubricError,
    ScoreValidationError,
    compute_signals,
    extract_complex_expression,
    load_rubric,
)

# ── rubric ────────────────────────────────────────────────────────────────


def test_default_rubric_has_five_questions_and_threshold_35():
    rubric = load_rubric()
    assert [q.id for q in rubric.questions] == [
        "tautology",
        "unit_format",
        "grain",
        "sql_logic",
        "business_clarity",
    ]
    assert rubric.max_total == 50
    assert rubric.threshold == 35
    assert rubric.passes(35)
    assert not rubric.passes(34.9)


def _write_rubric(tmp_path, **overrides):
    data = {
        "version": 1,
        "pass_ratio": 0.7,
        "max_score_per_question": 10,
        "questions": [{"id": "a", "text": "A?"}, {"id": "b", "text": "B?"}],
    }
    data.update(overrides)
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_custom_rubric_threshold_follows_number_of_questions(tmp_path):
    rubric = load_rubric(_write_rubric(tmp_path))
    assert rubric.max_total == 20
    assert rubric.threshold == 14


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pass_ratio": 0}, "pass_ratio"),
        ({"pass_ratio": 1.5}, "pass_ratio"),
        ({"max_score_per_question": 0}, "max_score_per_question"),
        ({"questions": []}, "questions"),
        ({"questions": [{"id": "A b", "text": "x"}]}, "invalid 'id'"),
        ({"questions": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]}, "Duplicate"),
        ({"questions": [{"id": "a", "text": " "}]}, "non-empty 'text'"),
    ],
)
def test_invalid_rubric_is_rejected(tmp_path, overrides, message):
    with pytest.raises(RubricError, match=message):
        load_rubric(_write_rubric(tmp_path, **overrides))


def test_missing_and_malformed_rubric_files(tmp_path):
    with pytest.raises(RubricError, match="Cannot read"):
        load_rubric(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RubricError, match="not valid JSON"):
        load_rubric(bad)


# ── signals ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "column, description, expected",
    [
        ("customer_id", "", "empty_description"),
        ("customer_id", "Customer ID", "repeats_column_name"),
        ("customer_id", "The customer id", "repeats_column_name"),
        ("customerid", "Customer ID", "repeats_column_name"),
        ("status", "Order status", "very_short"),
    ],
)
def test_weak_description_signals(column, description, expected):
    assert expected in compute_signals(column, None, description, None)


def test_good_description_fires_no_text_signals():
    signals = compute_signals(
        "customer_id", None, "Identifier of the customer who placed the order", None
    )
    assert signals == {}


@pytest.mark.parametrize(
    "column, data_type, description, kind",
    [
        ("order_amount", None, "Total value of the order", "currency"),
        ("created_at", None, "When the order was created", "timestamp"),
        ("birth_date", None, "Customer birth", "date"),
        ("is_active", None, "State of the customer account today", "boolean"),
        ("discount_pct", None, "Discount applied to the order", "percentage"),
        ("delivery_seconds", None, "Time taken to deliver the parcel", "duration"),
        ("flag_x", "boolean", "Marks the special customers here", "boolean"),
    ],
)
def test_unit_or_format_missing(column, data_type, description, kind):
    signals = compute_signals(column, data_type, description, None)
    assert kind in signals.get("unit_or_format_not_stated", [])


@pytest.mark.parametrize(
    "column, description",
    [
        ("order_amount", "Order total in BRL, including taxes"),
        ("created_at", "Creation time of the order in UTC"),
        ("is_active", "True when the account is active, false otherwise"),
    ],
)
def test_unit_or_format_stated_does_not_fire(column, description):
    assert "unit_or_format_not_stated" not in compute_signals(column, None, description, None)


SQL = """
with base as (select order_id, status from raw.orders)
select
    order_id,
    case when status = 'x' then 'cancelled' else status end as order_status,
    coalesce(amount, 0) as amount_clean,
    sum(amount) over (partition by customer_id) as customer_total,
    status as plain_status
from base
"""


@pytest.mark.parametrize(
    "column, fragment",
    [
        ("order_status", "case when status"),
        ("amount_clean", "coalesce(amount, 0)"),
        ("customer_total", "over (partition by customer_id)"),
    ],
)
def test_extract_complex_expression(column, fragment):
    assert fragment in extract_complex_expression(SQL, column)


@pytest.mark.parametrize("column", ["plain_status", "order_id", "missing_col"])
def test_plain_passthrough_has_no_complex_expression(column):
    assert extract_complex_expression(SQL, column) is None


def test_expression_in_signals():
    assert "complex_sql_expression" in compute_signals("order_status", None, "x y z", SQL)


# ── batching ──────────────────────────────────────────────────────────────


def make_details(n_columns=3, description="Identifier of the thing that is described"):
    columns = {
        f"col_{i}": {"description": description, "tests": ["unique"] if i == 0 else []}
        for i in range(n_columns)
    }
    return {
        "name": "my_model",
        "description": "A model",
        "schema": "marts",
        "compiled_sql": "select 1 as col_0",
        "raw_sql": "select 1 as col_0",
        "columns": columns,
    }


@pytest.fixture
def evaluator(tmp_path):
    return DescriptionQualityEvaluator(load_rubric(), tmp_path / "state")


def perfect(columns, total=10, suggestion=None):
    ids = [q.id for q in load_rubric().questions]
    return [
        {
            "column": c,
            "scores": {i: total for i in ids},
            **({"suggested_description": suggestion} if suggestion else {}),
        }
        for c in columns
    ]


def test_small_table_fits_in_one_batch(evaluator):
    batch = evaluator.next_batch(make_details(3))
    assert [c["name"] for c in batch["columns"]] == ["col_0", "col_1", "col_2"]
    assert batch["next_offset"] is None
    assert batch["unique_columns"] == ["col_0"]
    assert batch["rubric"]["pass_total"] == 35
    assert batch["sql"]["kind"] == "compiled"


def test_wide_table_is_split_by_budget_and_offsets_chain(evaluator):
    details = make_details(60, description="x " * 40)
    first = evaluator.next_batch(details, max_chars=6000)
    assert 5 <= len(first["columns"]) < 60
    assert first["next_offset"] == len(first["columns"])

    seen = [c["name"] for c in first["columns"]]
    offset = first["next_offset"]
    while offset is not None:
        batch = evaluator.next_batch(details, offset=offset, max_chars=6000)
        seen += [c["name"] for c in batch["columns"]]
        offset = batch["next_offset"]
    assert seen == list(details["columns"])


def test_minimum_columns_per_batch_even_with_tiny_budget(evaluator):
    batch = evaluator.next_batch(make_details(20), max_chars=10)
    assert len(batch["columns"]) == dq.MIN_COLUMNS_PER_BATCH


def test_batch_errors(evaluator):
    with pytest.raises(ValueError, match="offset"):
        evaluator.next_batch(make_details(3), offset=3)
    with pytest.raises(ValueError, match="offset"):
        evaluator.next_batch(make_details(3), offset=-1)
    with pytest.raises(ValueError, match="no columns"):
        evaluator.next_batch(make_details(0))


def test_large_sql_is_omitted_and_raw_sql_is_labelled(evaluator):
    details = make_details(2)
    details["compiled_sql"] = "select 1 as x -- " + "a" * dq.SQL_INLINE_LIMIT
    batch = evaluator.next_batch(details)
    assert batch["sql"]["code"] is None and "omitted" in batch["sql"]["kind"]

    details = make_details(2)
    details["compiled_sql"] = ""
    assert "raw" in evaluator.next_batch(details)["sql"]["kind"]

    details["raw_sql"] = ""
    assert evaluator.next_batch(details)["sql"] == {"kind": None, "code": None}


# ── scoring ───────────────────────────────────────────────────────────────


def test_pass_and_rewrite_at_threshold_boundary(evaluator):
    details = make_details(2)
    ids = [q.id for q in load_rubric().questions]
    scores = [
        {"column": "col_0", "scores": dict.fromkeys(ids, 7)},  # 35 -> ok
        {
            "column": "col_1",
            "scores": {**dict.fromkeys(ids, 7), "grain": 6},  # 34 -> rewrite
            "suggested_description": "Better text",
        },
    ]
    result = evaluator.submit(details, scores)
    by_col = {c["column"]: c for c in result["columns"]}
    assert result["status"] == "complete"
    assert by_col["col_0"]["status"] == "ok" and by_col["col_0"]["total"] == 35
    assert by_col["col_1"]["status"] == "rewrite"
    assert by_col["col_1"]["suggested_description"] == "Better text"
    assert result["to_rewrite"] == 1


def test_progress_across_batches_and_state_file_lifecycle(evaluator, tmp_path):
    details = make_details(3)
    state_file = tmp_path / "state" / "description_eval_my_model.json"

    progress = evaluator.submit(details, perfect(["col_0"]))
    assert progress == {
        "status": "in_progress",
        "model": "my_model",
        "evaluated": 1,
        "total_columns": 3,
        "next_offset": 1,
    }
    assert state_file.exists()

    # Resending the same column is idempotent.
    assert evaluator.submit(details, perfect(["col_0"]))["evaluated"] == 1

    done = evaluator.submit(details, perfect(["col_1", "col_2"]))
    assert done["status"] == "complete"
    assert [c["column"] for c in done["columns"]] == ["col_0", "col_1", "col_2"]
    assert not state_file.exists()


def test_offset_zero_discards_unfinished_evaluation(evaluator, tmp_path):
    details = make_details(3)
    evaluator.submit(details, perfect(["col_0"]))
    state_file = tmp_path / "state" / "description_eval_my_model.json"
    assert state_file.exists()
    evaluator.next_batch(details, offset=0)
    assert not state_file.exists()
    assert evaluator.submit(details, perfect(["col_1"]))["evaluated"] == 1


def test_editing_the_rubric_discards_earlier_scores(tmp_path):
    details = make_details(2)
    state_dir = tmp_path / "state"
    DescriptionQualityEvaluator(load_rubric(), state_dir).submit(details, perfect(["col_0"]))

    changed = DescriptionQualityEvaluator(load_rubric(_write_rubric(tmp_path)), state_dir)
    progress = changed.submit(
        details,
        [{"column": "col_1", "scores": {"a": 10, "b": 10}}],
    )
    assert progress["evaluated"] == 1  # col_0 from the old rubric was dropped


@pytest.mark.parametrize(
    "scores, message",
    [
        ([], "non-empty list"),
        ("nope", "non-empty list"),
        (["x"], "must be an object"),
        ([{"column": "ghost", "scores": {}}], "unknown column 'ghost'"),
        ([{"column": "col_0"}], "'scores' must be an object"),
        ([{"column": "col_0", "scores": {"tautology": 5}}], "missing"),
        (
            [
                {
                    "column": "col_0",
                    "scores": {
                        **dict.fromkeys(
                            ["tautology", "unit_format", "grain", "sql_logic", "business_clarity"],
                            5,
                        ),
                        "extra": 1,
                    },
                }
            ],
            "unknown \\['extra'\\]",
        ),
        (
            [
                {
                    "column": "col_0",
                    "scores": dict.fromkeys(
                        ["tautology", "unit_format", "grain", "sql_logic", "business_clarity"], 11
                    ),
                }
            ],
            "from 0 to 10",
        ),
        (
            [
                {
                    "column": "col_0",
                    "scores": dict.fromkeys(
                        ["tautology", "unit_format", "grain", "sql_logic", "business_clarity"], "8"
                    ),
                }
            ],
            "from 0 to 10",
        ),
        (
            [
                {
                    "column": "col_0",
                    "scores": dict.fromkeys(
                        ["tautology", "unit_format", "grain", "sql_logic", "business_clarity"], 1
                    ),
                }
            ],
            "suggested_description",
        ),
    ],
)
def test_invalid_scores_are_rejected(evaluator, scores, message):
    with pytest.raises(ScoreValidationError, match=message):
        evaluator.submit(make_details(2), scores)


def test_rejected_submit_records_nothing(evaluator, tmp_path):
    details = make_details(2)
    good = perfect(["col_0"])
    bad = [{"column": "ghost", "scores": {}}]
    with pytest.raises(ScoreValidationError):
        evaluator.submit(details, good + bad)
    assert not (tmp_path / "state" / "description_eval_my_model.json").exists()


def test_duplicate_column_in_one_call_is_rejected(evaluator):
    with pytest.raises(ScoreValidationError, match="more than once"):
        evaluator.submit(make_details(2), perfect(["col_0", "col_0"]))
