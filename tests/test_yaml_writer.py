"""Tests: YamlWriter — smart merge, comment preservation, new file creation."""

from pathlib import Path

from dbt_doctor.generators.yaml_writer import YamlWriter


def test_creates_new_yaml_file(tmp_path):
    # Create a fake .sql file to establish the path
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "fct_orders.sql").write_text("SELECT 1")

    writer = YamlWriter(str(tmp_path))
    result = writer.upsert_model_yaml(
        model_name="fct_orders",
        model_sql_path="models/fct_orders.sql",
        description="My model description",
        columns=[{"name": "order_id", "description": "PK", "tests": ["not_null", "unique"]}],
    )
    assert "file_path" in result
    assert "error" not in result
    written_path = Path(result["file_path"])
    assert written_path.exists()
    content = written_path.read_text()
    assert "fct_orders" in content
    assert "order_id" in content
    assert "not_null" in content


def test_updates_existing_yaml_preserves_other_models(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "stg_users.sql").write_text("SELECT 1")

    # Write a schema.yml with an existing model
    schema_path = models_dir / "_stg_users.yml"
    schema_path.write_text("""version: 2
models:
  - name: another_model
    description: Should not be touched
  - name: stg_users
    description: ''
""")

    writer = YamlWriter(str(tmp_path))
    result = writer.upsert_model_yaml(
        model_name="stg_users",
        model_sql_path="models/stg_users.sql",
        description="Updated description",
        columns=[{"name": "user_id", "tests": ["not_null"]}],
    )
    assert "file_path" in result
    content = Path(result["file_path"]).read_text()
    # Both models should still be present
    assert "another_model" in content
    assert "Should not be touched" in content
    assert "stg_users" in content
    assert "Updated description" in content


def test_merge_does_not_remove_existing_tests(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "stg_orders.sql").write_text("SELECT 1")

    schema_path = models_dir / "_stg_orders.yml"
    schema_path.write_text("""version: 2
models:
  - name: stg_orders
    columns:
      - name: order_id
        tests:
          - not_null
          - unique
""")

    writer = YamlWriter(str(tmp_path))
    # Add a new test, should not remove existing
    result = writer.upsert_model_yaml(
        model_name="stg_orders",
        model_sql_path="models/stg_orders.sql",
        columns=[{"name": "order_id", "tests": ["relationships"]}],
    )
    content = Path(result["file_path"]).read_text()
    assert "not_null" in content
    assert "unique" in content
    assert "relationships" in content


def test_adds_new_column_to_existing(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "stg_orders.sql").write_text("SELECT 1")

    schema_path = models_dir / "_stg_orders.yml"
    schema_path.write_text("""version: 2
models:
  - name: stg_orders
    columns:
      - name: order_id
        tests: [not_null]
""")

    writer = YamlWriter(str(tmp_path))
    result = writer.upsert_model_yaml(
        model_name="stg_orders",
        model_sql_path="models/stg_orders.sql",
        columns=[{"name": "status", "tests": ["accepted_values"]}],
    )
    content = Path(result["file_path"]).read_text()
    assert "order_id" in content  # old column preserved
    assert "status" in content  # new column added


# ── overwrite and dry run ─────────────────────────────────────────────────

EXISTING_YAML = """\
version: 2

models:
  - name: fct_orders
    description: Old model text
    columns:
      # keep this comment
      - name: order_id
        description: Old order id text
        tests:
          - not_null
      - name: status
        description: ""
"""


def _project_with_yaml(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "fct_orders.sql").write_text("SELECT 1")
    yaml_file = models_dir / "_fct_orders.yml"
    yaml_file.write_text(EXISTING_YAML, encoding="utf-8")
    return YamlWriter(str(tmp_path)), yaml_file


def _upsert(writer, **kwargs):
    return writer.upsert_model_yaml(
        model_name="fct_orders",
        model_sql_path="models/fct_orders.sql",
        columns=[
            {"name": "order_id", "description": "New order id text", "tests": ["unique"]},
            {"name": "status", "description": "Order status"},
            {"name": "amount", "description": "Order amount"},
        ],
        **kwargs,
    )


def test_existing_descriptions_are_kept_by_default_and_empty_ones_filled(tmp_path):
    writer, yaml_file = _project_with_yaml(tmp_path)
    result = _upsert(writer)
    content = yaml_file.read_text(encoding="utf-8")
    assert "Old order id text" in content and "New order id text" not in content
    assert "Order status" in content and "Order amount" in content
    assert {c["column"] for c in result["changes"]} == {"status", "amount"}


def test_overwrite_replaces_existing_descriptions_and_keeps_tests_and_comments(tmp_path):
    writer, yaml_file = _project_with_yaml(tmp_path)
    result = _upsert(writer, overwrite_descriptions=True)
    content = yaml_file.read_text(encoding="utf-8")
    assert "New order id text" in content and "Old order id text" not in content
    assert "# keep this comment" in content
    assert "not_null" in content and "unique" in content  # tests are only ever added
    change = next(c for c in result["changes"] if c["column"] == "order_id")
    assert (change["old"], change["new"]) == ("Old order id text", "New order id text")
    assert result["dry_run"] is False


def test_dry_run_reports_changes_without_touching_the_file(tmp_path):
    writer, yaml_file = _project_with_yaml(tmp_path)
    result = _upsert(writer, overwrite_descriptions=True, dry_run=True)
    assert yaml_file.read_text(encoding="utf-8") == EXISTING_YAML
    assert result["dry_run"] is True and len(result["changes"]) == 3


def test_identical_text_is_not_reported_as_a_change(tmp_path):
    writer, _ = _project_with_yaml(tmp_path)
    result = writer.upsert_model_yaml(
        model_name="fct_orders",
        model_sql_path="models/fct_orders.sql",
        description="Old  model text",
        columns=[{"name": "order_id", "description": "Old order id\ntext"}],
        overwrite_descriptions=True,
        dry_run=True,
    )
    assert result["changes"] == []
