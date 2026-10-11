# Changelog

All notable changes to `dbt-doctor` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Run history in DuckDB (`.dbt-doctor/dbt_doctor.duckdb`): every tool of the documentation flow records into one run, linked by `run_id`: the batches served, description snapshots, scores per question, plan decisions with their reasons, lineage edges and every write (old and new text, copied, skipped or rejected). New `start_run`, `get_run_report` and `list_runs` tools; `--run-store` and `--no-run-store` options. Recording is best-effort and never breaks a tool; one connection per tool call, a score cache for the planner and a cooldown after a failure keep it fast. The JSON score file is imported once.
- Description consistency across layers, driven by the real dependency graph from `manifest.json` (no `dbt compile` needed):
  - `plan_description_updates`, `trace_column_lineage` and `check_description_consistency` MCP tools. Models are classified from their raw SQL (copy, projection, row-altering effects such as `WHERE`, `GROUP BY`, `DISTINCT`, `UNION`, `LIMIT`; filters inside `is_incremental()` blocks are ignored) and each column is traced to its parent column. The bronze layer is the root; the layer order (`--layer-order`) only labels and groups models.
  - A plain passthrough that keeps the rows must repeat its parent's description identically, and only once the parent's description passed the quality threshold. A transformed column, or one that goes through a filtering or regrouping model, must have its own description (copying the parent's text is rejected as a stale copy). Pure type casts are not enforced by default (`--treat-casts-as-passthrough` copies them too). Unreadable lineage (joins, CTEs, unknown Jinja) is never enforced.
  - Durable score store (`target/dbt_doctor/description_scores.json`): a score is only valid while the description text and the rubric are unchanged.
- `evaluate_model_descriptions` now attaches an `upstream` context to each column (parent model and column, how it is built, the parent's description, the expression that builds it and the chain to the root), so the client LLM judges columns against their lineage instead of in isolation. The review still works if the context cannot be built.
- A column whose parent is a source (not a dbt model), such as a staging model over a bronze source, is described on its own instead of being blocked, since sources are never scored.
- `update_model_yaml` enforces those rules when writing, copies identical parent descriptions (`apply_copies`), and supports `overwrite_descriptions` and `dry_run`.
- `evaluate_model_descriptions` and `submit_description_scores` MCP tools: a description-quality review for one model at a time. The client LLM scores each column against an editable rubric (`description_rubric.json`, overridable with `--rubric-file`); dbt-doctor provides the context and deterministic hints, applies the pass threshold and returns every column with its score in points and a status (`ok`, `rewrite` or `missing`). It only scores; it does not write or suggest descriptions. When the database is connected it also describes the table, so columns missing from `schema.yml` are reported too. Wide tables are split into batches automatically. Read-only: nothing is written to the dbt project.
- `DescriptionQualityEvaluator` analyzer with tests.
- Trino support: new `TrinoConnector` selected when the dbt profile has `type: trino`. Supports `none`, `ldap`/`password` and `jwt` authentication. Install with `pip install "dbt-doctor[trino]"`.
- Tests for `TrinoConnector` (using a fake client) and for adapter selection in `_build_connector`.

### Changed
- `update_model_yaml` now replaces existing column descriptions by default (`overwrite_descriptions=True`) and returns a JSON report (status, file, per-column changes, copied/skipped/rejected columns) instead of a text message. Pass `overwrite_descriptions=False` for the previous fill-only behaviour.
- The data profiler now treats `tinyint` columns as numeric.

## [0.1.0] - 2025-12-23

### Added
- Initial release of the `dbt-doctor` MCP server.
- **Analyzers**:
  - `DataProfiler`: DuckDB-powered fast data profiling for mart tables (row count, distinct count, null rates, cardinality).
  - `ProjectAuditor`: Rules-based auditor for undocumented models, missing descriptions, and naming convention overrides.
  - `DagAnalyzer`: Extract upstream and downstream lineage from the dbt manifest.
  - `TestCoverageAnalyzer`: Calculate coverage score and find uncovered column constraints.
- **Server Tools (MCP)**:
  - `list_models_tool`: Get all models in the project with key metadata.
  - `get_model_details_tool`: Fetch column data, SQL, and description for a specific dbt model.
  - `audit_dbt_project_tool`: Run a full project audit and get a health score.
  - `check_test_coverage_tool`: Get detailed coverage metrics for all models.
  - `analyze_dag_tool`: See model dependencies including parents and children.
  - `get_project_health_summary_tool`: Aggregate metrics (coverage, data profiling, audit score) into one payload.
- **Core Engine**:
  - `ProjectParser`: Lightweight wrapper around `manifest.json` parsing.
  - DuckDB singleton connection manager for zero-copy memory operations.

### fixed
- Replaced redundant string-matching in tests with proper parameterized paths.
