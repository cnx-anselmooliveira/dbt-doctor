<div align="center">
  <img src="logo.png" alt="dbt-doctor logo" width="300"/>
  <p><strong>AI-driven quality and governance MCP Server for dbt projects.</strong></p>
  <p>Audit coverage, profile data, detect schema drift, and auto-generate documentation—all through natural language with an AI assistant.</p>
</div>

<p align="center">
  <a href="https://github.com/Astoriel/dbt-doctor/actions/workflows/ci.yml">
    <img src="https://github.com/Astoriel/dbt-doctor/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <a href="https://pypi.org/project/dbt-doctor/">
    <img src="https://badge.fury.io/py/dbt-doctor.svg" alt="PyPI version"/>
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
  </a>
</p>

---

## Project status

Active alpha. Snapshot date: 2025-12-25. See [STATUS.md](STATUS.md) and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for what is working today, what is planned, and what is not claimed.

## What is dbt-doctor?

dbt-doctor is a Model Context Protocol (MCP) server that provides your AI coding assistant with deep context regarding your dbt project's health. Instead of manually running CLI commands and analyzing outputs, you can interact with your AI:

- "What's the overall health of my dbt project?"
- "Profile the fct_orders model and suggest appropriate tests."
- "Auto-document the models that have the lowest test coverage."

The tool handles the heavy operations—reading the manifest, profiling your data warehouse, detecting schema drift, and writing back to schema.yml files—without requiring you to leave the chat.

Note: This tool is designed to complement the official dbt-labs/dbt-mcp. While dbt-labs/dbt-mcp focuses on running dbt commands, dbt-doctor focuses on auditing, profiling, and documentation.

## Key Features

### Project Auditing
Evaluate your project with a 0–100% score based on documentation, testing, and naming conventions. Access a ranked list of models lacking coverage to prioritize your efforts.

### Data Profiling  
Perform efficient single-pass column statistics—including NULL rates, cardinality, min/max values, and uniqueness—using one batched SQL query per table to avoid slow row-by-row scanning.

### Schema Drift Detection
Compare the current state of your data warehouse against the definitions in your `manifest.json`. Instantly identify added, removed, or type-changed columns.

### Intelligent Test Suggestions
Translate profiling statistics into actionable dbt test recommendations. For example, a uniquely populated column without nulls will prompt suggestions for `not_null` and `unique` tests, while low cardinality will suggest `accepted_values` with predefined options.

### Non-Destructive YAML Writing
Update `schema.yml` files using `ruamel.yaml` to retain hand-written comments, existing tests, and formatting. The tool only appends missing information and preserves your manual configurations.

### Description Quality Review
Ask your assistant to review the column descriptions of one model, for example: *"Use dbt-doctor to evaluate the descriptions of `fct_orders`."*

dbt-doctor does not call any LLM itself. It hands your assistant the columns, the compiled SQL, a few deterministic hints (empty or repeated descriptions, missing units, complex `CASE`/`COALESCE` logic) and a rubric. The assistant scores every column on every question (0 to 10) and sends the scores back; it does not write descriptions. The result lists every column with its total in points and a status: `ok`, `rewrite` (total below the pass threshold, which is `pass_ratio` of the maximum, 70% by default, so 35 of 50 with the default rubric) or `missing` (empty description, recorded without scoring). When the database is connected, the tool also describes the real table, so columns that exist there but are not documented in `schema.yml` are included too (as `missing`, with `declared: false`); `database_check` in the result says whether that check ran. A future documentation-update step can use that list to change only the `missing` and `rewrite` columns. Wide tables are processed in batches automatically, and nothing is written to your `schema.yml`.

When the model has a parent, each column also carries its **upstream context**: the parent model and column, how it is built (passthrough, renamed, cast or transformed), the parent's description, the expression that builds it (for example a `CASE WHEN` in the staging model) and the chain up to the root. The assistant judges a copied column against its parent's description and a transformed one against the transformation it can see, with no extra prompting.

The default rubric has five questions (not just repeating the column name, explicit unit/format, clear grain, SQL logic reflected, understandable by business users). To change them, copy [description_rubric.json](src/dbt_doctor/description_rubric.json), edit it, and start the server with `--rubric-file /path/to/rubric.json`. The pass threshold follows the number of questions. Run `dbt compile` first so the compiled SQL is available.

### Description Consistency Across Layers
The same column often gets a different description in each layer even when nothing changed. dbt-doctor reads the dependency graph from `manifest.json` and the raw SQL of each model (no `dbt compile` needed), traces each column to its parent column and applies these rules when you write with `update_model_yaml`:

| Column | What the description must be |
|---|---|
| Plain passthrough, model keeps the rows | **Identical** to the parent's, and only once the parent's description has passed the quality threshold. Otherwise the write is blocked until the parent is fixed or scored |
| Transformed, or the model filters, regroups, dedupes, unions or limits rows | Its **own** text. Sending the parent's text unchanged is rejected as a stale copy |
| Parent is a source, not a dbt model | Not enforced: described on its own, like the root layer |
| Pure type cast | Not enforced by default (`--treat-casts-as-passthrough` copies it too) |
| Lineage not readable (joins, CTEs, unknown Jinja) | Not enforced |

Work from the root layer (bronze) downwards: evaluate a model with `evaluate_model_descriptions`/`submit_description_scores`, run `plan_description_updates` on its children, write, then run `dbt parse` so the manifest reflects the new text. A description's score stops applying as soon as its text changes. `update_model_yaml` always updates the file the manifest says documents the model (for example `<model>_schema.yaml`) and never creates a second definition; a model with no YAML gets `<model>_schema.yaml` (`--schema-file-pattern` changes the name). Use `dry_run=True` to preview a write. The layer order (`--layer-order`, default `landing,bronze,stg,silver,intermediary,gold|view,az`) only labels and groups models; the dependency graph decides how descriptions propagate.

### Documentation Campaign (analysis phase)
Give `start_campaign` a list of tables and drive it with `get_next_work`. Nothing is modified: the campaign only finds out which tables have a problem and builds the work list.

1. Each listed table is **tested**: your assistant scores its descriptions with `evaluate_model_descriptions` and `submit_description_scores`. A column that plainly inherits from a parent whose description passed, with the same text, counts as ok without a score of its own.
2. A table **passes** when at least `pass_ratio` (70% by default) of its columns are ok. One that fails, by itself or because of its parents, pulls its **parents** into the list; they are tested the same way, one level at a time, until a parent passes or the bronze layer is reached. Sources and landing tables stay in the list as `deferred` and are never processed.
3. The **direct children** of the listed tables and of the failing parents are added to the list but **not tested** (`children_scope` changes whose children are added). A child is tested only once one of its direct parents has been **updated**, which is detected with a checksum of the parent's descriptions kept in the run; the list then grows one level at a time.
4. **Nothing is tested twice.** Scores are kept by description text, so a table already scored is never asked for again, and editing a description sends it back to testing.

Statuses: `needs_test`, `passed`, `failed_self`, `blocked_by_parent`, `untested_child`, `deferred`, `skipped`, `error`. They are derived from the evidence on every call; only the membership of the list is persisted (in the run), so it survives restarts (`start_run` with `resume=True`) and only grows. A table asked for repeatedly with no progress becomes `error` after a few requests. When `get_next_work` reports `A_complete`, review the whole list with `get_campaign_status`. Generating the documentation for the list is a later phase and is not available yet.

`get_table_lineage` gives the plain lineage of one table: its parents one level up and every descendant, with the layer, the depth and how its SQL reads.

### Run History
Every tool of the documentation flow records what it does into a **run**, in a DuckDB file (`<project>/.dbt-doctor/dbt_doctor.duckdb`, with its own `.gitignore` so nothing reaches your repository). A run opens by itself on first use; `start_run` gives it a goal and a target model, and `start_run(resume=True)` continues the latest open one after a restart or after `dbt parse` changed the manifest.

| Tool | What is recorded in the run |
|---|---|
| `evaluate_model_descriptions` | the batch served, the descriptions of the model at first contact, and the lineage edges it used |
| `submit_description_scores` | the score of every question, the total and the status |
| `plan_description_updates` | the decision per column (rule, action, verdict, reason, parent status) |
| `update_model_yaml` | each change with the old and the new text, and what was copied, skipped or rejected (dry runs included, flagged) |

Texts are stored once by hash, and scores are global: a score stays valid for any run as long as the description text and the rubric are the same. Use `get_run_report` to review a campaign, or query the file with DuckDB (`v_latest_scores` is the current score of each description).

Recording never breaks a tool: if the file is locked by another process or unavailable, the tool still works, the result says `recording: unavailable (reason)`, and recording resumes a few seconds later. Options: `--run-store PATH` to move the file, `--no-run-store` to turn it off (scores then fall back to a JSON file under `target/dbt_doctor/`). An existing `description_scores.json` is imported once.

### End-to-End Documentation Generation
Execute a complete workflow in a single conversational turn: profile a model, suggest tests, preview changes, and write to schema.yml.

---

## Included MCP Tools

| Category | Tool | Description |
|---|---|---|
| Context | `list_models` | Overview of all models and their coverage status |
| Context | `get_model_details` | Detailed model information including SQL, columns, lineage, and tests |
| Audit | `audit_project` | Project health score and naming convention violations |
| Audit | `check_test_coverage` | Models ranked by their test coverage percentage |
| Audit | `analyze_dag` | Detection of orphan models and high fan-out nodes |
| Audit | `get_project_health` | Single-call dashboard summarizing project status |
| Quality | `evaluate_model_descriptions` | Prepares one model's columns, SQL and a scoring rubric for your AI assistant to judge description quality |
| Quality | `submit_description_scores` | Records the scores, applies the pass threshold and returns which columns need a rewrite |
| Lineage | `plan_description_updates` | For one model, which column descriptions must be copied from the parent, written anew, or are blocked |
| Lineage | `trace_column_lineage` | Follows a column upstream through the layers |
| Lineage | `check_description_consistency` | Paged overview of description consistency, parents before children |
| Campaign | `get_table_lineage` | One table's parents (one level up) and every table below it, as a plain list read from the manifest |
| Campaign | `start_campaign` | Starts a documentation campaign from a list of tables (phase A: analysis, read-only) |
| Campaign | `get_next_work` | The next tables the campaign needs tested; recomputes the list from the evidence on every call |
| Campaign | `get_campaign_status` | The whole campaign list, parents before children, paged |
| Campaign | `skip_work_item` | Takes a table out of the campaign on purpose |
| History | `start_run` | Opens a run with a goal and a target model, or resumes the latest open one |
| History | `get_run_report` | What a run did: events per tool, and per model the columns seen, scores, plan actions and writes |
| History | `list_runs` | The recorded runs, newest first |
| Profiling | `profile_model` | Batched column statistics |
| Profiling | `execute_query` | Read-only SQL execution against your warehouse |
| Profiling | `detect_schema_drift` | Comparison of database columns against manifest definitions |
| Generation | `suggest_tests` | Translation of profile data into dbt test recommendations |
| Generation | `update_model_yaml` | Writes new descriptions (replacing existing ones, following the lineage rules) and merges tests into the model's schema file |
| Generation | `generate_model_docs` | Complete end-to-end documentation workflow |

---

## Quick Start

### Installation

```bash
pip install dbt-doctor

# Optional: Trino support
pip install "dbt-doctor[trino]"
```

### Supported adapters

dbt-doctor reads the connection from the `type` of your dbt `profiles.yml` target.

| Adapter | `type` | Install |
|---|---|---|
| PostgreSQL | `postgres` | included |
| DuckDB | `duckdb` | included |
| Trino | `trino` | `pip install "dbt-doctor[trino]"` |

For Trino, the same keys as a [dbt-trino](https://github.com/starburstdata/dbt-trino) profile are used:

```yaml
my_project:
  target: dev
  outputs:
    dev:
      type: trino
      method: ldap            # none (default), ldap/password, or jwt
      host: trino.example.com
      port: 443               # defaults to 8080 (http) or 443 (https)
      http_scheme: https      # defaults to http for method none, https for ldap/jwt
      user: my_user
      password: "{{ env_var('TRINO_PASSWORD') }}"
      database: hive          # the Trino catalog (required)
      schema: analytics
```

For `method: jwt`, provide `jwt_token`. Other methods (Kerberos, OAuth, certificate) are not supported yet.

### Configuration (Claude Desktop)

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dbt-doctor": {
      "command": "dbt-doctor",
      "args": ["--project-dir", "/absolute/path/to/your/dbt/project"]
    }
  }
}
```

### Configuration (Cursor)

Add the following to your `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "dbt-doctor": {
      "command": "dbt-doctor",
      "args": ["--project-dir", "/absolute/path/to/your/dbt/project"]
    }
  }
}
```

*Prerequisite: Run `dbt compile` prior to usage to ensure `target/manifest.json` is available for dbt-doctor to parse.*

---

## Architecture

![dbt-doctor architecture](docs/architecture.png)

The application connects the AI Assistant with your dbt project and database through the MCP protocol. It features a read-only analysis layer combined with a secure generation toolkit that merges changes seamlessly into your existing YAML schemas.

---

## Security Design

- **Read-only execution:** All `execute_query` operations operate within a read-only transaction. Write processes are restricted at the database connector level.
- **SQL validation:** Table and column identifiers are strictly validated against a whitelist to prevent injection.
- **Stateless connections:** Data warehouse credentials are instantiated per connection and are never cached in memory.
- **Preview before commit:** The document generation process provides a difference preview prior to rewriting `schema.yml`, ensuring you retain control over modifications.

---

## Related Projects

| Project | Description |
|---|---|
| dbt-labs/dbt-mcp | Official MCP focused on dbt command execution |
| dbt-coverage | CLI tool for coverage reporting without AI integration |
| dbt-project-evaluator | dbt package for project evaluation, requiring installation per project |

dbt-doctor uniquely consolidates auditing, profiling, drift detection, and AI-driven YAML updates into a single server interface.

---

## License

MIT — see the [LICENSE](LICENSE) file.
