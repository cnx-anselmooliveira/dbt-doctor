# Known Limitations

**Snapshot date:** 2025-12-25

- SQL safety is intentionally conservative but still lightweight. The project should move from prefix-based checks toward AST-based validation before being used against sensitive warehouses.
- Adapter support is limited to Postgres, DuckDB and Trino. Trino supports `none`, `ldap`/`password` and `jwt` authentication only, and requires the optional `trino` extra. Trino `impersonation_user` and custom TLS certificates (`verify`/`cert`) are not supported yet.
- The package is alpha quality; public APIs and tool names may change.
- Profiling large tables can be expensive because it runs aggregate queries across columns.
- The description-quality review only sees columns declared in `schema.yml` unless the database is connected (then undocumented columns of the table are included; `database_check` reports when it could not run), and detects complex SQL logic with a textual heuristic that needs an explicit `AS <column>` alias. The scores come from your AI assistant, so they can vary between runs. The review only reports which columns are `ok`, `rewrite` or `missing`; rewriting descriptions in `schema.yml` is not implemented yet.
- Column lineage is read from the raw SQL with textual heuristics, not a SQL parser. Models with joins, CTEs, subqueries or unknown Jinja (about half of a typical project) are not enforced yet. A parser-based level is planned. `update_model_yaml` writes descriptions as plain scalars: the folded (`>`) style is not kept for new text.
- The score of a description is stored only for models evaluated with `evaluate_model_descriptions`; descriptions of sources are never scored, so a column whose parent is a source (typically a staging model over a bronze source) is described on its own and no copy rule applies.
- The documentation campaign only has its analysis phase: it lists what has a problem but does not generate documentation yet. A table passes at the share of ok columns (70% by default), so a table that passed can still have failing columns that block its children: the generation phase will have to fix those too. Inheritance is only recognised when the SQL is readable and the table has a single parent; joins and CTEs are scored on their own. Only the direct children are listed; the update of a table is detected from the manifest, so after writing descriptions the manifest has to reflect them (`dbt parse`) for the children to be promoted.
- The run history is a DuckDB file, which allows a single writing process: two sessions recording into the same project at once may see `recording: unavailable` for a few seconds (the tools keep working). Descriptions are stored in clear text, and the file is not cleaned up automatically.
- The documentation generator prepares structured suggestions, but human review is still expected before writing docs/tests into a production dbt project.
- The repository overlaps with the broader dbt MCP ecosystem. The intended wedge is practical quality and documentation workflows, not full dbt platform control.

