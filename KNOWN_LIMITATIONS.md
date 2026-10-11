# Known Limitations

**Snapshot date:** 2025-12-25

- SQL safety is intentionally conservative but still lightweight. The project should move from prefix-based checks toward AST-based validation before being used against sensitive warehouses.
- Adapter support is limited to Postgres and DuckDB.
- The package is alpha quality; public APIs and tool names may change.
- Profiling large tables can be expensive because it runs aggregate queries across columns.
- The description-quality review only sees columns declared in `schema.yml` (that is what the manifest contains), and detects complex SQL logic with a textual heuristic that needs an explicit `AS <column>` alias. The scores come from your AI assistant, so they can vary between runs. Applying the suggested rewrites is not implemented yet.
- The documentation generator prepares structured suggestions, but human review is still expected before writing docs/tests into a production dbt project.
- The repository overlaps with the broader dbt MCP ecosystem. The intended wedge is practical quality and documentation workflows, not full dbt platform control.

