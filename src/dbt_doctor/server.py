"""
dbt-doctor MCP Server — AI-driven quality & governance for dbt projects.

All 20 MCP Tools:
  Audit:      audit_project, check_test_coverage, analyze_dag, get_project_health
  Quality:    evaluate_model_descriptions, submit_description_scores
  Lineage:    plan_description_updates, trace_column_lineage, check_description_consistency
  History:    start_run, get_run_report, list_runs
  Profiling:  profile_model, execute_query, detect_schema_drift
  Generation: suggest_tests, update_model_yaml, generate_model_docs
  Context:    list_models, get_model_details
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .analyzers.auditor import ProjectAuditor
from .analyzers.dag_analyzer import DagAnalyzer
from .analyzers.description_quality import (
    DescriptionQualityEvaluator,
    RubricError,
    load_rubric,
)
from .analyzers.drift_detector import SchemaDriftDetector
from .analyzers.lineage import DEFAULT_LAYER_ORDER, LayerMap, LineageGraph, LineagePlanner
from .analyzers.profiler import DataProfiler
from .analyzers.score_store import STORE_FILENAME, ScoreStore
from .connectors.base import BaseConnector, ColumnInfo
from .core.manifest import DbtManifestParser
from .core.profiles import DbtProfileParser
from .core.project import DbtProjectParser
from .core.run_recorder import RunRecorder
from .core.run_store import DEFAULT_RELATIVE_PATH, RunStore
from .generators.doc_generator import DocGenerator
from .generators.test_suggester import TestSuggester
from .generators.yaml_writer import YamlWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("dbt_doctor")

# ── Singleton server instance ──────────────────────────────────────────────
mcp = FastMCP("dbt-doctor")

# ── Module-level singletons (set by main() before mcp.run()) ──────────────
_manifest: DbtManifestParser | None = None
_connector: BaseConnector | None = None
_rubric_path: Path | None = None
_layer_map: LayerMap = LayerMap()
_casts_as_passthrough: bool = False
_schema_file_pattern: str = "{model}_schema.yaml"
_run_store_enabled: bool = True
_run_store_path: Path | None = None
_recorder: RunRecorder | None = None
_auditor = ProjectAuditor()
_dag_analyzer = DagAnalyzer()


def _get_manifest() -> DbtManifestParser:
    if _manifest is None:
        raise RuntimeError("dbt-doctor not initialized. Pass --project-dir to the server.")
    _manifest.load()
    return _manifest


def _run_context() -> dict[str, Any]:
    """What a new run records about its environment."""
    manifest_info: dict[str, Any] = {}
    if _manifest is not None:
        meta = (_manifest.get_raw().get("metadata") or {}) if _manifest else {}
        try:
            mtime = _manifest.manifest_path.stat().st_mtime
        except OSError:
            mtime = None
        manifest_info = {
            "path": str(_manifest.manifest_path),
            "mtime": mtime,
            "generated_at": meta.get("generated_at"),
            "invocation_id": meta.get("invocation_id"),
            "project_name": meta.get("project_name"),
        }
    try:
        fingerprint = load_rubric(_rubric_path).fingerprint()
    except Exception:
        fingerprint = ""
    params = {"layer_order": _layer_map.order, "casts_as_passthrough": _casts_as_passthrough}
    return {"manifest": manifest_info, "rubric_fp": fingerprint, "params": params}


def _get_recorder() -> RunRecorder | None:
    """The run recorder of the current project, or None when the run store is disabled."""
    global _recorder
    if not _run_store_enabled:
        return None
    project_dir = _get_manifest().project_dir
    path = _run_store_path or project_dir / DEFAULT_RELATIVE_PATH
    if _recorder is None or _recorder.store.path != path:
        _recorder = RunRecorder(RunStore(path), _run_context)
        _recorder.import_legacy_scores(project_dir / "target" / "dbt_doctor" / STORE_FILENAME)
    return _recorder


def _score_store() -> Any:
    """Where scores live: the run store, or the legacy JSON file when it is disabled."""
    recorder = _get_recorder()
    if recorder is not None:
        return recorder
    return ScoreStore(_get_manifest().project_dir / "target" / "dbt_doctor" / STORE_FILENAME)


class _NoEvent:
    ok = True

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}


@contextmanager
def _flow_event(tool: str, model: str | None) -> Iterator[tuple[RunRecorder | None, Any]]:
    """Wrap a tool call so what it records is linked to one event of the current run."""
    recorder = _get_recorder()
    if recorder is None:
        yield None, _NoEvent()
    else:
        with recorder.event(tool, model) as event:
            yield recorder, event


def _descriptions(details: dict[str, Any]) -> dict[str, str]:
    return {c: (d.get("description") or "") for c, d in details.get("columns", {}).items()}


def _edges_from_upstream(upstream: dict[str, dict] | None) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "parent": info.get("parent"),
            "parent_column": info.get("parent_column"),
            "kind": info.get("kind"),
            "effects": info.get("model_effects"),
            "confidence": "low" if info.get("kind") == "unknown" else "high",
            "evidence": (info.get("built_by") or {}).get("expression") or info.get("why"),
        }
        for column, info in (upstream or {}).items()
    ]


def _edges_from_plan(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for e in entries:
        lineage = e.get("lineage")
        if not lineage:
            continue
        edges.append(
            {
                "column": e["column"],
                "parent": lineage.get("parent"),
                "parent_column": lineage.get("parent_column"),
                "kind": lineage.get("kind"),
                "effects": lineage.get("effects"),
                "confidence": "low" if lineage.get("kind") == "unknown" else "high",
                "evidence": lineage.get("evidence"),
            }
        )
    return edges


def _write_rows(
    model_name: str, result: dict[str, Any], report: dict[str, list], dry_run: bool
) -> list[dict[str, Any]]:
    copied = {c["column"].lower() for c in report["copied"]}
    file_path = result["file_path"]
    rows: list[dict[str, Any]] = []
    for change in result["changes"]:
        column = change.get("column")
        rows.append(
            {
                "model": model_name,
                "scope": change["scope"],
                "column": column,
                "file_path": file_path,
                "action": "changed",
                "via": "copy" if column and column.lower() in copied else "supplied",
                "old": change.get("old"),
                "new": change.get("new"),
                "dry_run": dry_run,
            }
        )
    for action, key in (("skipped", "skipped"), ("rejected", "rejected")):
        for item in report[key]:
            rows.append(
                {
                    "model": model_name,
                    "scope": "column",
                    "column": item["column"],
                    "file_path": file_path,
                    "action": action,
                    "reason": item["reason"],
                    "dry_run": dry_run,
                }
            )
    return rows


def _get_evaluator() -> DescriptionQualityEvaluator:
    manifest = _get_manifest()
    # Reloaded on every call so edits to the rubric file apply without restarting the server.
    state_dir = manifest.project_dir / "target" / "dbt_doctor"
    return DescriptionQualityEvaluator(load_rubric(_rubric_path), state_dir, _score_store())


def _upstream(model_name: str) -> dict[str, dict] | None:
    """Parent context for each column; the review still works if it cannot be built."""
    try:
        return _get_planner().upstream_context(model_name) or None
    except Exception as exc:
        logger.warning("Could not build the upstream context for %s: %s", model_name, exc)
        return None


def _get_planner() -> LineagePlanner:
    manifest = _get_manifest()
    graph = LineageGraph(manifest.get_raw(), _layer_map)
    fingerprint = load_rubric(_rubric_path).fingerprint()
    return LineagePlanner(graph, _score_store(), fingerprint, _casts_as_passthrough)


def _database_columns(details: dict[str, Any]) -> tuple[list[ColumnInfo] | None, str | None]:
    """Describe the model's table in the database. Never raises.

    Returns ``(columns, None)`` on success, or ``(None, reason)`` when the check could not
    be done, so the review still works without a database connection.
    """
    if _connector is None:
        return None, "database not connected (check profiles.yml); columns were not verified"
    schema = details.get("schema")
    table = details.get("alias") or details["name"]
    if not schema:
        return None, "the model has no schema in the manifest; columns were not verified"
    try:
        columns = _connector.get_table_columns(schema, table)
    except Exception as exc:
        return None, f"could not describe {schema}.{table}: {exc}"
    if not columns:
        return None, f"table {schema}.{table} was not found in the database"
    return columns, None


def _get_connector() -> BaseConnector:
    if _connector is None:
        raise RuntimeError(
            "Database connector not available. "
            "Check that profiles.yml is reachable and credentials are correct."
        )
    return _connector


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def list_models() -> str:
    """
    List all dbt models in the project.

    Returns a summary of every model including name, materialization, description status,
    and documentation coverage. Use this as your starting point before any other tool.
    """
    parser = _get_manifest()
    models = parser.get_models()
    if not models:
        return "No models found. Run `dbt compile` in your project first."

    lines = [f"📦 dbt Project — {len(models)} models found:\n"]
    for m in sorted(models, key=lambda x: x["name"]):
        doc_status = "✅" if m["has_description"] else "❌"
        col_coverage = (
            f"{m['documented_column_count']}/{m['column_count']}"
            if m["column_count"] > 0
            else "no cols"
        )
        lines.append(
            f"  {doc_status} {m['name']:<40} [{m['materialized']:<8}]  "
            f"cols: {col_coverage}  schema: {m['schema']}"
        )
    return "\n".join(lines)


@mcp.tool()
def get_model_details(model_name: str) -> str:
    """
    Get full details for a specific dbt model: SQL code, columns, lineage, and config.

    Use this before writing documentation or suggesting tests for a specific model.
    Also use this to understand what a model does before profiling it.

    Args:
        model_name: Exact name of the dbt model (e.g. 'fct_orders' or 'stg_users').
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found. Run `list_models` to see available models."

    lines = [
        f"Model: {details['name']}",
        f"  Description: {details['description'] or '(not documented)'}",
        f"  Path:        {details['path']}",
        f"  Schema:      {details['schema']}  |  Alias: {details['alias']}",
        f"  Materialized: {details['materialized']}",
        f"  Tags: {', '.join(details['tags']) if details['tags'] else 'none'}",
        "",
        "─── Columns ───────────────────────────────────────────",
    ]
    if details["columns"]:
        for col_name, col in details["columns"].items():
            desc = col.get("description", "")
            tests = col.get("tests", [])
            test_str = ", ".join(t if isinstance(t, str) else next(iter(t)) for t in tests)
            lines.append(
                f"  {col_name:<35}  desc={'✅' if desc else '❌'}  tests=[{test_str or 'none'}]"
            )
    else:
        lines.append("  (no columns in manifest — run `dbt parse` to populate)")

    lines.append("\n─── Upstream Dependencies ──────────────────────────────")
    if details["depends_on"]:
        for dep in details["depends_on"]:
            lines.append(f"  ← {dep}")
    else:
        lines.append("  (no upstream dependencies — this is a root model)")

    lines.append("\n─── Raw SQL ────────────────────────────────────────────")
    sql = details.get("raw_sql", "")
    lines.append(sql[:2000] + ("\n... [truncated]" if len(sql) > 2000 else ""))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# DESCRIPTION QUALITY TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def evaluate_model_descriptions(model_name: str, offset: int = 0) -> str:
    """
    Prepare the next batch of columns of ONE model for a description-quality review.

    Read-only. Returns the model context, compiled SQL, deterministic signals per column
    and a scoring rubric. You (the AI assistant) judge each column on each rubric question,
    then send the scores with `submit_description_scores`.

    Each column also carries its `upstream` context when it can be read: the parent model
    and column, how it is built (passthrough, renamed, cast, transformed), the parent's
    description and the expression that builds it. When the database is connected the
    table is described too, so columns that exist there but are not documented in
    schema.yml are included (reported as 'missing', `declared: false`), and
    `database_check` says whether the check ran.

    Wide tables are split into batches: keep calling this with `offset=next_offset` until
    `next_offset` is null. Start a new evaluation with offset=0 (this discards any
    unfinished evaluation of the same model). Requires `dbt compile` for the compiled SQL.
    What it serves is recorded in the current run (`run` in the result).

    Args:
        model_name: Exact dbt model name.
        offset: Index of the first column of the batch. 0 starts a new evaluation.
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found. Run `list_models` to see available models."
    with _flow_event("evaluate_model_descriptions", model_name) as (recorder, event):
        try:
            db_columns, db_note = _database_columns(details)
            upstream = _upstream(model_name)
            batch = _get_evaluator().next_batch(
                details, offset, db_columns=db_columns, db_note=db_note, upstream=upstream
            )
        except (RubricError, ValueError) as exc:
            event.ok = False
            event.summary["error"] = str(exc)
            return f"❌ {exc}"
        if recorder is not None:
            event.summary.update(
                offset=offset,
                total_columns=batch["total_columns"],
                returned=len(batch["columns"]),
                next_offset=batch["next_offset"],
                database_check=batch["database_check"]["status"],
                upstream_columns=len(upstream or {}),
            )
            recorder.record_snapshot(model_name, _descriptions(details))
            recorder.record_lineage(model_name, _edges_from_upstream(upstream))
            batch["run"] = recorder.info()
    return json.dumps(batch, ensure_ascii=False)


@mcp.tool()
def submit_description_scores(model_name: str, scores: list[dict]) -> str:
    """
    Record your scores for a batch from `evaluate_model_descriptions`.

    Each item: {'column': str, 'scores': {<question_id>: <number>}}. Every rubric question id
    must be present. Columns with an empty description are recorded as 'missing' automatically
    and need no scores (send an empty list if a batch has only those). Invalid input is
    rejected as a whole. This tool only scores; it never writes or suggests descriptions.

    While columns remain it returns progress and the next offset. When every column of the
    model has been covered it returns the final report: each column with its total (in points,
    not a percentage) and status: 'ok', 'rewrite' (total below the pass threshold) or
    'missing' (empty description). `to_update` lists the 'rewrite' and 'missing' columns.
    The scores are recorded in the current run. This tool does not modify any dbt file.

    Args:
        model_name: Exact dbt model name.
        scores: List of per-column scores for the columns of the current batch.
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found. Run `list_models` to see available models."
    with _flow_event("submit_description_scores", model_name) as (recorder, event):
        try:
            db_columns, db_note = _database_columns(details)
            result = _get_evaluator().submit(
                details, scores, db_columns=db_columns, db_note=db_note
            )
        except (RubricError, ValueError) as exc:
            event.ok = False
            event.summary["error"] = str(exc)
            return f"❌ {exc}"
        if recorder is not None:
            keys = ("status", "evaluated", "total_columns", "summary")
            event.summary.update({k: result[k] for k in keys if k in result})
            if "to_update" in result:
                event.summary["to_update"] = len(result["to_update"])
            result["run"] = recorder.info()
    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# LINEAGE TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def plan_description_updates(model_name: str) -> str:
    """
    Read-only plan for the column descriptions of ONE model, based on its lineage.

    For each documented column it says what the description must be:
    - `copy`: a plain passthrough that keeps the rows. It must repeat the parent column's
      description identically, and the parent's description passed the quality threshold.
    - `write_new`: the column is empty, transformed, or goes through a model that filters
      or regroups rows, so it needs a description of its own (a copy of the parent's text
      would be stale).
    - `blocked`: the parent column has no description, has not been scored, or failed the
      threshold. Fix or score the parent first (`evaluate_model_descriptions`).
    - `keep`: nothing to do.

    Work models from the root layer (bronze) downwards. `update_model_yaml` enforces the
    same rules when writing. Run `dbt parse` after writing so the manifest reflects it.
    The plan is recorded in the current run.

    Args:
        model_name: Exact dbt model name.
    """
    parser = _get_manifest()
    with _flow_event("plan_description_updates", model_name) as (recorder, event):
        try:
            plan = _get_planner().plan_model(model_name)
        except (RubricError, ValueError) as exc:
            event.ok = False
            event.summary["error"] = str(exc)
            return f"❌ {exc}"
        if recorder is not None:
            event.summary.update(
                layer=plan["layer"], summary=plan["summary"], blocked_by=plan["blocked_by"]
            )
            details = parser.get_model_details(model_name)
            if details:
                recorder.record_snapshot(model_name, _descriptions(details))
            recorder.record_plan(model_name, plan["columns"])
            recorder.record_lineage(model_name, _edges_from_plan(plan["columns"]))
            plan["run"] = recorder.info()
    return json.dumps(plan, ensure_ascii=False)


@mcp.tool()
def trace_column_lineage(model_name: str, column: str) -> str:
    """
    Follow a column upstream through the dbt layers, up to the root layer or the first
    model where its lineage cannot be read with confidence.

    Each step shows the model, its layer, whether the column is described there, how it is
    built (passthrough, renamed, transformed) and the row-altering effects of the model.

    Args:
        model_name: Exact dbt model name.
        column: Column name.
    """
    try:
        chain = _get_planner().trace(model_name, column)
    except (RubricError, ValueError) as exc:
        return f"❌ {exc}"
    return json.dumps({"model": model_name, "column": column, "chain": chain}, ensure_ascii=False)


@mcp.tool()
def check_description_consistency(layer: str = "", offset: int = 0, limit: int = 50) -> str:
    """
    Paged overview of description consistency across the project, parents before children.

    For every model it counts the columns that need `copy`, `write_new`, are `blocked` or
    fine (`keep`). Use `layer` (e.g. 'bronze', 'silver', 'gold') to focus on one layer and
    `offset`/`limit` to page. Follow up with `plan_description_updates` on a model.

    Args:
        layer: Optional layer label to filter by (empty = all models).
        offset: Index of the first model of the page.
        limit: Models per page.
    """
    with _flow_event("check_description_consistency", None) as (recorder, event):
        try:
            report = _get_planner().consistency(layer, max(offset, 0), max(1, min(limit, 200)))
        except (RubricError, ValueError) as exc:
            event.ok = False
            event.summary["error"] = str(exc)
            return f"❌ {exc}"
        if recorder is not None:
            event.summary.update(
                layer=report["layer"],
                total_models=report["total_models"],
                totals=report["totals"],
                offset=report["offset"],
            )
            report["run"] = recorder.info()
    return json.dumps(report, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# RUN HISTORY TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def start_run(goal: str = "", target_model: str = "", resume: bool = False) -> str:
    """
    Open a run with an explicit context, or resume the latest open one.

    Every tool of the documentation flow records what it does (batches served, scores,
    plans, writes) into the current run, so the whole campaign can be reviewed later with
    `get_run_report`. A run opens by itself on first use; call this to give it a goal and a
    target model, or with `resume=True` to continue the latest open run (for the same
    target model, if given) after a restart or after `dbt parse` changed the manifest.
    Starting a new run closes the previous open ones.

    Args:
        goal: What this campaign is for, in your own words.
        target_model: The model the campaign is about (optional).
        resume: Continue the latest open run instead of starting a new one.
    """
    _get_manifest()
    recorder = _get_recorder()
    if recorder is None:
        return "The run store is disabled (--no-run-store); nothing is being recorded."
    return json.dumps(recorder.start_run(goal, target_model, resume), ensure_ascii=False)


@mcp.tool()
def get_run_report(run_id: str = "") -> str:
    """
    Summary of one run: what each tool did, per model (columns seen, scores, plan actions,
    writes), the most common blocking reasons and the latest events.

    Args:
        run_id: The run to report (empty = the current run).
    """
    _get_manifest()
    recorder = _get_recorder()
    if recorder is None:
        return "The run store is disabled (--no-run-store); nothing is being recorded."
    target = run_id or recorder.run_id
    if not target:
        return "No run yet. Use a flow tool or `start_run` first."
    try:
        report = recorder.store.run_report(target)
    except Exception as exc:
        return f"❌ Could not read the run store: {exc}"
    if report is None:
        return f"Run '{target}' not found. Use `list_runs` to see the available runs."
    return json.dumps(report, ensure_ascii=False, default=str)


@mcp.tool()
def list_runs(limit: int = 20) -> str:
    """
    List the recorded runs, newest first: status, goal, target model, number of events and
    of models touched.

    Args:
        limit: Maximum number of runs to list.
    """
    _get_manifest()
    recorder = _get_recorder()
    if recorder is None:
        return "The run store is disabled (--no-run-store); nothing is being recorded."
    try:
        runs = recorder.store.list_runs(limit)
    except Exception as exc:
        return f"❌ Could not read the run store: {exc}"
    return json.dumps(
        {"current_run": recorder.run_id, "runs": runs}, ensure_ascii=False, default=str
    )


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def audit_project() -> str:
    """
    Run a comprehensive quality audit of the entire dbt project.

    Checks:
    - Model-level documentation coverage (% of models with descriptions)
    - Column-level documentation coverage (% of columns with descriptions)
    - Test coverage (% of models with at least one dbt test)
    - Naming convention adherence (snake_case, layer prefixes)

    Returns an overall health score (0-100%) and a list of specific issues.
    Use this as the FIRST step before any documentation or test generation work.
    """
    parser = _get_manifest()
    models = parser.get_models()
    if not models:
        return "No models found. Run `dbt compile` first."

    # Enrich models with full column info from manifest
    enriched = []
    for m in models:
        details = parser.get_model_details(m["name"])
        if details:
            enriched.append(
                {
                    **m,
                    "columns": details.get("columns", {}),
                    "description": details.get("description", ""),
                }
            )
        else:
            enriched.append(m)

    report = _auditor.audit(enriched)
    return _auditor.format_report(report)


@mcp.tool()
def check_test_coverage() -> str:
    """
    Show detailed test coverage statistics for all models.

    Returns a ranked list of models sorted by test coverage, from worst to best.
    Use this to identify which models need tests the most, then call `suggest_tests`
    on the worst offenders.
    """
    parser = _get_manifest()
    models = parser.get_models()
    if not models:
        return "No models found."

    enriched = []
    for m in models:
        details = parser.get_model_details(m["name"]) or {}
        enriched.append(
            {
                **m,
                "columns": details.get("columns", {}),
                "description": details.get("description", ""),
            }
        )

    report = _auditor.audit(enriched)

    lines = [
        f"  🧪 Test Coverage Report ({report.test_score:.1f}% of models have ≥1 test)\n",
        f"  {'Model':<40} {'Test Cov':>8}  {'Doc Cov':>7}  Columns",
        "  " + "─" * 65,
    ]
    for mr in report.model_results:
        test_pct = round(mr.test_coverage * 100)
        doc_pct = round(mr.doc_coverage * 100)
        status = "🔴" if test_pct < 20 else ("🟡" if test_pct < 60 else "🟢")
        lines.append(
            f"  {status} {mr.name:<38} {test_pct:>7}%  {doc_pct:>6}%  "
            f"{mr.tested_columns}/{mr.total_columns}"
        )

    return "\n".join(lines)


@mcp.tool()
def analyze_dag() -> str:
    """
    Analyze the dbt project DAG (Directed Acyclic Graph) structure.

    Detects:
    - Orphan models (no downstream consumers or exposures)
    - Root models (no model dependencies, only sources)
    - High fan-out models (one model feeding many downstream consumers)
    - Maximum chain depth (long lineage chains)

    Use this to find structural issues and potential refactoring opportunities.
    """
    parser = _get_manifest()
    models = parser.get_models()
    exposures = parser.get_all_exposures()
    if not models:
        return "No models found."

    dag_report = _dag_analyzer.analyze(models, exposures)
    return _dag_analyzer.format_report(dag_report)


@mcp.tool()
def get_project_health() -> str:
    """
    Get a comprehensive health dashboard for the dbt project.

    This is the ENTRY POINT tool — call this first to get an overview of:
    - Overall quality score
    - Top coverage issues
    - DAG structural issues
    - Recommended next actions

    Think of it as a doctor's overall diagnosis before deciding on treatments.
    """
    parser = _get_manifest()
    models = parser.get_models()
    if not models:
        return "No models found. Run `dbt compile` in your dbt project directory first."

    enriched = []
    for m in models:
        details = parser.get_model_details(m["name"]) or {}
        enriched.append(
            {
                **m,
                "columns": details.get("columns", {}),
                "description": details.get("description", ""),
            }
        )

    report = _auditor.audit(enriched)
    exposures = parser.get_all_exposures()
    dag_report = _dag_analyzer.analyze(models, exposures)

    lines = [
        "╔" + "═" * 53 + "╗",
        "║  🩺 dbt-doctor — Project Health Dashboard         ║",
        "╠" + "═" * 53 + "╣",
        f"║  Overall Score:    {report.overall_score:>5.1f}%                          ║",
        "╟" + "─" * 53 + "╢",
        f"║  Documentation:    {report.model_doc_score:>5.1f}%  model descriptions      ║",
        f"║                    {report.column_doc_score:>5.1f}%  column descriptions     ║",
        f"║  Test Coverage:    {report.test_score:>5.1f}%  models with ≥1 test     ║",
        f"║  Naming Score:     {report.naming_score:>5.1f}%                          ║",
        "╟" + "─" * 53 + "╢",
        f"║  DAG: depth={dag_report.max_chain_depth:<2}  orphans={len(dag_report.orphan_models):<3}  "
        f"high-fanout={len(dag_report.high_fanout_models):<2}     ║",
        "╚" + "═" * 53 + "╝",
        "",
        "  📋 Recommended Actions:",
    ]

    priority = 1
    if report.overall_score < 50:
        lines.append(
            f"  {priority}. 🔴 Low overall quality. Start with: `audit_project` for details"
        )
        priority += 1
    if report.test_score < 60:
        worst = report.worst_models[:3]
        worst_names = ", ".join(m.name for m in worst)
        lines.append(f"  {priority}. Run `suggest_tests` on: {worst_names}")
        priority += 1
    if dag_report.orphan_models:
        lines.append(
            f"  {priority}. Review {len(dag_report.orphan_models)} orphan models "
            f"(first: {dag_report.orphan_models[0]})"
        )
        priority += 1
    if report.column_doc_score < 30:
        lines.append(f"  {priority}. Use `generate_model_docs` for E2E auto-documentation")
        priority += 1
    if _connector is not None:
        lines.append(
            f"  {priority}. ✅ DB connected — run `detect_schema_drift` to check for drift"
        )
    else:
        lines.append(
            f"  {priority}. ⚠️  No DB connection — profiling tools unavailable. "
            "Check profiles.yml setup."
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PROFILING TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def profile_model(model_name: str) -> str:
    """
    Profile a dbt model's output table in the data warehouse.

    Automatically locates the model's target schema and table from the manifest,
    then runs statistical analysis: null rates, unique counts, min/max values.

    Returns per-column statistics to help design accurate dbt tests.
    ALWAYS call this before `suggest_tests` or `generate_model_docs`.

    Args:
        model_name: Exact dbt model name (e.g. 'fct_orders').
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found."

    schema = details.get("schema") or ""
    table = details.get("alias") or model_name

    if not schema:
        return (
            f"Could not determine schema for model '{model_name}'. "
            "Make sure the manifest is up to date (`dbt compile`)."
        )

    connector = _get_connector()
    profiler = DataProfiler(connector)
    try:
        profile = profiler.profile_table(schema, table)
        return profiler.format_profile(profile)
    except Exception as exc:
        return f"Profiling failed for {schema}.{table}: {exc}"


@mcp.tool()
def execute_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT query against the connected data warehouse.

    Use this to manually inspect data, verify assumptions, or investigate anomalies
    before writing tests. Only SELECT statements are allowed.

    Args:
        query: A read-only (SELECT) SQL query to run.
    """
    connector = _get_connector()
    try:
        rows = connector.execute_query(query)
    except ValueError as exc:
        return f"❌ Rejected: {exc}"
    except Exception as exc:
        return f"❌ Query failed: {exc}"

    if not rows:
        return "Query succeeded — 0 rows returned."

    output_lines = [f"✅ {len(rows)} row(s) returned (max 200):"]
    for i, row in enumerate(rows[:20], 1):
        output_lines.append(f"  Row {i}: {json.dumps(row, default=str)}")
    if len(rows) > 20:
        output_lines.append(f"  ... and {len(rows) - 20} more rows (showing first 20)")
    return "\n".join(output_lines)


@mcp.tool()
def detect_schema_drift(model_name: str) -> str:
    """
    Detect schema drift between the actual database and the dbt manifest documentation.

    Compares columns in the DB vs columns documented in manifest.json and finds:
    - New columns added to DB but not yet documented in dbt
    - Documented columns that no longer exist in DB (stale docs)
    - Columns where the data type has changed

    This is a UNIQUE feature — run this regularly to keep your dbt docs in sync.

    Args:
        model_name: Exact dbt model name (e.g. 'stg_orders').
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found in manifest."

    schema = details.get("schema") or ""
    table = details.get("alias") or model_name
    manifest_columns: dict[str, Any] = details.get("columns") or {}

    if not schema:
        return f"Cannot determine schema for '{model_name}'."

    connector = _get_connector()
    detector = SchemaDriftDetector(connector)
    try:
        drift_report = detector.detect(schema, table, manifest_columns)
        return detector.format_report(drift_report)
    except Exception as exc:
        return f"Schema drift check failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def suggest_tests(model_name: str) -> str:
    """
    Suggest data quality tests for a dbt model based on actual data statistics.

    This tool profiles the model's data in the warehouse and intelligently recommends:
    - not_null: when null rate is 0%
    - unique: when all values are distinct
    - accepted_values: when a column has very few distinct values
    - Additional heuristics for ID columns, timestamp columns, etc.

    Each suggestion includes a confidence level (HIGH/MEDIUM) and reasoning.
    After reviewing, call `update_model_yaml` to apply the suggestions.

    Args:
        model_name: Exact dbt model name.
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found in manifest."

    schema = details.get("schema") or ""
    table = details.get("alias") or model_name

    if not schema:
        return f"Cannot determine schema for '{model_name}'."

    connector = _get_connector()
    profiler = DataProfiler(connector)
    suggester = TestSuggester()

    try:
        profile = profiler.profile_table(schema, table)
        suggestions = suggester.suggest(profile, model_name)
        return suggester.format_suggestions(suggestions)
    except Exception as exc:
        return f"Could not generate suggestions for '{model_name}': {exc}"


@mcp.tool()
def update_model_yaml(
    model_name: str,
    description: str,
    columns: list[dict],
    overwrite_descriptions: bool = True,
    dry_run: bool = False,
    apply_copies: bool = True,
) -> str:
    """
    Write descriptions and tests to the model's schema YAML, following the lineage rules.

    Uses ruamel.yaml to preserve existing human comments and other models in the same file.
    Descriptions you send REPLACE the existing ones (`overwrite_descriptions`); tests are
    only added, never removed. Before writing, each column is checked against its lineage:
    - a plain passthrough that keeps the rows gets the parent's description copied
      identically (the text you send is ignored), and only if the parent's description
      passed the quality threshold; otherwise the column is skipped and the reason given;
    - a transformed column, or one in a model that filters/regroups rows, is rejected if
      you send the parent's text unchanged: it needs a description of its own;
    - with `apply_copies`, columns that must be copied are copied even if not sent.

    Use `dry_run=True` to preview the changes without writing. Run `dbt parse` after
    writing so the manifest reflects the new descriptions. Every change (old and new text,
    copied, skipped or rejected) is recorded in the current run.

    Args:
        model_name: Exact dbt model name (e.g. 'stg_orders').
        description: Model-level description (empty = leave unchanged).
        columns: List of column dicts. Each: {'name': str, 'description': str, 'tests': list}
                 Tests can be strings ('not_null') or dicts ({'accepted_values': {'values': [...]}}).
        overwrite_descriptions: Replace existing column descriptions (default True).
        dry_run: Report the changes without writing.
        apply_copies: Also copy parent descriptions into columns that must repeat them.
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found in manifest. Cannot determine file path."

    model_path = details.get("path")
    if not model_path:
        return f"Could not determine SQL file path for '{model_name}'."

    with _flow_event("update_model_yaml", model_name) as (recorder, event):
        try:
            to_write, report = _get_planner().enforce(model_name, columns, apply_copies)
        except (RubricError, ValueError) as exc:
            event.ok = False
            event.summary["error"] = str(exc)
            return f"❌ {exc}"
        if recorder is not None:  # the state before this run changes anything
            recorder.record_snapshot(model_name, _descriptions(details))

        project_dir = _manifest.project_dir if _manifest else Path(".")
        writer = YamlWriter(str(project_dir), _schema_file_pattern)

        result = writer.upsert_model_yaml(
            model_name=model_name,
            model_sql_path=model_path,
            description=description,
            columns=to_write,
            overwrite_descriptions=overwrite_descriptions,
            dry_run=dry_run,
            patch_path=details.get("patch_path"),
        )

        if "error" in result:
            event.ok = False
            event.summary["error"] = result["error"]
            return f"❌ Failed to update YAML: {result['error']}"

        payload = {
            "status": "dry_run" if dry_run else "written",
            "model": model_name,
            "file": result["file_path"],
            "changes": result["changes"],
            **report,
            "next_step": None if dry_run else "Run `dbt parse` to refresh the manifest.",
        }
        if recorder is not None:
            event.summary.update(
                status=payload["status"],
                changed=len(result["changes"]),
                copied=len(report["copied"]),
                skipped=len(report["skipped"]),
                rejected=len(report["rejected"]),
                file=result["file_path"],
            )
            recorder.record_writes(_write_rows(model_name, result, report, dry_run))
            payload["run"] = recorder.info()
    return json.dumps(payload, ensure_ascii=False, default=str)


@mcp.tool()
def generate_model_docs(model_name: str, description: str = "") -> str:
    """
    Full E2E documentation pipeline for a single dbt model.

    This is the KILLER FEATURE of dbt-doctor. In one call it:
    1. Profiles the model's data in the warehouse
    2. Generates intelligent test suggestions from the real statistics
    3. Prepares a complete schema.yml entry

    Returns a PREVIEW of what will be written (dry-run).
    After reviewing, call `update_model_yaml` with the suggested columns to apply.

    Use this when you want the AI to automatically document a model end-to-end.

    Args:
        model_name: Exact dbt model name.
        description: Optional model-level description. If empty, only column tests are suggested.
    """
    parser = _get_manifest()
    details = parser.get_model_details(model_name)
    if not details:
        return f"Model '{model_name}' not found in manifest."

    schema = details.get("schema") or ""
    table = details.get("alias") or model_name
    model_path = details.get("path") or ""

    if not schema:
        return f"Cannot determine schema for '{model_name}'."

    project_dir = _manifest.project_dir if _manifest else Path(".")
    connector = _get_connector()
    profiler = DataProfiler(connector)
    suggester = TestSuggester()
    writer = YamlWriter(str(project_dir))
    generator = DocGenerator(profiler, suggester, writer)

    try:
        preview = generator.generate(
            model_name=model_name,
            model_sql_path=model_path,
            schema=schema,
            table=table,
            description=description,
        )
        return generator.format_preview(preview)
    except Exception as exc:
        return f"Doc generation failed for '{model_name}': {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════


def _build_connector(project_dir: str) -> BaseConnector | None:
    """Try to build a database connector from dbt profiles.yml."""
    proj_parser = DbtProjectParser(project_dir)
    if not proj_parser.load():
        logger.warning("Could not load dbt_project.yml — DB tools will be unavailable.")
        return None

    profile_name = proj_parser.profile_name
    if not profile_name:
        logger.warning("No profile name in dbt_project.yml.")
        return None

    logger.info("Found profile '%s' in dbt_project.yml", profile_name)
    prof_parser = DbtProfileParser(profile_name)
    creds = prof_parser.get_credentials()
    if not creds:
        logger.warning("Could not extract credentials from profiles.yml.")
        return None

    db_type = creds.get("type", "postgres").lower()

    if db_type == "postgres":
        try:
            from .connectors.postgres import PostgresConnector

            conn = PostgresConnector(creds)
            logger.info("PostgreSQL connector initialized.")
            return conn
        except Exception as exc:
            logger.error("Failed to create PostgreSQL connector: %s", exc)
    elif db_type == "duckdb":
        try:
            from .connectors.duckdb import DuckDBConnector

            db_path = creds.get("path", ":memory:")
            conn = DuckDBConnector(db_path)
            logger.info("DuckDB connector initialized (path=%s).", db_path)
            return conn
        except Exception as exc:
            logger.error("Failed to create DuckDB connector: %s", exc)
    elif db_type == "trino":
        try:
            from .connectors.trino import TrinoConnector

            conn = TrinoConnector(creds)
            logger.info("Trino connector initialized.")
            return conn
        except Exception as exc:
            logger.error("Failed to create Trino connector: %s", exc)
    else:
        logger.warning(
            "Unsupported DB type '%s'. Only 'postgres', 'duckdb' and 'trino' are supported. "
            "Profiling tools will be unavailable.",
            db_type,
        )
    return None


def main() -> None:
    """CLI entrypoint for dbt-doctor MCP server."""
    arg_parser = argparse.ArgumentParser(
        prog="dbt-doctor",
        description=(
            "dbt-doctor: AI-driven quality & governance MCP Server for dbt projects.\n"
            "Audit coverage, profile data, detect drift, and auto-generate documentation."
        ),
    )
    arg_parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Absolute path to your dbt project directory (contains dbt_project.yml).",
    )
    arg_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    arg_parser.add_argument(
        "--rubric-file",
        type=str,
        default=None,
        help=(
            "Path to a custom description-quality rubric JSON "
            "(default: the rubric bundled with dbt-doctor)."
        ),
    )

    arg_parser.add_argument(
        "--layer-order",
        type=str,
        default=DEFAULT_LAYER_ORDER,
        help=(
            "Comma-separated layer order used to label models in lineage reports; join "
            "labels that share a level with '|' (default: %(default)s). The dependency "
            "graph, not this order, decides how descriptions propagate."
        ),
    )

    arg_parser.add_argument(
        "--treat-casts-as-passthrough",
        action="store_true",
        help=(
            "Treat a plain cast(col as type) as a passthrough, so its description is copied "
            "identically from the parent. By default casts are not enforced."
        ),
    )

    arg_parser.add_argument(
        "--schema-file-pattern",
        type=str,
        default="{model}_schema.yaml",
        help=(
            "Name of the YAML file created for a model that has none, next to its .sql "
            "(default: %(default)s). A model already documented keeps its own file."
        ),
    )
    arg_parser.add_argument(
        "--run-store",
        type=str,
        default=None,
        help=(
            "Path of the DuckDB file that records every run "
            "(default: <project>/.dbt-doctor/dbt_doctor.duckdb)."
        ),
    )
    arg_parser.add_argument(
        "--no-run-store",
        action="store_true",
        help="Do not record runs; scores fall back to a JSON file under target/dbt_doctor/.",
    )

    args, _ = arg_parser.parse_known_args()

    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    global _manifest, _connector, _rubric_path, _layer_map, _casts_as_passthrough
    global _run_store_enabled, _run_store_path, _schema_file_pattern

    _casts_as_passthrough = args.treat_casts_as_passthrough
    _run_store_enabled = not args.no_run_store
    if (
        "{model}" not in args.schema_file_pattern
        or "/" in args.schema_file_pattern
        or "\\" in args.schema_file_pattern
        or not args.schema_file_pattern.endswith((".yaml", ".yml"))
    ):
        arg_parser.error(
            "--schema-file-pattern must contain {model}, have no folder and end in .yaml or .yml."
        )
    _schema_file_pattern = args.schema_file_pattern
    _run_store_path = Path(args.run_store) if args.run_store else None

    try:
        _layer_map = LayerMap(args.layer_order)
    except ValueError as exc:
        arg_parser.error(str(exc))

    if args.rubric_file:
        _rubric_path = Path(args.rubric_file)
        try:
            load_rubric(_rubric_path)
        except RubricError as exc:
            arg_parser.error(str(exc))

    project_dir = args.project_dir
    logger.info("Starting dbt-doctor for project at: %s", project_dir)

    _manifest = DbtManifestParser(project_dir)
    if not _manifest.load():
        logger.warning(
            "manifest.json not found or invalid. Read-only context tools will still work "
            "once you run `dbt compile` in your project."
        )

    _connector = _build_connector(project_dir)
    if _connector:
        logger.info("DB connected — all 20 tools available.")
    else:
        logger.info(
            "DB not connected — profiling/drift tools unavailable. "
            "Audit and context tools are still fully functional."
        )

    logger.info("dbt-doctor ready. Listening on stdio...")
    mcp.run()


if __name__ == "__main__":
    main()
