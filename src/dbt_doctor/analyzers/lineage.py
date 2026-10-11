"""Analyzers: column description lineage across dbt layers.

Builds, from manifest.json, which column of a model comes from which column of its parent
and decides what the documentation of that column must look like:

* a plain passthrough that keeps the rows (no filter, regroup, dedupe, union or limit) must
  repeat the parent's description *identically*, and only if the parent's description
  already passed the quality threshold;
* a column that is transformed, or that goes through a row-altering model, must have its
  own description: copying the parent's text is a stale copy;
* when the lineage cannot be read with confidence, nothing is enforced.

The direction of propagation always follows the real dependency graph. The layer order is
only used to label models, to pick the root (bronze) and to group reports.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .score_store import ScoreStore, normalize_description
from .sql_shape import SqlShape, analyze_sql

DEFAULT_LAYER_ORDER = "landing,bronze,stg,silver,intermediary,gold|view,az"
DEFAULT_ROOT_LAYER = "bronze"
_PREFIX_LAYERS = (("landing_", "landing"), ("raw_", "bronze"), ("stg_", "stg"), ("az_", "az"))


class LayerMap:
    """Labels models with a layer and ranks the layers.

    ``order`` is a comma-separated list; labels joined with ``|`` share a level. A label
    that appears more than once keeps its first position (the rank is informational only).
    """

    def __init__(self, order: str = DEFAULT_LAYER_ORDER, root: str = DEFAULT_ROOT_LAYER) -> None:
        self.rank: dict[str, int] = {}
        for level, group in enumerate(order.split(",")):
            for label in group.split("|"):
                label = label.strip().lower()
                if label:
                    self.rank.setdefault(label, level)
        root = root.lower()
        if root not in self.rank:
            raise ValueError(f"Root layer '{root}' is not in the layer order '{order}'.")
        self.root = root
        self._root_rank = self.rank[root]

    def layer_of(self, name: str, fqn: list[str]) -> str:
        lowered = name.lower()
        for prefix, label in _PREFIX_LAYERS:
            if lowered.startswith(prefix) and label in self.rank:
                return label
        for segment in reversed([s.lower() for s in fqn[1:-1]]):
            if segment in self.rank:
                return segment
        return "unknown"

    def is_root_or_below(self, layer: str) -> bool:
        return layer in self.rank and self.rank[layer] <= self._root_rank


@dataclass
class ModelInfo:
    uid: str
    name: str
    layer: str
    raw_code: str
    columns: dict[str, str]  # declared columns -> description
    parents: list[str]  # unique_ids of model/source/seed/snapshot parents
    columns_ci: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.columns_ci = {k.lower(): v for k, v in self.columns.items()}


@dataclass
class ColumnEdge:
    """Where a column comes from."""

    parent: str | None
    parent_column: str | None
    kind: str  # "passthrough" | "renamed" | "transformed" | "unknown"
    effects: list[str]  # row-altering effects of the model
    confidence: str  # "high" | "low"
    evidence: str = ""


class LineageGraph:
    def __init__(self, manifest: dict[str, Any], layers: LayerMap | None = None) -> None:
        self.layers = layers or LayerMap()
        self.models: dict[str, ModelInfo] = {}
        self.sources: dict[str, dict[str, str]] = {}
        self._shapes: dict[str, SqlShape] = {}

        for uid, node in manifest.get("nodes", {}).items():
            if node.get("resource_type") != "model":
                continue
            parents = [
                p
                for p in node.get("depends_on", {}).get("nodes", [])
                if p.startswith(("model.", "source.", "seed.", "snapshot."))
            ]
            name = node.get("name", uid)
            self.models[uid] = ModelInfo(
                uid=uid,
                name=name,
                layer=self.layers.layer_of(name, node.get("fqn", [])),
                raw_code=node.get("raw_code") or node.get("raw_sql") or "",
                columns={
                    c: (d.get("description") or "") for c, d in (node.get("columns") or {}).items()
                },
                parents=parents,
            )
        for uid, src in manifest.get("sources", {}).items():
            self.sources[uid] = {
                c: (d.get("description") or "") for c, d in (src.get("columns") or {}).items()
            }
        self.by_name = {m.name: uid for uid, m in self.models.items()}

    # ── model level ───────────────────────────────────────────────────────

    def shape(self, uid: str) -> SqlShape:
        if uid not in self._shapes:
            self._shapes[uid] = analyze_sql(self.models[uid].raw_code)
        return self._shapes[uid]

    def parent_models(self, uid: str) -> list[str]:
        return [p for p in self.models[uid].parents if p in self.models]

    def topological_order(self) -> list[str]:
        """Parents before children; ties broken by name so the order is stable."""
        indegree = {uid: len(self.parent_models(uid)) for uid in self.models}
        children: dict[str, list[str]] = {uid: [] for uid in self.models}
        for uid in self.models:
            for p in self.parent_models(uid):
                children[p].append(uid)
        ready = sorted(
            (u for u, d in indegree.items() if d == 0), key=lambda u: self.models[u].name
        )
        order: list[str] = []
        while ready:
            uid = ready.pop(0)
            order.append(uid)
            for child in children[uid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort(key=lambda u: self.models[u].name)
        return order

    # ── column level ──────────────────────────────────────────────────────

    def parent_description(self, parent_uid: str | None, column: str | None) -> str | None:
        if not parent_uid or not column:
            return None
        if parent_uid in self.models:
            return self.models[parent_uid].columns_ci.get(column.lower())
        if parent_uid in self.sources:
            return {k.lower(): v for k, v in self.sources[parent_uid].items()}.get(column.lower())
        return None

    def _parent_column_names(self, parent_uid: str) -> set[str]:
        if parent_uid in self.models:
            return set(self.models[parent_uid].columns_ci)
        return {c.lower() for c in self.sources.get(parent_uid, {})}

    def _source_of_expression(self, parent_uid: str, expression: str, fallback: str) -> str:
        """The parent column a transformed column is built from, when it is unambiguous."""
        referenced = set(re.findall(r"[a-z_][a-z0-9_]*", expression.lower()))
        candidates = referenced & self._parent_column_names(parent_uid)
        return next(iter(candidates)) if len(candidates) == 1 else fallback

    def edge(self, uid: str, column: str) -> ColumnEdge | None:
        """Lineage of one column, or None for models at the root layer or below."""
        model = self.models[uid]
        if self.layers.is_root_or_below(model.layer):
            return None
        shape = self.shape(uid)
        if len(model.parents) != 1 or not shape.readable:
            reasons = list(shape.uncertain)
            if len(model.parents) != 1:
                reasons.append(f"{len(model.parents)} parents")
            if len(shape.refs) != 1:
                reasons.append(f"{len(shape.refs)} tables referenced")
            if shape.kind == "unknown" and not reasons:
                reasons.append("select list not readable")
            return ColumnEdge(None, None, "unknown", [], "low", ", ".join(reasons))

        parent, wanted = model.parents[0], column.lower()
        for item in shape.items:
            if item.output and item.output.lower() == wanted:
                if item.kind == "transformed":
                    source = self._source_of_expression(parent, item.expression, column)
                    return ColumnEdge(
                        parent, source, "transformed", shape.effects, "high", item.expression[:400]
                    )
                return ColumnEdge(parent, item.source, item.kind, shape.effects, "high")
        if any(i.kind == "star" for i in shape.items):
            return ColumnEdge(parent, column, "passthrough", shape.effects, "high", "select *")
        return ColumnEdge(None, None, "unknown", [], "low", "column not found in the select list")


# ═══════════════════════════════════════════════════════════════════════════
# PLANNER
# ═══════════════════════════════════════════════════════════════════════════

_ACTIONS = ("copy", "write_new", "blocked", "keep")
_PARENT_TEXT_LIMIT = 400


class LineagePlanner:
    """Decides, per column, what its description must be, and enforces it on writes."""

    def __init__(
        self,
        graph: LineageGraph,
        store: ScoreStore | None,
        rubric_fingerprint: str,
        casts_are_passthrough: bool = False,
    ) -> None:
        self.graph = graph
        self.store = store
        self.rubric_fingerprint = rubric_fingerprint
        # A pure ``cast(col as type)`` keeps the meaning but may change the format
        # (e.g. timestamp to date), so by default no description is enforced for it.
        self.casts_are_passthrough = casts_are_passthrough

    def _resolve(self, name: str) -> str:
        uid = self.graph.by_name.get(name)
        if uid is None:
            raise ValueError(f"Model '{name}' not found in the manifest.")
        return uid

    def _parent_status(self, parent_uid: str, column: str, text: str | None) -> str:
        if not normalize_description(text):
            return "missing"
        parent = self.graph.models.get(parent_uid)
        if parent is None:
            return "source"  # sources are never scored
        entry = None
        if self.store:
            entry = self.store.lookup(parent.name, column, text or "", self.rubric_fingerprint)
        if entry is None:
            return "unscored"
        return "passed" if entry.get("status") == "ok" else "failed"

    def plan_column(self, uid: str, column: str) -> dict[str, Any]:
        graph = self.graph
        model = graph.models[uid]
        current = model.columns_ci.get(column.lower(), "")
        empty = not normalize_description(current)
        entry: dict[str, Any] = {"column": column, "layer": model.layer}

        edge = graph.edge(uid, column)
        if edge is None or edge.confidence == "low":
            reason = "root layer: described on its own" if edge is None else edge.evidence
            entry.update(
                rule="free",
                action="write_new" if empty else "keep",
                verdict="missing" if empty else ("root" if edge is None else "uncertain"),
                reason=reason,
            )
            if edge is not None:
                entry["lineage"] = {"kind": "unknown", "evidence": edge.evidence}
            return entry

        parent_model = graph.models.get(edge.parent or "")
        parent_name = parent_model.name if parent_model else (edge.parent or "").split(".")[-1]
        parent_text = graph.parent_description(edge.parent, edge.parent_column)
        entry["lineage"] = {
            "parent": parent_name,
            "parent_column": edge.parent_column,
            "kind": edge.kind,
            "effects": edge.effects,
            "evidence": edge.evidence,
        }
        entry["_parent_text"] = parent_text

        if edge.kind == "cast" and not edge.effects and not self.casts_are_passthrough:
            entry.update(
                rule="free",
                action="write_new" if empty else "keep",
                verdict="missing" if empty else "cast",
                reason="type cast only: the description is not enforced; align it with the "
                "parent by hand if the cast does not change the format",
            )
            return entry

        plain = edge.kind in ("passthrough", "renamed") or (
            edge.kind == "cast" and self.casts_are_passthrough
        )
        if plain and not edge.effects and parent_model is None:
            # The parent is a source (or another external object), not a dbt model: its
            # descriptions are never scored, so there is nothing to copy from and the
            # column is described on its own, like the root layer.
            entry.update(
                rule="free",
                action="write_new" if empty else "keep",
                verdict="missing" if empty else "source_parent",
                reason="the parent is a source, not a dbt model: described on its own",
            )
            return entry

        if plain and not edge.effects:
            status = self._parent_status(edge.parent or "", edge.parent_column or "", parent_text)
            entry.update(rule="must_copy", parent_status=status)
            if status != "passed":
                entry.update(
                    action="blocked",
                    verdict="blocked",
                    reason=_BLOCK_REASONS[status].format(parent=parent_name),
                )
            elif normalize_description(current) == normalize_description(parent_text):
                entry.update(action="keep", verdict="ok", reason="identical to the parent")
            else:
                entry.update(
                    action="copy",
                    verdict="missing" if empty else "inconsistent",
                    reason="plain passthrough: must repeat the parent's description",
                    copy_text=(parent_text or "").strip(),
                )
            return entry

        why = (
            f"model alters rows ({', '.join(edge.effects)})"
            if edge.kind != "transformed"
            else "column is transformed"
        )
        entry.update(rule="must_differ", reason=why)
        if empty:
            entry.update(action="write_new", verdict="missing")
        elif parent_text and normalize_description(current) == normalize_description(parent_text):
            entry.update(action="write_new", verdict="stale_copy")
        else:
            entry.update(action="keep", verdict="ok")
        return entry

    # ── reports ───────────────────────────────────────────────────────────

    def plan_model(self, name: str) -> dict[str, Any]:
        uid = self._resolve(name)
        graph, model = self.graph, self.graph.models[uid]
        entries = [self.plan_column(uid, c) for c in model.columns]
        blocked_by = sorted(
            {
                e["lineage"]["parent"]
                for e in entries
                if e.get("action") == "blocked" and "lineage" in e
            }
        )
        shape = graph.shape(uid)
        return {
            "model": name,
            "layer": model.layer,
            "shape": {
                "kind": shape.kind,
                "effects": shape.effects,
                "uncertain": shape.uncertain,
                "refs": shape.refs,
            },
            "parents": [graph.models[p].name for p in graph.parent_models(uid)],
            "summary": dict(Counter(e["action"] for e in entries)),
            "blocked_by": blocked_by,
            "columns": [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries],
        }

    def consistency(self, layer: str = "", offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Paged overview of every model (parents first) with counts per action."""
        graph = self.graph
        wanted = layer.strip().lower()
        order = [
            u for u in graph.topological_order() if not wanted or graph.models[u].layer == wanted
        ]
        totals: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        for position, uid in enumerate(order):
            model = graph.models[uid]
            entries = [self.plan_column(uid, c) for c in model.columns]
            counts = Counter(e["action"] for e in entries)
            totals.update(counts)
            if offset <= position < offset + limit:
                rows.append(
                    {
                        "model": model.name,
                        "layer": model.layer,
                        "columns": len(entries),
                        "counts": dict(counts),
                        "blocked_by": sorted(
                            {
                                e["lineage"]["parent"]
                                for e in entries
                                if e.get("action") == "blocked" and "lineage" in e
                            }
                        ),
                    }
                )
        end = offset + len(rows)
        return {
            "layer": wanted or "all",
            "total_models": len(order),
            "offset": offset,
            "next_offset": end if end < len(order) else None,
            "totals": {a: totals.get(a, 0) for a in _ACTIONS},
            "models": rows,
        }

    def upstream_context(self, name: str) -> dict[str, dict[str, Any]]:
        """What the client needs to judge each column against its parent.

        Keyed by lower-case column name. Root-layer models have no upstream and yield an
        empty dict. Per column: the parent model and column, how it is built, the parent's
        description and, when the column is built by an expression (here or further up),
        that expression.
        """
        graph = self.graph
        uid = self._resolve(name)
        context: dict[str, dict[str, Any]] = {}
        for column in graph.models[uid].columns:
            edge = graph.edge(uid, column)
            if edge is None:
                continue
            if edge.confidence == "low" or edge.parent is None:
                context[column.lower()] = {"kind": "unknown", "why": edge.evidence}
                continue
            parent_model = graph.models.get(edge.parent)
            parent_text = (graph.parent_description(edge.parent, edge.parent_column) or "").strip()
            info: dict[str, Any] = {
                "parent": parent_model.name if parent_model else edge.parent.split(".")[-1],
                "parent_is_source": parent_model is None,
                "parent_column": edge.parent_column,
                "kind": edge.kind,
                "parent_description": parent_text[:_PARENT_TEXT_LIMIT] or None,
            }
            if edge.effects:
                info["model_effects"] = edge.effects
            steps = self.trace(name, column)
            built = next((s for s in steps if s.get("kind") == "transformed"), None)
            if built:
                info["built_by"] = {"model": built["model"], "expression": built.get("evidence")}
            if len(steps) > 1:
                info["chain"] = [
                    {"model": s["model"], "column": s["column"], "kind": s.get("kind", "root")}
                    for s in steps[1:]
                ]
            context[column.lower()] = info
        return context

    def trace(self, name: str, column: str, max_depth: int = 25) -> list[dict[str, Any]]:
        graph = self.graph
        uid, col = self._resolve(name), column
        chain: list[dict[str, Any]] = []
        for _ in range(max_depth):
            model = graph.models[uid]
            step: dict[str, Any] = {"model": model.name, "layer": model.layer, "column": col}
            step["has_description"] = bool(normalize_description(model.columns_ci.get(col.lower())))
            chain.append(step)
            edge = graph.edge(uid, col)
            if edge is None:
                step["stop"] = "root layer"
                break
            step.update(kind=edge.kind, effects=edge.effects, confidence=edge.confidence)
            if edge.evidence:
                step["evidence"] = edge.evidence
            if edge.confidence == "low" or edge.parent is None:
                step["stop"] = "lineage not readable"
                break
            if edge.parent not in graph.models:
                step["stop"] = "parent is a source"
                break
            if edge.kind == "transformed":
                step["stop"] = "column is built by an expression here"
                break
            uid, col = edge.parent, edge.parent_column or col
        return chain

    # ── write barrier ─────────────────────────────────────────────────────

    def enforce(
        self, name: str, columns: list[dict[str, Any]] | None, apply_copies: bool = True
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Apply the description rules to a write request.

        Returns the columns that may be written and a report of what was copied, skipped
        or rejected. Tests and other keys of each column are always kept.
        """
        uid = self._resolve(name)
        model = self.graph.models[uid]
        plan = {c.lower(): self.plan_column(uid, c) for c in model.columns}
        report: dict[str, list[dict[str, Any]]] = {"copied": [], "skipped": [], "rejected": []}
        final: list[dict[str, Any]] = []
        seen: set[str] = set()

        for item in columns or []:
            col = item.get("name")
            if not col:
                continue
            seen.add(col.lower())
            supplied = (item.get("description") or "").strip()
            entry = plan.get(col.lower()) or self.plan_column(uid, col)
            out = {k: v for k, v in item.items() if k != "description"}

            if entry["rule"] == "must_copy":
                if entry["action"] == "blocked":
                    report["skipped"].append({"column": col, "reason": entry["reason"]})
                elif entry["action"] == "copy":
                    out["description"] = entry["copy_text"]
                    report["copied"].append(
                        {
                            "column": col,
                            "from": f"{entry['lineage']['parent']}.{entry['lineage']['parent_column']}",
                            "supplied_text_ignored": bool(
                                supplied
                                and normalize_description(supplied)
                                != normalize_description(entry["copy_text"])
                            ),
                        }
                    )
            elif entry["rule"] == "must_differ":
                parent_text = entry.get("_parent_text")
                if (
                    supplied
                    and parent_text
                    and (normalize_description(supplied) == normalize_description(parent_text))
                ):
                    report["rejected"].append(
                        {
                            "column": col,
                            "reason": f"identical to the parent's text, but {entry['reason']}; "
                            "write a description of its own",
                        }
                    )
                elif supplied:
                    out["description"] = supplied
            elif supplied:
                out["description"] = supplied
            final.append(out)

        if apply_copies:
            for key, entry in plan.items():
                if key in seen or entry["action"] != "copy":
                    continue
                final.append({"name": entry["column"], "description": entry["copy_text"]})
                report["copied"].append(
                    {
                        "column": entry["column"],
                        "from": f"{entry['lineage']['parent']}.{entry['lineage']['parent_column']}",
                        "supplied_text_ignored": False,
                    }
                )
        return final, report


_BLOCK_REASONS = {
    "missing": "the parent column in {parent} has no description: document it first",
    "unscored": "the parent column in {parent} has not been scored: run evaluate_model_descriptions on {parent} first",
    "failed": "the parent column in {parent} did not pass the quality threshold: rewrite it first",
    "source": "the parent is a source, whose descriptions are not scored",
}
