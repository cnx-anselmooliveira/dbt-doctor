"""Analyzers: documentation campaign, phase A (analysis only, nothing is written).

A campaign starts from a list of tables (the *seeds*). The engine decides, from the evidence
alone (the manifest and the stored scores), which tables are in good shape and which are not,
and keeps a persistent work list in the run store:

* a table is *tested* by scoring its descriptions (done by the client LLM, tool by tool);
  columns that plainly inherit from a parent whose description passed, with the same text,
  count as ok without a score of their own;
* a table that fails, by itself or because of its parents, pulls its parents into the list
  and they are tested the same way, one level at a time, until a parent is ok or the root
  (bronze) is reached; sources and landing tables stay in the list as *deferred*;
* the direct children of the listed tables are added to the list but not tested; a child is
  tested only once one of its direct parents has been *updated*, and the list then grows one
  level at a time;
* anything already tested is not tested again: scores are keyed by the description text, and
  nothing is modified in this phase.

"Updated" is detected with a checksum of a table's descriptions, read from the manifest (the
simplest source). Each child remembers the checksum of every direct parent as it last saw it,
inside its work item, so no extra table is needed.

The statuses are derived on every call, so the engine can be called again at any moment. Only
the *membership* of the list is persisted, and it only grows.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..core.run_store import RunStore
from .lineage import LineageGraph, LineagePlanner
from .score_store import description_hash, normalize_description

DEFERRED_LAYERS = frozenset({"landing", "source", "seed", "snapshot"})
MAX_ROUNDS = 200
MAX_ATTEMPTS = 5
CHILDREN_SCOPES = ("needing_docs", "seeds", "none")

NEEDS_TEST = "needs_test"
PASSED = "passed"
FAILED_SELF = "failed_self"
BLOCKED_BY_PARENT = "blocked_by_parent"
UNTESTED_CHILD = "untested_child"
DEFERRED = "deferred"
SKIPPED = "skipped"
ERROR = "error"
_FAILED = (FAILED_SELF, BLOCKED_BY_PARENT)


class CampaignError(ValueError):
    """The campaign request is invalid."""


# Keys of a work item's summary that survive the recomputation of its health.
_KEPT_SUMMARY = ("last_issued_unscored", "parents_seen", "pending_seen", "promoted")


def table_checksum(graph: LineageGraph, model_name: str) -> str:
    """A short checksum of every documented column's description of one table.

    Whitespace and column order do not matter, so it only changes when a description does.
    """
    model = graph.models[graph.by_name[model_name]]
    pairs = sorted((c.lower(), description_hash(d)) for c, d in model.columns.items())
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# TABLE HEALTH
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TableHealth:
    model: str
    layer: str
    total: int
    counts: dict[str, int]
    status: str
    ratio: float
    reason: str
    blockers: list[str]


def _column_state(planner: LineagePlanner, uid: str, column: str, entry: dict[str, Any]) -> str:
    """How one column stands: ok (own score or inherited), failing, or still unknown."""
    model = planner.graph.models[uid]
    text = model.columns_ci.get(column.lower(), "")
    if entry["rule"] == "must_copy":
        if entry["action"] == "keep":
            return "ok_inherited"
        if entry["action"] == "copy":
            return "needs_copy"
        return "waiting_parent" if entry.get("parent_status") == "unscored" else "parent_failed"
    if not normalize_description(text):
        return "missing"
    if entry.get("verdict") == "stale_copy":
        return "stale_copy"
    score = None
    if planner.store is not None:
        score = planner.store.lookup(model.name, column, text, planner.rubric_fingerprint)
    if score is None:
        return "unscored"
    return "ok_own" if score.get("status") == "ok" else "rewrite"


def table_health(planner: LineagePlanner, model_name: str, pass_ratio: float) -> TableHealth:
    """Pass/fail of one table: the share of its columns that are ok must reach ``pass_ratio``."""
    graph = planner.graph
    uid = graph.by_name[model_name]
    model = graph.models[uid]
    counts: Counter[str] = Counter()
    blockers: set[str] = set()
    for column in model.columns:
        entry = planner.plan_column(uid, column)
        state = _column_state(planner, uid, column, entry)
        counts[state] += 1
        if state in ("waiting_parent", "parent_failed") and entry.get("lineage"):
            blockers.add(entry["lineage"]["parent"])

    total = len(model.columns)
    ok = counts["ok_own"] + counts["ok_inherited"]
    ratio = ok / total if total else 0.0
    parent_dependent = counts["waiting_parent"] + counts["parent_failed"]
    failing = {k: v for k, v in counts.items() if k not in ("ok_own", "ok_inherited") and v}

    if total == 0:
        status, reason = FAILED_SELF, "no documented columns"
    elif counts["unscored"]:
        status, reason = NEEDS_TEST, f"{counts['unscored']} column(s) without a valid score"
    elif ratio >= pass_ratio:
        status, reason = PASSED, f"{ok}/{total} columns ok ({ratio:.0%})"
    elif parent_dependent and (ok + parent_dependent) / total >= pass_ratio:
        status = BLOCKED_BY_PARENT
        reason = f"{ok}/{total} ok; waiting on parent: {', '.join(sorted(blockers))}"
    else:
        status = FAILED_SELF
        reason = f"{ok}/{total} ok ({ratio:.0%}); " + ", ".join(
            f"{n} {k}" for k, n in sorted(failing.items())
        )
    return TableHealth(
        model=model_name,
        layer=model.layer,
        total=total,
        counts=dict(counts),
        status=status,
        ratio=ratio,
        reason=reason,
        blockers=sorted(blockers),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════


def validate_tables(graph: LineageGraph, tables: list[str]) -> list[str]:
    """The cleaned list of seeds, or a CampaignError that names every problem."""
    cleaned = list(dict.fromkeys(t.strip() for t in tables or [] if t and t.strip()))
    if not cleaned:
        raise CampaignError("Pass at least one table.")
    unknown = [t for t in cleaned if t not in graph.by_name]
    if unknown:
        lines = []
        for name in unknown:
            close = difflib.get_close_matches(name, list(graph.by_name), n=3, cutoff=0.6)
            lines.append(f"- '{name}'" + (f" (did you mean {', '.join(close)}?)" if close else ""))
        raise CampaignError("Unknown models, nothing was started:\n" + "\n".join(lines))
    return cleaned


def _new_item(
    model: str, role: str, depth: int, discovered_from: str | None, status: str, layer: str
) -> dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "depth": depth,
        "discovered_from": discovered_from,
        "status": status,
        "reason": None,
        "layer": layer,
        "ratio": None,
        "summary": {},
        "attempts": 0,
    }


def _signature(item: dict[str, Any]) -> str:
    keys = ("role", "depth", "status", "reason", "ratio", "attempts", "summary")
    return json.dumps([item.get(k) for k in keys], default=str, sort_keys=True)


class CampaignEngine:
    def __init__(
        self,
        graph: LineageGraph,
        planner: LineagePlanner,
        store: RunStore,
        run_id: str,
        config: dict[str, Any],
    ) -> None:
        self.graph = graph
        self.planner = planner
        self.store = store
        self.run_id = run_id
        self.pass_ratio = float(config.get("pass_ratio", 0.7))
        self.children_scope = config.get("children_scope", "needing_docs")
        self._order = {graph.models[u].name: i for i, u in enumerate(graph.topological_order())}

    # ── helpers ───────────────────────────────────────────────────────────

    def _rank(self, name: str) -> tuple[int, str]:
        return (self._order.get(name, -1), name)  # sources and unknowns first

    def _load(self) -> dict[str, dict[str, Any]]:
        return {r["model"]: r for r in self.store.load_work_items(self.run_id)}

    def _save(self, items: dict[str, dict[str, Any]], before: dict[str, str]) -> None:
        dirty = [i for name, i in items.items() if before.get(name) != _signature(i)]
        self.store.upsert_work_items(self.run_id, dirty)

    # ── start ─────────────────────────────────────────────────────────────

    def seed(self, tables: list[str]) -> dict[str, Any]:
        items = []
        for name in validate_tables(self.graph, tables):
            model = self.graph.models[self.graph.by_name[name]]
            status = DEFERRED if model.layer in DEFERRED_LAYERS else NEEDS_TEST
            item = _new_item(name, "seed", 0, None, status, model.layer)
            if status == DEFERRED:
                item["reason"] = f"{model.layer} tables are never processed"
            items.append(item)
        self.store.upsert_work_items(self.run_id, items)
        return {
            "seeds": [i["model"] for i in items],
            "deferred": [i["model"] for i in items if i["status"] == DEFERRED],
        }

    # ── advance ───────────────────────────────────────────────────────────

    def _parent_checksums(self, name: str) -> dict[str, str]:
        uid = self.graph.by_name[name]
        return {
            self.graph.models[p].name: table_checksum(self.graph, self.graph.models[p].name)
            for p in self.graph.parent_models(uid)
        }

    def _promote(self, items: dict[str, dict[str, Any]]) -> None:
        """A child is tested only when a direct parent's descriptions changed."""
        for name, item in items.items():
            if item["role"] != "descendant" or item["status"] in (SKIPPED, ERROR, DEFERRED):
                continue
            if item["status"] == NEEDS_TEST or name not in self.graph.by_name:
                continue
            seen = item["summary"].get("parents_seen") or {}
            current = self._parent_checksums(name)
            updated = sorted(p for p, c in current.items() if p in seen and seen[p] != c)
            if updated:
                item["summary"]["pending_seen"] = current
                item["summary"]["promoted"] = True
                item.update(
                    status=NEEDS_TEST, reason=f"direct parent updated: {', '.join(updated)}"
                )

    def _acknowledge(self, items: dict[str, dict[str, Any]]) -> None:
        """Once a promoted child has been looked at, its parents' versions are the new baseline."""
        for item in items.values():
            pending = item["summary"].get("pending_seen")
            if pending and item["status"] != NEEDS_TEST:
                item["summary"]["parents_seen"] = pending
                item["summary"].pop("pending_seen")

    def _refresh(self, items: dict[str, dict[str, Any]]) -> bool:
        """Re-derive every table's status; pull the parents of failing tables into the list."""
        added = False
        for name in sorted(items, key=self._rank):
            item = items[name]
            if item["status"] in (SKIPPED, ERROR, DEFERRED):
                continue
            if item["role"] == "descendant" and not item["summary"].get("promoted"):
                continue
            if name not in self.graph.by_name:
                item.update(
                    status=ERROR,
                    reason="model not found in the manifest (did the manifest change?)",
                )
                continue
            health = table_health(self.planner, name, self.pass_ratio)
            item.update(
                status=health.status,
                reason=health.reason,
                ratio=round(health.ratio, 4),
                layer=health.layer,
            )
            item["summary"] = {
                **{k: v for k, v in item["summary"].items() if k in _KEPT_SUMMARY},
                "counts": health.counts,
                "total": health.total,
            }
            if health.status not in _FAILED:
                continue
            uid = self.graph.by_name[name]
            for parent in self.graph.parent_entries(uid):
                key = (
                    parent["name"]
                    if parent["kind"] == "model"
                    else f"{parent['kind']}:{parent['name']}"
                )
                existing = items.get(key)
                deferred = parent["kind"] != "model" or parent["layer"] in DEFERRED_LAYERS
                if existing is None:
                    status = DEFERRED if deferred else NEEDS_TEST
                    new = _new_item(
                        key, "ancestor", item["depth"] + 1, name, status, parent["layer"]
                    )
                    if deferred:
                        new["reason"] = f"{parent['layer']} is never processed"
                    items[key] = new
                    added = True
                elif existing["role"] == "descendant" and not deferred:
                    existing.update(
                        role="ancestor",
                        status=NEEDS_TEST,
                        depth=item["depth"] + 1,
                        discovered_from=name,
                    )
                    added = True
        return added

    def _add_children(self, items: dict[str, dict[str, Any]]) -> None:
        """List the direct children of the tables that are (or may become) work."""
        if self.children_scope == "none":
            return
        origins = [
            n
            for n, i in items.items()
            if i["role"] == "seed"
            or (
                self.children_scope == "needing_docs"
                and i["status"] in _FAILED
                and (i["role"] == "ancestor" or i["summary"].get("promoted"))
            )
        ]
        for origin in sorted(origins, key=self._rank):
            uid = self.graph.by_name.get(origin)
            if uid is None:
                continue
            base = items[origin]
            for child_uid in self.graph.child_models(uid):
                child = self.graph.models[child_uid]
                if child.name in items:
                    continue
                status = DEFERRED if child.layer in DEFERRED_LAYERS else UNTESTED_CHILD
                depth = 1 if base["role"] != "descendant" else base["depth"] + 1
                item = _new_item(child.name, "descendant", depth, origin, status, child.layer)
                item["summary"]["parents_seen"] = self._parent_checksums(child.name)
                items[child.name] = item

    def advance(self, limit: int = 10) -> dict[str, Any]:
        """Recompute the list from the evidence and return the next tables to test."""
        items = self._load()
        if not items:
            raise CampaignError("The campaign has no tables.")
        before = {name: _signature(i) for name, i in items.items()}

        self._promote(items)
        for _ in range(MAX_ROUNDS):
            if not self._refresh(items):
                break
        self._add_children(items)
        self._acknowledge(items)

        tasks: list[dict[str, Any]] = []
        for name in sorted(items, key=self._rank):
            item = items[name]
            if item["status"] != NEEDS_TEST or len(tasks) >= max(1, limit):
                continue
            unscored = item["summary"].get("counts", {}).get("unscored", 0)
            last = item["summary"].get("last_issued_unscored")
            if last is not None:
                # Asked again with no progress counts against the table; progress resets it.
                item["attempts"] = item["attempts"] + 1 if unscored >= last else 0
            item["summary"]["last_issued_unscored"] = unscored
            if item["attempts"] >= MAX_ATTEMPTS:
                item.update(
                    status=ERROR,
                    reason=f"no progress after {MAX_ATTEMPTS} requests: scores are missing",
                )
                continue
            tasks.append(
                {
                    "type": "evaluate",
                    "model": name,
                    "role": item["role"],
                    "layer": item["layer"],
                    "columns_without_score": unscored,
                    "why": "selected"
                    if item["role"] == "seed"
                    else f"parent of {item['discovered_from']}",
                    "call": f"evaluate_model_descriptions(model_name='{name}')",
                }
            )
        self._save(items, before)

        pending = sum(1 for i in items.values() if i["status"] == NEEDS_TEST)
        phase = "A_in_progress" if pending else "A_complete"
        return {
            "phase": phase,
            "tasks": tasks,
            "remaining_to_test": pending,
            "counts": dict(Counter(i["status"] for i in items.values())),
            "instructions": self._instructions(phase),
        }

    @staticmethod
    def _instructions(phase: str) -> str:
        if phase == "A_complete":
            return (
                "Phase A is complete: every table that had to be tested was. Review the whole "
                "list with get_campaign_status. Generating the documentation (phase B) is not "
                "available in this version, and nothing has been modified."
            )
        return (
            "For each task in order: call evaluate_model_descriptions, judge every column on "
            "every rubric question, and send the scores with submit_description_scores, "
            "following next_offset until the model is complete. Do not write descriptions. "
            "Then call get_next_work again: the list grows as failing tables reach their "
            "parents, and tables already scored are never asked for again."
        )

    # ── status, skip ──────────────────────────────────────────────────────

    def status(self, status: str = "", offset: int = 0, limit: int = 50) -> dict[str, Any]:
        items = self._load()
        rows = sorted(items.values(), key=lambda i: self._rank(i["model"]))
        if status:
            rows = [r for r in rows if r["status"] == status]
        page = rows[max(offset, 0) : max(offset, 0) + max(1, limit)]
        end = max(offset, 0) + len(page)
        pending = sum(1 for i in items.values() if i["status"] == NEEDS_TEST)
        return {
            "phase": "A_in_progress" if pending else "A_complete",
            "total": len(items),
            "by_status": dict(Counter(i["status"] for i in items.values())),
            "by_role": dict(Counter(i["role"] for i in items.values())),
            "filter": status or None,
            "matching": len(rows),
            "offset": offset,
            "next_offset": end if end < len(rows) else None,
            "tables": [
                {
                    "model": r["model"],
                    "role": r["role"],
                    "depth": r["depth"],
                    "layer": r["layer"],
                    "status": r["status"],
                    "ratio": r["ratio"],
                    "reason": r["reason"],
                    "discovered_from": r["discovered_from"],
                }
                for r in page
            ],
        }

    def skip(self, model: str, reason: str) -> dict[str, Any]:
        items = self._load()
        if model not in items:
            raise CampaignError(f"'{model}' is not in this campaign.")
        before = {name: _signature(i) for name, i in items.items()}
        items[model].update(status=SKIPPED, reason=reason or "skipped by the user")
        self._save(items, before)
        return {"model": model, "status": SKIPPED, "reason": items[model]["reason"]}
