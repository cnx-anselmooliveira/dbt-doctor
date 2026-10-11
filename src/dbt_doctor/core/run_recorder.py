"""Core: records the documentation flow into the run store, never breaking a tool.

One ``RunRecorder`` lives per server process and keeps the *current run*. Tools wrap their
work in ``recorder.event(...)``; whatever they record inside is linked to that event and run.
Every recording call is best-effort: if the database is locked or unavailable the tool keeps
working and ``status()`` explains why nothing was recorded.

The recorder also implements the two methods the score consumers need (``record_many`` and
``lookup``), so it can replace the JSON score store.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..analyzers.score_store import ScoreStore, description_hash
from .run_store import RunStore, StoreUnavailable

logger = logging.getLogger(__name__)

# After the store fails, skip recording for this long instead of waiting for the lock again.
COOLDOWN_SECONDS = 5.0


@dataclass
class Event:
    tool: str
    model: str | None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ok: bool = True
    summary: dict[str, Any] = field(default_factory=dict)


class RunRecorder:
    def __init__(self, store: RunStore, context: Callable[[], dict[str, Any]]) -> None:
        """``context()`` returns ``{"manifest": {...}, "rubric_fp": str, "params": {...}}``."""
        self.store = store
        self._context = context
        self.run_id: str | None = None
        self._event_id: str | None = None
        self._manifest_cache: tuple[Any, str] | None = None
        self._score_cache: dict[tuple[str, str, str, str], dict[str, Any]] | None = None
        self._down_until = 0.0
        self.cooldown = COOLDOWN_SECONDS
        self.last_error: str | None = None

    # ── plumbing ──────────────────────────────────────────────────────────

    def _safe(self, fn: Callable[[], Any], default: Any = None) -> Any:
        if time.monotonic() < self._down_until:
            return default  # the store failed a moment ago: do not wait for it again
        try:
            result = fn()
            self.last_error = None
            return result
        except Exception as exc:
            if isinstance(exc, StoreUnavailable):
                self._down_until = time.monotonic() + self.cooldown
            self.last_error = str(exc)
            logger.warning("Run store: %s", exc)
            return default

    def status(self) -> str:
        return "ok" if self.last_error is None else f"unavailable: {self.last_error}"

    def info(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "recording": self.status()}

    def _manifest_id(self) -> str | None:
        info = self._context().get("manifest") or {}
        key = (info.get("path"), info.get("mtime"))
        if self._manifest_cache and self._manifest_cache[0] == key:
            return self._manifest_cache[1]
        manifest_id = self.store.register_manifest(info)
        self._manifest_cache = (key, manifest_id)
        return manifest_id

    # ── runs ──────────────────────────────────────────────────────────────

    def _new_run(
        self, goal: str, target_model: str, extra_params: dict[str, Any] | None = None
    ) -> str:
        ctx = self._context()
        manifest_id = self._manifest_id()
        params = {**(ctx.get("params") or {}), **(extra_params or {})}
        run_id = self.store.create_run(
            goal, target_model, manifest_id, ctx.get("rubric_fp", ""), params
        )
        self.store.close_open_runs(except_run=run_id)
        self.run_id = run_id
        return run_id

    def ensure_run(self, model: str | None = None) -> str | None:
        """The current run, opening an implicit one on first use."""
        if self.run_id is None:
            self._safe(lambda: self._new_run("(implicit) documentation flow", model or ""))
        return self.run_id

    def start_run(
        self,
        goal: str = "",
        target_model: str = "",
        resume: bool = False,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Open a run with an explicit context, or resume the latest open one."""
        result: dict[str, Any] = {"resumed": False, "manifest_changed": False}

        def go() -> None:
            if resume:
                run = self.store.latest_open_run(target_model)
                if run:
                    self.run_id = run["run_id"]
                    result["resumed"] = True
                    result["manifest_changed"] = run.get("manifest_id") != self._manifest_id()
                    return
            self._new_run(goal or "(no goal given)", target_model, params)

        self._safe(go)
        result.update(self.info())
        if self.run_id is None:
            result["error"] = self.last_error
        return result

    # ── events ────────────────────────────────────────────────────────────

    @contextmanager
    def event(self, tool: str, model: str | None = None) -> Iterator[Event]:
        ev = Event(tool=tool, model=model)
        self._score_cache = None
        with ExitStack() as stack:
            # One connection for the whole call; if it cannot be opened, every recording
            # below fails softly and the tool still works.
            self._safe(lambda: stack.enter_context(self.store.session()))
            self.ensure_run(model)
            previous = self._event_id
            self._event_id = ev.event_id
            try:
                yield ev
            except Exception:
                ev.ok = False
                raise
            finally:
                self._event_id = previous
                self._score_cache = None
                if self.run_id:
                    self._safe(
                        lambda: self.store.add_event(
                            ev.event_id,
                            self.run_id,  # type: ignore[arg-type]
                            tool,
                            model,
                            ev.ok,
                            self._manifest_id(),
                            ev.summary,
                        )
                    )

    # ── artifacts ─────────────────────────────────────────────────────────

    def record_snapshot(self, model: str, descriptions: dict[str, str]) -> None:
        """Each column's description the first time this run touches the model."""
        if not self.ensure_run(model):
            return
        items = [(c, d or "") for c, d in descriptions.items()]
        self._safe(lambda: self.store.add_snapshots(self.run_id, self._event_id, model, items))

    def record_lineage(self, model: str, edges: list[dict[str, Any]]) -> None:
        if self.ensure_run(model):
            self._safe(lambda: self.store.add_lineage_edges(self.run_id, model, edges))

    def record_plan(self, model: str, entries: list[dict[str, Any]]) -> None:
        if self.ensure_run(model):
            self._safe(
                lambda: self.store.add_plan_items(self.run_id, self._event_id, model, entries)
            )

    def record_writes(self, rows: list[dict[str, Any]]) -> None:
        if rows and self.ensure_run(rows[0].get("model")):
            self._safe(lambda: self.store.add_writes(self.run_id, self._event_id, rows))

    # ── score backend (same interface as ScoreStore) ──────────────────────

    def record_many(self, model: str, rubric_fingerprint: str, items: list[dict[str, Any]]) -> None:
        if items and self.ensure_run(model):
            self._score_cache = None  # the scores changed
            self._safe(
                lambda: self.store.add_scores(
                    self.run_id, self._event_id, model, rubric_fingerprint, items
                )
            )

    def record(
        self,
        model: str,
        column: str,
        description: str,
        rubric_fingerprint: str,
        total: float,
        status: str,
    ) -> None:
        item = {"column": column, "description": description, "total": total, "status": status}
        self.record_many(model, rubric_fingerprint, [item])

    def _load_score_cache(self) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        rows = self.store.query(
            "SELECT model, column_name, hash, rubric_fp, total, status, scored_at "
            "FROM v_latest_scores"
        )
        return {
            (r["model"], r["column_name"].lower(), r["hash"], r["rubric_fp"]): {
                "hash": r["hash"],
                "rubric": r["rubric_fp"],
                "total": r["total"],
                "status": r["status"],
                "scored_at": r["scored_at"],
            }
            for r in rows
        }

    def lookup(
        self, model: str, column: str, description: str, rubric_fingerprint: str
    ) -> dict[str, Any] | None:
        if self._event_id is None:  # outside a tool call: always read the current state
            return self._safe(
                lambda: self.store.latest_score(model, column, description, rubric_fingerprint)
            )
        # Inside a tool call the planner asks for thousands of scores: read them once.
        if self._score_cache is None:
            self._score_cache = self._safe(self._load_score_cache)
            if self._score_cache is None:
                return None
        return self._score_cache.get(
            (model, column.lower(), description_hash(description), rubric_fingerprint)
        )

    # ── migration of the previous JSON score store ────────────────────────

    def import_legacy_scores(self, json_path: Path) -> int:
        """Import ``description_scores.json`` once, then rename it. Returns rows imported."""
        legacy = ScoreStore(json_path)
        if not json_path.exists():
            return 0
        entries = legacy._load().get("entries", {})
        by_model: dict[str, list[dict[str, Any]]] = {}
        for key, entry in entries.items():
            model, _, column = key.rpartition(".")
            if not model:
                continue
            by_model.setdefault(model, []).append(
                {
                    "column": column,
                    "hash": entry.get("hash"),
                    "rubric": entry.get("rubric"),
                    "total": entry.get("total"),
                    "status": entry.get("status"),
                }
            )

        def go() -> int:
            run_id = self.store.create_run(
                f"import of {json_path.name}", "", None, "", {"imported": True}
            )
            imported = self.store.import_hashed_scores(run_id, by_model)
            self.store.close_run(run_id)
            json_path.rename(json_path.with_suffix(".json.migrated"))
            return imported

        return self._safe(go, 0) or 0


__all__ = ["Event", "RunRecorder"]
