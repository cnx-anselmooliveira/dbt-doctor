"""Analyzers: durable store of description-quality scores.

The evaluation flow keeps its batch progress in a temporary file, but the write step needs
to know whether a *parent* column's description already passed the quality threshold. This
store keeps that verdict. An entry is only valid while the description text and the rubric
are the same, so editing either one invalidates the score by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STORE_FILENAME = "description_scores.json"


def normalize_description(text: str | None) -> str:
    """Whitespace-insensitive form, so folded (``>``) and plain YAML scalars compare equal."""
    return " ".join((text or "").split())


def description_hash(text: str | None) -> str:
    return hashlib.sha256(normalize_description(text).encode("utf-8")).hexdigest()[:16]


class ScoreStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "entries": {}}
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _key(model: str, column: str) -> str:
        return f"{model}.{column.lower()}"

    def record_many(self, model: str, rubric_fingerprint: str, items: list[dict[str, Any]]) -> None:
        """Store several scores with a single write.

        Each item: ``{"column", "description", "total", "status"}``.
        """
        if not items:
            return
        data = self._load()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for item in items:
            data["entries"][self._key(model, item["column"])] = {
                "hash": description_hash(item["description"]),
                "rubric": rubric_fingerprint,
                "total": item["total"],
                "status": item["status"],
                "scored_at": now,
            }
        self._save(data)

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

    def lookup(
        self, model: str, column: str, description: str, rubric_fingerprint: str
    ) -> dict[str, Any] | None:
        """The stored score, or None when there is none or it no longer applies."""
        entry = self._load()["entries"].get(self._key(model, column))
        if not entry:
            return None
        if entry.get("hash") != description_hash(description):
            return None
        if entry.get("rubric") != rubric_fingerprint:
            return None
        return entry
