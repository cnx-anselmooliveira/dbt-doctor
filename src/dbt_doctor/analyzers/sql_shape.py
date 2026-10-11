"""Analyzers: structural reading of a dbt model's SQL, without a SQL parser (level N1).

Works on the raw dbt code (Jinja included), so it does not need ``dbt compile``. It answers
two questions with deliberately conservative heuristics:

* does the model change the rows (filter, regroup, dedupe, union, limit)? Those are the
  *certain* effects. Joins, CTEs and unreadable Jinja make the model *uncertain* instead.
* for a simple single-source ``SELECT``, which output column comes from which source
  column, and is it a plain passthrough, a rename or an expression?

Filters inside ``{% if is_incremental() %}`` blocks only limit the load window, so they are
dropped before analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INCREMENTAL_RE = re.compile(
    r"\{%-?\s*if\s+is_incremental\(\)\s*-?%\}(.*?)(?:\{%-?\s*else\s*-?%\}(.*?))?\{%-?\s*endif\s*-?%\}",
    re.DOTALL,
)
_COMMENT_JINJA_RE = re.compile(r"\{#.*?#\}", re.DOTALL)
_CONFIG_RE = re.compile(r"\{\{-?\s*config\s*\(.*?\)\s*-?\}\}", re.DOTALL)
_REF_RE = re.compile(
    r"\{\{-?\s*ref\(\s*(?:['\"][^'\"]+['\"]\s*,\s*)?['\"]([^'\"]+)['\"][^)]*\)\s*-?\}\}"
)
_SOURCE_RE = re.compile(
    r"\{\{-?\s*source\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*-?\}\}"
)
_THIS_RE = re.compile(r"\{\{-?\s*this\s*-?\}\}")
_JINJA_EXPR_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_JINJA_TAG_RE = re.compile(r"\{%.*?%\}", re.DOTALL)

_EFFECT_PATTERNS = {
    "where": re.compile(r"\bwhere\b"),
    "group_by": re.compile(r"\bgroup\s+by\b"),
    "distinct": re.compile(r"\bselect\s+distinct\b"),
    "union": re.compile(r"\bunion\b"),
    "limit": re.compile(r"\blimit\b"),
    "having": re.compile(r"\bhaving\b"),
    "qualify": re.compile(r"\bqualify\b"),
}
_JOIN_RE = re.compile(r"\bjoin\b")
_PLAIN_RE = re.compile(r'^(?:"?[a-z_][a-z0-9_]*"?\.)?"?([a-z_][a-z0-9_]*)"?$')
_CAST_RE = re.compile(
    r'^(?:try_)?cast\(\s*(?:"?[a-z_][a-z0-9_]*"?\.)?"?([a-z_][a-z0-9_]*)"?\s+as\s+([a-z0-9_ ,()]+)\)$'
)
_TEMPORAL_TARGET_RE = re.compile(r"^(date|time|timestamp|interval)")
_ALIAS_RE = re.compile(r'^(.*\S)\s+as\s+"?([a-z_][a-z0-9_]*)"?$', re.DOTALL)
_TABLE_REF_RE = re.compile(r"__(?:ref|source)_([a-z0-9_]+?)__")


@dataclass(frozen=True)
class SelectItem:
    """One item of the SELECT list."""

    output: str | None  # output column name (None when it cannot be known)
    source: str | None  # source column for plain and renamed columns
    kind: str  # "passthrough" | "renamed" | "cast" | "transformed" | "star"
    expression: str = ""


@dataclass
class SqlShape:
    kind: str = "unknown"  # "copy" (only *), "projection", or "unknown"
    items: list[SelectItem] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)  # certain row-altering effects
    uncertain: list[str] = field(default_factory=list)  # why the analysis cannot be trusted
    refs: list[str] = field(default_factory=list)  # distinct ref/source names found

    @property
    def altering(self) -> bool:
        return bool(self.effects)

    @property
    def readable(self) -> bool:
        """True when the select list could be read and nothing makes it ambiguous."""
        return self.kind != "unknown" and not self.uncertain and len(self.refs) == 1


def _mask(sql: str) -> str:
    """Drop comments and blank out string literals so keywords inside them never match."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if sql.startswith("--", i):
            while i < n and sql[i] != "\n":
                i += 1
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'" and sql[i + 1 : i + 2] == "'":
                    i += 2
                elif sql[i] == "'":
                    break
                else:
                    i += 1
            out.append("''")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _strip_jinja(raw: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    sql = _COMMENT_JINJA_RE.sub("", raw or "")
    sql = _CONFIG_RE.sub("", sql)
    sql = _INCREMENTAL_RE.sub(lambda m: m.group(2) or "", sql)
    sql = _REF_RE.sub(lambda m: f" __ref_{m.group(1)}__ ", sql)
    sql = _SOURCE_RE.sub(lambda m: f" __source_{m.group(1)}_{m.group(2)}__ ", sql)
    sql = _THIS_RE.sub(" __this__ ", sql)
    if _JINJA_EXPR_RE.search(sql):
        flags.append("jinja_expression")
        sql = _JINJA_EXPR_RE.sub(" __jinja__ ", sql)
    if _JINJA_TAG_RE.search(sql):
        flags.append("jinja_control")
        sql = _JINJA_TAG_RE.sub(" ", sql)
    return sql, flags


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth, start = 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _find_top_level_from(sql: str, start: int) -> int:
    depth = 0
    for m in re.finditer(r"[()]|\bfrom\b", sql[start:]):
        token = m.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            return start + m.start()
    return -1


def _parse_item(text: str) -> SelectItem:
    if re.fullmatch(r"(?:[a-z_][a-z0-9_]*\.)?\*", text):
        return SelectItem(output=None, source=None, kind="star")
    alias_match = _ALIAS_RE.match(text)
    expr, alias = (alias_match.group(1), alias_match.group(2)) if alias_match else (text, None)
    plain = _PLAIN_RE.match(expr.strip())
    if plain:
        source = plain.group(1)
        if alias and alias != source:
            return SelectItem(output=alias, source=source, kind="renamed", expression=expr)
        return SelectItem(output=alias or source, source=source, kind="passthrough")
    cast = _CAST_RE.match(expr.strip())
    if cast and alias and not _TEMPORAL_TARGET_RE.match(cast.group(2).strip()):
        # A pure type cast keeps the meaning of the value, so it is not a real transformation.
        # Casting to a date/time type changes the format, so that stays a transformation.
        return SelectItem(output=alias, source=cast.group(1), kind="cast", expression=expr)
    return SelectItem(
        output=alias, source=None, kind="transformed", expression=" ".join(expr.split())
    )


def analyze_sql(raw_code: str) -> SqlShape:
    """Read a model's raw dbt code. Never raises; unreadable code yields kind='unknown'."""
    shape = SqlShape()
    cleaned, flags = _strip_jinja(raw_code)
    shape.uncertain.extend(flags)
    sql = _mask(cleaned).lower()
    if not sql.strip():
        shape.uncertain.append("empty")
        return shape

    shape.effects = [name for name, pattern in _EFFECT_PATTERNS.items() if pattern.search(sql)]
    if _JOIN_RE.search(sql):
        shape.uncertain.append("join")
    if re.match(r"\s*with\b", sql):
        shape.uncertain.append("cte")
    elif len(re.findall(r"\bselect\b", sql)) > 1:
        shape.uncertain.append("subquery")
    shape.refs = sorted(set(_TABLE_REF_RE.findall(sql)))

    if "cte" in shape.uncertain:
        return shape
    head = re.match(r"\s*select\s+(distinct\s+)?", sql)
    if not head:
        shape.uncertain.append("unreadable_select")
        return shape
    from_pos = _find_top_level_from(sql, head.end())
    if from_pos == -1:
        shape.uncertain.append("unreadable_select")
        return shape

    shape.items = [_parse_item(p) for p in _split_top_level(sql[head.end() : from_pos])]
    only_star = len(shape.items) == 1 and shape.items[0].kind == "star"
    shape.kind = "copy" if only_star else "projection"
    return shape
