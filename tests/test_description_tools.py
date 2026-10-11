"""Tests for the MCP tools evaluate_model_descriptions / submit_description_scores."""

import json

import pytest

from dbt_doctor import server
from dbt_doctor.core.manifest import DbtManifestParser


@pytest.fixture
def tools(project_dir_with_manifest, monkeypatch):
    monkeypatch.setattr(server, "_manifest", DbtManifestParser(str(project_dir_with_manifest)))
    monkeypatch.setattr(server, "_rubric_path", None)
    return project_dir_with_manifest


def all_columns(batch):
    return [c["name"] for c in batch["columns"]]


def test_full_flow_on_fixture_model(tools):
    batch = json.loads(server.evaluate_model_descriptions("stg_orders"))
    assert batch["model"] == "stg_orders"
    assert batch["next_offset"] is None
    assert "order_id" in all_columns(batch)
    assert batch["sql"]["kind"] == "compiled"
    assert "order_id" in batch["unique_columns"]
    by_name = {c["name"]: c for c in batch["columns"]}
    assert by_name["status"]["signals"]["empty_description"] is True

    ids = [q["id"] for q in batch["rubric"]["questions"]]
    scores = [
        {
            "column": name,
            "scores": dict.fromkeys(ids, 2 if by_name[name]["description"] == "" else 9),
            "suggested_description": f"Rewritten {name}",
        }
        for name in all_columns(batch)
    ]
    report = json.loads(server.submit_description_scores("stg_orders", scores))
    assert report["status"] == "complete"
    rewrite = {c["column"] for c in report["columns"] if c["status"] == "rewrite"}
    assert rewrite == {n for n, c in by_name.items() if c["description"] == ""}
    # The state file is cleaned up at the end.
    assert not list((tools / "target" / "dbt_doctor").glob("description_eval_*.json"))


def test_unknown_model(tools):
    assert "not found" in server.evaluate_model_descriptions("ghost")
    assert "not found" in server.submit_description_scores("ghost", [])


def test_invalid_scores_return_readable_error(tools):
    out = server.submit_description_scores("stg_orders", [{"column": "ghost", "scores": {}}])
    assert out.startswith("❌") and "unknown column 'ghost'" in out


def test_invalid_offset_returns_error(tools):
    assert server.evaluate_model_descriptions("stg_orders", offset=99).startswith("❌")


def test_custom_rubric_file_is_used_and_reloaded(tools, tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "pass_ratio": 0.5,
                "max_score_per_question": 10,
                "questions": [{"id": "only_one", "text": "Is it good?"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_rubric_path", rubric)
    batch = json.loads(server.evaluate_model_descriptions("stg_orders"))
    assert [q["id"] for q in batch["rubric"]["questions"]] == ["only_one"]
    assert batch["rubric"]["pass_total"] == 5

    # Edits apply without restarting the server.
    rubric.write_text(rubric.read_text().replace("Is it good?", "Is it great?"), encoding="utf-8")
    batch = json.loads(server.evaluate_model_descriptions("stg_orders"))
    assert batch["rubric"]["questions"][0]["text"] == "Is it great?"


def test_broken_rubric_file_returns_error(tools, tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    monkeypatch.setattr(server, "_rubric_path", bad)
    assert server.evaluate_model_descriptions("stg_orders").startswith("❌")
