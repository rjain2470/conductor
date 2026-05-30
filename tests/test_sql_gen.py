"""
Description: Unit tests for pipeline/sql_gen.py, in particular for the  _validate safety function.
"""

import pytest
from pipeline.sql_gen import _parse, _validate


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_parse_clean_json():
    result = _parse('{"sql": "SELECT 1", "explanation": "test"}')
    assert result["sql"] == "SELECT 1"

def test_parse_strips_markdown():
    result = _parse('```json\n{"sql": "SELECT 1", "explanation": "test"}\n```')
    assert result["sql"] == "SELECT 1"

def test_validate_passes_valid_sql():
    _validate("SELECT label FROM mf_newforms WHERE level < 100 LIMIT 10")

def test_validate_rejects_insert():
    with pytest.raises(ValueError, match="INSERT"):
        _validate("INSERT INTO mf_newforms VALUES (1)")

def test_validate_rejects_drop():
    with pytest.raises(ValueError, match="DROP"):
        _validate("DROP TABLE mf_newforms")

def test_validate_rejects_missing_limit():
    with pytest.raises(ValueError, match="LIMIT"):
        _validate("SELECT label FROM mf_newforms WHERE level < 100")

def test_validate_case_insensitive():
    with pytest.raises(ValueError):
        _validate("delete from mf_newforms limit 10")


# ── Integration tests ─────────────────────────────────────────────────────────

import os

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_generate_sql_newforms():
    from pipeline.sql_gen import generate_sql
    result = generate_sql(
        "Weight-2 newforms with trivial character and analytic rank 1, level under 200",
        ["mf_newforms"]
    )
    assert result.get("sql") is not None
    assert "mf_newforms" in result["sql"]
    assert "LIMIT" in result["sql"].upper()
