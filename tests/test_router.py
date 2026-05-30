"""
Description: Unit tests for pipeline/router.py. Integration tests are skipped unless ANTHROPIC_API_KEY is set in the environment.
"""

import os
import pytest
from pipeline.router import _parse


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_parse_clean_json():
    assert _parse('{"tables": ["mf_newforms"]}') == {"tables": ["mf_newforms"]}

def test_parse_strips_markdown():
    assert _parse('```json\n{"tables": ["mf_newforms"]}\n```') == {"tables": ["mf_newforms"]}

def test_parse_empty_raises():
    with pytest.raises(Exception):
        _parse("")

def test_parse_multiple_tables():
    result = _parse('{"tables": ["mf_newforms", "lfunc_instances"]}')
    assert result == {"tables": ["mf_newforms", "lfunc_instances"]}


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_route_newforms_query():
    from pipeline.router import route
    tables = route("Give me weight-2 newforms with trivial character and analytic rank 1")
    assert "mf_newforms" in tables

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_route_elliptic_curves():
    from pipeline.router import route
    tables = route("Elliptic curves with rank 2 and conductor under 1000")
    assert "ec_curvedata" in tables
