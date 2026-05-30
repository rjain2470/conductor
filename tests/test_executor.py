"""
Description: Tests for pipeline/executor.py.
All tests requiring a live DB connection are gated on LMFDB_HOST being set.
"""

import os
import pytest


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("LMFDB_HOST"),
    reason="LMFDB_HOST not set — no DB connection available"
)
def test_execute_simple_query():
    from pipeline.executor import execute_sql
    df = execute_sql("SELECT label FROM mf_newforms WHERE level = 11 LIMIT 5")
    assert len(df) > 0
    assert "label" in df.columns

@pytest.mark.skipif(
    not os.environ.get("LMFDB_HOST"),
    reason="LMFDB_HOST not set — no DB connection available"
)
def test_execute_returns_dataframe():
    import pandas as pd
    from pipeline.executor import execute_sql
    df = execute_sql("SELECT lmfdb_label, rank FROM ec_curvedata WHERE conductor = 11 LIMIT 5")
    assert isinstance(df, pd.DataFrame)
    assert "lmfdb_label" in df.columns
