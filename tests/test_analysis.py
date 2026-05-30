"""
Description: Tests for pipeline/analysis.py.
Note: We test execute_analysis without API calls by passing in literal code strings.
"""

import pytest
import pandas as pd
from pipeline.analysis import _parse, execute_analysis


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_parse_clean_json():
    result = _parse('{"code": "result = 1", "explanation": "test"}')
    assert result["code"] == "result = 1"

def test_execute_analysis_returns_none_for_no_plot():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = execute_analysis("result = df['x'].mean()", df)
    assert result is None

def test_execute_analysis_returns_base64_for_plot():
    import base64
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    code = "plt.plot(df['x'], df['y'])\nplt.xlabel('x')\nplt.ylabel('y')\nplt.title('test')"
    result = execute_analysis(code, df)
    assert result is not None
    # Verify it is valid base64
    base64.b64decode(result)

def test_execute_analysis_blocks_os_import():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(Exception):
        execute_analysis("import os; os.listdir('/')", df)

def test_execute_analysis_timeout():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(Exception):
        execute_analysis("while True: pass", df)


# ── Integration tests ─────────────────────────────────────────────────────────

import os

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_generate_analysis_histogram():
    from pipeline.analysis import generate_analysis
    df = pd.DataFrame({"level": [11, 23, 37, 11, 23], "dim": [1, 1, 2, 1, 1]})
    result = generate_analysis("Plot a histogram of the level column", df)
    assert result.get("code") is not None
