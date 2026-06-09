"""
Description: This file generates and executes Python analysis code on the current DataFrame. It is step 4 in our pipeline.
If the code produces a matplotlib figure, it is returned as a base64 PNG string.
"""

import io
import re
import json
import base64
import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib imports
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional
from anthropic import Anthropic

_prompt_path = Path(__file__).parent.parent / "prompts" / "analysis_prompt.txt"
_style_path  = Path(__file__).parent.parent / "prompts" / "analysis_style.txt"


def _load_prompt() -> str:
    with open(_prompt_path) as f:
        base = f.read()
    try:
        with open(_style_path) as f:
            style = f.read()
        return base + "\n" + style
    except FileNotFoundError:
        return base


_PROMPT_TEMPLATE = _load_prompt()
_TIMEOUT_SECONDS = 30


def generate_analysis(instruction: str, df) -> dict:
    """Return {"code": "...", "explanation": "..."}."""
    system = (
        _PROMPT_TEMPLATE
        .replace("<<DF_DESCRIPTION>>", str(df.dtypes))
        .replace("<<DF_HEAD>>", df.head(5).to_json(orient="records", indent=2))
    )
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": instruction}]
    )
    return _parse(r.content[0].text)


def execute_analysis(code: str, df) -> dict:
    """
    Execute generated analysis code in a restricted namespace.

    Returns {"plot": <base64 PNG or None>, "result": <captured text or None>}:
      - plot: a base64-encoded PNG if the code produced a matplotlib figure, else None.
      - result: the code's printed stdout (preferred) or str(result) variable, so that
        non-plot analyses (summary statistics, correlations, grouped aggregates, ...)
        can be synthesised into a prose answer rather than discarded.

    Raises TimeoutError if execution exceeds _TIMEOUT_SECONDS.

    Uses ThreadPoolExecutor instead of signal.alarm so it works on any
    thread (signal.alarm is main-thread only and fails on Render/uvicorn).
    matplotlib.use("Agg") is set at module level to avoid thread-safety issues.
    plt.close("all") clears stale figure state before each execution.
    """
    import numpy as np
    import scipy
    import scipy.stats  # ensure the stats submodule is loaded so scipy.stats works in generated code

    namespace = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sorted": sorted,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        },
        "df": df,
        "pd": __import__("pandas"),
        "np": np,
        "plt": plt,
        "scipy": scipy,
        "result": None,
    }

    def _run():
        plt.close("all")
        out = io.StringIO()
        with redirect_stdout(out):
            exec(code, namespace)  # noqa: S102
        fig = plt.gcf()
        plot_b64 = None
        if fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        printed = out.getvalue().strip()
        result_val = namespace.get("result")
        if printed:
            result_text = printed
        elif result_val is not None:
            result_text = str(result_val)
        else:
            result_text = None
        return {"plot": plot_b64, "result": result_text}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            raise TimeoutError(f"Analysis exceeded {_TIMEOUT_SECONDS}s time limit.")


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
