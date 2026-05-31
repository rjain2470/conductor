"""
Description: This file generates and executes Python analysis code on the current DataFrame. It is step 4 in our pipeline.
If the code produces a matplotlib figure, it is returned as a base64 PNG string.
"""

import io
import re
import json
import signal
from pathlib import Path
from typing import Optional
from anthropic import Anthropic

_prompt_path = Path(__file__).parent.parent / "prompts" / "analysis_prompt.txt"
_style_path = Path(__file__).parent.parent / "prompts" / "analysis_style.txt"

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


def execute_analysis(code: str, df) -> Optional[str]:
    """
    Execute generated analysis code in a restricted namespace.
    Returns a base64-encoded PNG if a plot was produced, else None.
    Raises TimeoutError if execution exceeds _TIMEOUT_SECONDS.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy

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

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Analysis exceeded {_TIMEOUT_SECONDS}s time limit.")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_TIMEOUT_SECONDS)
    try:
        exec(code, namespace)  # noqa: S102
    finally:
        signal.alarm(0)

    import base64
    fig = plt.gcf()
    if fig.get_axes():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return None


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
