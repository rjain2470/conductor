"""
Description: This file takes in the user's natural language query and returns a list of LMFDB table names that are relevant to answering it. It is the first step in our pipeline.

It uses a two-layer hierarchical schema index (domains -> tables).  The model reasons about the domains, then the tables, within a single call. Runs on claude-haiku-4-5 for speed.
"""

# pipeline/router.py
# Stage 1: given a NL query, return the list of relevant LMFDB table names.
#
# Uses a two-layer hierarchical routing index (domains -> tables).
# Runs on claude-haiku-4-5 for speed — routing is classification, not generation.
# max_tokens=2048 gives room for brief reasoning before the JSON output.
#
# The _parse function tolerates reasoning text before the JSON object,
# and raises a clear ValueError on empty or malformed responses.

import json
import re
from pathlib import Path
from anthropic import Anthropic

_index_path = Path(__file__).parent.parent / "schema" / "routing_index.json"


def _load_routing_index() -> str:
    with open(_index_path) as f:
        return json.dumps(json.load(f), indent=2)


_ROUTING_INDEX = _load_routing_index()

_SYSTEM = """You are the routing layer of a natural language interface to the LMFDB PostgreSQL database.

The schema is organized into domains. Each domain contains tables with brief descriptions of their key columns.

Your task: identify which specific tables are needed to answer the user's query.

Reasoning instructions:
- Think briefly (2-3 sentences maximum) about which domain applies and which tables are needed.
- Then immediately output the JSON object.
- Do not write long explanations. Be concise.

Output format: ONLY a raw JSON object on the final line: {"tables": ["table_name", ...]}
No markdown, no backticks — only the JSON object as the last thing you write.

Rules:
- Include all tables needed for joins.
- Use column hints to confirm a table is relevant before including it.
- Prefer precision: do not include tables speculatively.
- Never include belyi_galmaps_prim — it does not exist in the database.
Special cases:
- If the query gives a Weierstrass equation y² + a1*x*y + a3*y = x³ + a2*x² + a4*x + a6, route to ec_curvedata. The curve can be found by filtering on ainvs = ARRAY[a1,a2,a3,a4,a6]::numeric[].
- If the query asks for integral points on a specific elliptic curve, route to ec_curvedata (num_int_pts column stores the count).

Schema index:
""" + _ROUTING_INDEX


def route(query: str, history: str = "") -> list[str]:
    """Return relevant table names for the given query."""
    system = _SYSTEM
    if history:
        system += f"\n\nConversation so far:\n{history}"
    client = Anthropic()
    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    return _parse(r.content[0].text)["tables"]


def _parse(text: str) -> dict:
    """
    Extract JSON object from response, tolerating brief reasoning text before it.
    Raises ValueError with a clear message on empty or malformed responses.
    """
    text = text.strip()
    if not text:
        raise ValueError(
            "Router returned empty response. "
            "The query may be too complex — try rephrasing it more concisely."
        )

    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back: find the JSON object anywhere in the text.
    match = re.search(r'\{\s*"tables"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Router could not identify relevant tables for this query."
        "Try rephrasing or being more specific about the mathematical objects involved."
    )
