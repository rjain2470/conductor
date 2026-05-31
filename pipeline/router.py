"""
Description: This file takes in the user's natural language query and returns a list of LMFDB table names that are relevant to answering it. It is the first step in our pipeline.

It uses a two-layer hierarchical schema index (domains -> tables).  The model reasons about the domains, then the tables, within a single call. Runs on claude-haiku-4-5 for speed.
"""

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

Your task: given a user query, identify which specific tables are needed to answer it.

Reason in two steps:
1. Which domain or domains are relevant to this query?
2. Within those domains, which specific tables are needed?

Then return ONLY a raw JSON object: {"tables": ["table_name", ...]}
No markdown, no prose, no backticks — only the JSON object.

Rules:
- Include all tables needed for joins between objects and their L-functions, Galois representations, etc.
- Use the column hints to confirm a table is relevant before including it.
- Note any CamelCase warnings (artin_reps, artin_field_data) and naming quirks (cond vs conductor).
- Prefer precision: do not include tables speculatively.
- Never include belyi_galmaps_prim — it does not exist in the database.

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
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    return _parse(r.content[0].text)["tables"]


def _parse(text: str) -> dict:
    """Extract JSON object from response, tolerating reasoning text around it."""
    text = text.strip()
    if not text:
        raise ValueError(
            "Router returned empty response — model may have exceeded max_tokens."
        )
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: find the JSON object anywhere in the text
    match = re.search(r'\{[^{}]*"tables"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(
        f"Could not extract JSON from router response.\nRaw text:\n{text[:500]}"
    )Could not extract JSON from router response. Raw text:\n{text[:500]}")
