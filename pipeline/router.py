"""
Description: This file takes in the user's natural language query and returns a list of LMFDB table names that are relevant to answering it. 
It is the first step in our pipeline.
"""

import json
import re
from pathlib import Path
from anthropic import Anthropic

_schema_path = Path(__file__).parent.parent / "schema" / "compressed_schema.json"

def _load_compressed_schema() -> str:
    with open(_schema_path) as f:
        data = json.load(f)
    return "\n".join(f'"{k}": "{v}"' for k, v in data["tables"].items())

_COMPRESSED_SCHEMA = _load_compressed_schema()

_SYSTEM = """You are the routing layer of a natural language interface to the LMFDB PostgreSQL database.

Given a user query, return ONLY a raw JSON object: {"tables": ["table_name", ...]}
No markdown, no prose, no backticks.

Include all tables needed to answer the query, including those needed for joins.
Omit portrait, image, and inv_* tables unless explicitly requested.

Compressed schema:
""" + _COMPRESSED_SCHEMA


def route(query: str, history: str = "") -> list[str]:
    """Return relevant table names for the given query."""
    system = _SYSTEM
    if history:
        system += f"\n\nConversation so far:\n{history}"
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    return _parse(r.content[0].text)["tables"]


def _parse(text: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
