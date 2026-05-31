"""
Description: This file takes in the user's natural language query and returns a list of LMFDB table names that are relevant to answering it. It is the first step in our pipeline.

It uses a two-layer hierarchical schema index (domains -> tables).  The model reasons about the domains, then the tables, within a single call.
"""

import json
import re
from pathlib import Path
from anthropic import Anthropic

_schema_path = Path(__file__).parent.parent / "schema" / "schema_index.json"


def _load_schema_index() -> str:
    with open(_schema_path) as f:
        return json.dumps(json.load(f), indent=2)


_SCHEMA_INDEX = _load_schema_index()

_SYSTEM = """You are the routing layer of a natural language interface to the LMFDB PostgreSQL database.

The schema is organized into domains. Each domain contains tables with their key queryable columns.

Your task: given a user query, identify which tables are needed to answer it.

Reason in two steps within your response:
1. Which domain or domains are relevant?
2. Within those domains, which specific tables are needed?

Then return ONLY a raw JSON object: {"tables": ["table_name", ...]}
No markdown, no prose, no backticks — only the JSON object.

Rules:
- Include all tables needed for joins.
- Use the key column listings to confirm a table is relevant before including it.
- If a query mentions a concept that maps to a specific column noted in the index
  (e.g. 'conductor' for Artin reps is the column 'Conductor' with capital C),
  include that table and note it in your reasoning.
- Prefer precision over recall — do not include tables speculatively.
- Never include belyi_galmaps_prim — it does not exist in the database.

Schema index:
""" + _SCHEMA_INDEX


def route(query: str, history: str = "") -> list[str]:
    """Return relevant table names for the given query."""
    system = _SYSTEM
    if history:
        system += f"\n\nConversation so far:\n{history}"
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    return _parse(r.content[0].text)["tables"]


def _parse(text: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
