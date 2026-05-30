"""
Description: This file takes a NL query and list of table names, and returns a validated SQL SELECT statement from the database. This is step 2 in our pipeline.
It loads the full schema once at import time and slices it to only the relevant tables.
"""

import json
import re
from pathlib import Path
from anthropic import Anthropic

_schema_path = Path(__file__).parent.parent / "schema" / "lmfdb_schema.json"
_prompt_path = Path(__file__).parent.parent / "prompts" / "sql_prompt.txt"

def _load_full_schema() -> dict:
    with open(_schema_path) as f:
        return json.load(f)["tables"]

def _load_prompt() -> str:
    with open(_prompt_path) as f:
        return f.read()

_FULL_SCHEMA = _load_full_schema()
_PROMPT_TEMPLATE = _load_prompt()

_FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"}


def generate_sql(query: str, tables: list[str]) -> dict:
    """
    Return {"sql": "...", "explanation": "..."}.
    sql is None if the query cannot be answered with a single SELECT.
    """
    schema_slice = {t: _FULL_SCHEMA[t] for t in tables if t in _FULL_SCHEMA}
    system = _PROMPT_TEMPLATE.replace("<<SCHEMA>>", json.dumps(schema_slice, indent=2))
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    result = _parse(r.content[0].text)
    if result.get("sql"):
        _validate(result["sql"])
    return result


def _validate(sql: str) -> None:
    upper = sql.upper()
    for kw in _FORBIDDEN:
        if kw in upper:
            raise ValueError(f"Forbidden keyword in generated SQL: {kw}")
    if "LIMIT" not in upper:
        raise ValueError("Generated SQL is missing a LIMIT clause.")


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
