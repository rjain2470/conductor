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

_SELECT_STAR_NOTE = (
    "\n\nIMPORTANT: Your previous attempt used SELECT *. "
    "You MUST explicitly name every column in the SELECT list. "
    "Do not use SELECT *. "
    "Choose the 6-10 most relevant columns for the query."
)


def generate_sql(
    query: str,
    tables: list[str],
    lookup_info: dict | None = None,
    _retry: bool = False,
) -> dict:
    """
    Return {"sql": "...", "explanation": "..."}.
    sql is None if the query cannot be answered with a single SELECT.

    lookup_info: if provided by pipeline/lookup.py, injects a mandatory WHERE
    filter for the specific object that was identified. Array values require
    a ::numeric[] cast which is included automatically.
    """
    schema_slice = {t: _FULL_SCHEMA[t] for t in tables if t in _FULL_SCHEMA}
    prompt = _PROMPT_TEMPLATE

    if _retry:
        prompt = prompt + _SELECT_STAR_NOTE

    # Inject lookup constraint if a specific object was identified
    if lookup_info:
        value = lookup_info["value"]
        column = lookup_info["column"]
        table = lookup_info["table"]

        # Array columns are numeric[] in PostgreSQL — integer[] will fail
        if isinstance(value, list):
            value_repr = f"ARRAY{value}::numeric[]"
        else:
            value_repr = f"'{value}'"

        lookup_note = (
            f"\n\nCRITICAL: This query references a specific mathematical object. "
            f"You MUST include this exact filter in the WHERE clause: "
            f"{column} = {value_repr} "
            f"(this column is in table {table}). "
            f"The ::numeric[] cast is required for array comparison — "
            f"omitting it will cause a PostgreSQL type error. "
            f"Do not omit this filter under any circumstances."
        )
        prompt = prompt + lookup_note

    system = prompt.replace("<<SCHEMA>>", json.dumps(schema_slice, indent=2))
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    result = _parse(r.content[0].text)
    sql = result.get("sql")

    if sql:
        # Retry once if SELECT * was generated
        if _has_select_star(sql) and not _retry:
            return generate_sql(
                query, tables, lookup_info=lookup_info, _retry=True
            )
        _validate(sql)

    return result

def _has_select_star(sql: str) -> bool:
    """Return True if the SQL contains a bare SELECT * or SELECT table.*"""
    return bool(re.search(r'SELECT\s+(?:\w+\.)?\*', sql, re.IGNORECASE))

def _validate(sql: str) -> None:
    upper = sql.upper()
    for kw in _FORBIDDEN:
        if kw in upper:
            raise ValueError(f"Forbidden keyword in generated SQL: {kw}")
    if "LIMIT" not in upper:
        raise ValueError("Generated SQL is missing a LIMIT clause.")
    if _has_select_star(sql):
        raise ValueError(
            "Generated SQL uses SELECT * after retry. "
            "Please name the specific columns you need."
        )

def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
