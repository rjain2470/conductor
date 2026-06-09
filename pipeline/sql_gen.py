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

def _retry_note(query_type: str | None) -> str:
    """Type-aware instruction appended when the first attempt used SELECT *."""
    base = (
        "\n\nIMPORTANT: Your previous attempt used SELECT *. "
        "You MUST NOT use SELECT *. "
    )
    if query_type == "count":
        return base + "This is a cardinality query — use SELECT COUNT(*)."
    if query_type in ("scalar", "boolean", "aggregate"):
        return base + (
            "Select only the specific column(s) or aggregate the question asks "
            "for, not all columns."
        )
    return base + "Explicitly name the 6-10 most relevant columns for the query."


def generate_sql(
    query: str,
    tables: list[str],
    lookup_info: dict | None = None,
    query_type: str | None = None,
    sql_hint: str = "",
    _retry: bool = False,
) -> dict:
    """
    Return {"sql": "...", "explanation": "..."}.
    sql is None if the query cannot be answered with a single SELECT.

    lookup_info: optional dict from pipeline.lookup.resolve with keys
        table, column, value — injects a CRITICAL WHERE constraint into the
        prompt so the model cannot omit it.
    query_type / sql_hint: optional planner outputs that shape the SQL —
        query_type="count" forces SELECT COUNT(*); sql_hint is a free-form
        shaping instruction (aggregate / column / GROUP BY guidance).

    If the generated SQL uses SELECT *, retries once with an explicit instruction
    to name columns.
    """
    schema_slice = {t: _FULL_SCHEMA[t] for t in tables if t in _FULL_SCHEMA}
    prompt = _PROMPT_TEMPLATE

    if _retry:
        prompt = prompt + _retry_note(query_type)

    if query_type == "count":
        prompt = prompt + (
            "\n\nCRITICAL: This is a cardinality query. Return a single count: "
            "SELECT COUNT(*) FROM ... with the appropriate WHERE clause. "
            "Do not select individual columns and do not use SELECT *."
        )

    if query_type == "aggregate":
        prompt = prompt + (
            "\n\nCRITICAL: This is an aggregate query for a single statistic over the "
            "WHERE-filtered rows. Select only the aggregate value (no individual rows, "
            "no SELECT *):\n"
            "- average -> AVG(col); total -> SUM(col); extremes -> MIN(col)/MAX(col)\n"
            "- standard deviation -> STDDEV(col); variance -> VAR_SAMP(col)\n"
            "- median or percentile -> PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY col)\n"
            "Aggregate only over plain numeric columns. A single-row aggregate needs no LIMIT."
        )

    if query_type in ("scalar", "boolean"):
        prompt = prompt + (
            "\n\nCRITICAL: This question is about ONE specific object and wants a single fact. "
            "Select only the column(s) that answer the question, plus the identifying label "
            "column, and filter to that one object. Do not select unrelated columns and do not "
            "use SELECT *."
        )

    if sql_hint:
        prompt = prompt + f"\n\nSQL shaping hint from the planner: {sql_hint}"

    if lookup_info:
        table = lookup_info["table"]
        column = lookup_info["column"]
        value = lookup_info["value"]
        value_repr = f"ARRAY{value}::numeric[]" if isinstance(value, list) else f"'{value}'"
        prompt = prompt + (
            f"\n\nCRITICAL: This query references a specific mathematical object "
            f"that has already been resolved to a database identifier. "
            f"You MUST include this exact filter in the WHERE clause: "
            f"{column} = {value_repr} "
            f"Do not omit this filter under any circumstances. "
            f"Do not paraphrase or rewrite it."
        )

    system = prompt.replace("<<SCHEMA>>", json.dumps(schema_slice, indent=2))
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    result = _parse(r.content[0].text)
    sql = result.get("sql")

    if sql:
        # If SELECT * and this isn't already a retry, retry once
        if _has_select_star(sql) and not _retry:
            return generate_sql(
                query, tables, lookup_info=lookup_info,
                query_type=query_type, sql_hint=sql_hint, _retry=True,
            )
        sql = _ensure_limit(sql)
        result["sql"] = sql
        _validate(sql)

    return result


def _has_select_star(sql: str) -> bool:
    """Return True if the SQL contains a bare SELECT * or SELECT table.*"""
    # Match SELECT * or SELECT alias.* but not COUNT(*)
    return bool(re.search(r'SELECT\s+(?:\w+\.)?\*', sql, re.IGNORECASE))


_AGG_RE = re.compile(
    r"\b(?:COUNT|SUM|AVG|MIN|MAX|STDDEV|STDDEV_SAMP|STDDEV_POP|"
    r"VAR_SAMP|VAR_POP|VARIANCE|PERCENTILE_CONT|PERCENTILE_DISC)\s*\(",
    re.IGNORECASE,
)


def _is_scalar_aggregate(sql: str) -> bool:
    """
    True for a single-row aggregate query (e.g. SELECT COUNT(*) / AVG(x) ...
    with no GROUP BY). Such queries return one row, so a LIMIT is unnecessary.
    """
    if "GROUP BY" in sql.upper():
        return False
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    select_clause = m.group(1) if m else sql
    return bool(_AGG_RE.search(select_clause))


def _ensure_limit(sql: str, default: int = 10000) -> str:
    """
    Append a default LIMIT when one is absent, so result size stays bounded WITHOUT
    rejecting otherwise-valid SQL.

    Many legitimate queries naturally omit LIMIT — single-object lookups by primary
    key (scalar/boolean), and full-data analytical queries (distributions) where a
    LIMIT would distort the result. Enforcing the cap here, rather than failing
    validation, fixes that whole class uniformly. Scalar aggregates (COUNT/AVG/...
    with no GROUP BY) are single-row and left untouched.
    """
    if _is_scalar_aggregate(sql) or "LIMIT" in sql.upper():
        return sql
    return f"{sql.rstrip().rstrip(';').rstrip()} LIMIT {default}"


def _validate(sql: str) -> None:
    upper = sql.upper()
    for kw in _FORBIDDEN:
        if kw in upper:
            raise ValueError(f"Forbidden keyword in generated SQL: {kw}")
    if "LIMIT" not in upper and not _is_scalar_aggregate(sql):
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
