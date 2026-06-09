'''
Description: This file adds functionality to resolve mathematical objects in a query and resolve them
to database identifiers before the rest of the pipeline sees the query. If no concrete object is detected, 
it returns the query unchanged.
'''

import json
import re
from anthropic import Anthropic

_LOOKUP_SYSTEM = """You are a mathematical object resolver for the LMFDB database.

Your task: determine whether a query contains a concrete mathematical object
given by its explicit definition (equation, polynomial, or label), and if so,
extract the information needed to look it up in the database.

Return ONLY a raw JSON object with one of two structures:

If NO concrete object is present:
{"type": "none"}

If a concrete object IS present:
{
  "type": "elliptic_curve_equation" | "number_field_polynomial" | "lmfdb_label" | "genus2_equation",
  "object_class": "elliptic_curve" | "number_field" | "newform" | "dirichlet_character" | "artin_rep" | "genus2_curve" | "other",
  "lookup": {
    "table": "<table name>",
    "column": "<column name>",
    "value": <the value to match — array for ainvs/coeffs, string for labels>
  },
  "residual_query": "<the original query with the object reference replaced by 'this object'>"
}

Rules for elliptic curves given by Weierstrass equation y^2 + a1*x*y + a3*y = x^3 + a2*x^2 + a4*x + a6:
- Extract [a1, a2, a3, a4, a6] as integers
- table: "ec_curvedata", column: "ainvs"
- Example: y^2 + xy + y = x^3 - x^2 - 3x + 3 → ainvs = [1, -1, 1, -3, 3]
- Example: y^2 = x^3 - x → ainvs = [0, 0, 0, -1, 0]
- Example: y^2 + y = x^3 - x^2 → ainvs = [0, -1, 1, 0, 0]

Rules for number fields given by polynomial:
- Extract coefficients lowest degree first as integers
- table: "nf_fields", column: "coeffs"
- Example: x^3 - x - 1 → coeffs = [-1, -1, 0, 1]
- Example: x^4 - x - 1 → coeffs = [-1, -1, 0, 0, 1]
- Example: x^2 - 5 → coeffs = [-5, 0, 1]

Rules for LMFDB labels:
- Newform label (N.k.c.x format e.g. 11.2.a.a): table "mf_newforms", column "label"
- Dirichlet character label (q.n format e.g. 7.3): table "char_dirichlet", column "label"
- Elliptic curve LMFDB label (N.an format e.g. 11.a1): table "ec_curvedata", column "lmfdb_label"
- Elliptic curve Cremona label (e.g. 11a1): table "ec_curvedata", column "Clabel"
- Number field label (d.r.disc.n format e.g. 2.0.3.1): table "nf_fields", column "label"
- Artin rep label: table "artin_reps", column "Baselabel"
- Genus-2 curve label: table "g2c_curves", column "label"

No markdown, no backticks, no prose — only the JSON object."""


# Deterministic elliptic-curve label disambiguation. The LLM resolver is
# unreliable at choosing between the two EC label formats (and tends to strip the
# dot), so re-derive the label straight from the query text: an LMFDB label has a
# dot (e.g. 11.a1) and queries lmfdb_label; a Cremona label has no dot (e.g. 11a1)
# and queries Clabel. General over all conductors / isogeny classes / curve numbers.
_EC_LMFDB_LABEL = re.compile(r"\b\d+\.[a-z]+\d+\b")
_EC_CREMONA_LABEL = re.compile(r"\b\d+[a-z]+\d+\b")


def _correct_ec_label(query: str, lookup: dict) -> dict:
    """
    Fix the column/value for elliptic-curve label lookups deterministically.

    Acts only when the resolver already chose an ec_curvedata label column, then
    re-derives the label from the query so the dot (LMFDB) vs no-dot (Cremona)
    distinction cannot be lost: LMFDB -> lmfdb_label, Cremona -> Clabel. Leaves the
    lookup unchanged if no EC label is present in the query.
    """
    if lookup.get("table") != "ec_curvedata" or lookup.get("column") not in ("lmfdb_label", "Clabel"):
        return lookup
    m = _EC_LMFDB_LABEL.search(query)
    if m:
        return {**lookup, "column": "lmfdb_label", "value": m.group()}
    m = _EC_CREMONA_LABEL.search(query)
    if m:
        return {**lookup, "column": "Clabel", "value": m.group()}
    return lookup


def resolve(query: str) -> tuple[str, dict | None]:
    """
    Attempt to resolve a concrete mathematical object in the query.

    Returns:
        (resolved_query, lookup_info) where:
        - resolved_query is the query rewritten to include the database lookup
        - lookup_info is the lookup dict if an object was found, else None

    Fails open: returns (original_query, None) on any exception.
    """
    try:
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system=_LOOKUP_SYSTEM,
            messages=[{"role": "user", "content": query}]
        )
        result = _parse(r.content[0].text)

        if result.get("type") == "none":
            return query, None

        lookup = result.get("lookup")
        residual = result.get("residual_query", query)

        if not lookup:
            return query, None

        lookup = _correct_ec_label(query, lookup)

        table = lookup["table"]
        column = lookup["column"]
        value = lookup["value"]

        if isinstance(value, list):
            value_repr = f"ARRAY{value}::numeric[]"
        else:
            value_repr = f"'{value}'"

        injected = (
            f"{query} "
            f"[Resolved: look up in {table} WHERE {column} = {value_repr}]"
        )

        return injected, lookup

    except Exception:
        # Fail open — never block a query due to lookup errors
        return query, None


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
