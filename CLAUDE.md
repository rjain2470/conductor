# Conductor — backend notes for Claude

Conductor (conductormath.org) is a FastAPI backend answering natural-language questions over the
LMFDB. Pipeline (`pipeline/chat.py`, `LMFDBChat.chat`): intent → **planner** (`_plan` / `_PLAN_SYSTEM`:
clarity decision, history-resolved standalone `refined_query`, and query-type classification) →
resolve (`lookup.py`) → route (`router.py`, uses `schema/schema_index.json`) → SQL generation
(`sql_gen.py` + `prompts/sql_prompt.txt`) → execute (`executor.py`) → response shaping by
`plan["output_format"]` (prose synthesis via `prompts/synthesis_prompt.txt` / table / plot via
`analysis.py`).

Query types set by the planner: `scalar`, `boolean`, `count`, `aggregate`, `tabular`, `analytical`,
`prose`. `output_format`: `prose` | `table` | `plot`.

## Working principles

- Prefer root-cause fixes over localized patches. When you hit a bug, fix the whole class of problems
  it represents (e.g. add a query-type classification stage rather than special-casing one phrasing;
  enrich the schema index's column descriptions rather than hardcoding a router rule for one term).
  Localized special cases miss equivalent phrasings and accumulate as debt. Always look for the deeper
  root and fix the entire class of problems that could arise similarly.

## Pending tasks (other components / sessions)

- **Frontend → `/analysis` must send `last_sql`.** The backend's stateless rehydration
  (`LMFDBChat._ensure_dataframe`, which re-runs `last_sql` when `last_df` is missing) only fires if the
  frontend includes the prior SQL in the `/analysis` request body. Update the frontend to pass
  `lastSqlRef.current` as `last_sql` in the `/analysis` POST body. Without it, analysis follow-ups that
  land on a fresh worker still fail.
- **Backend: carry `last_query_type` in the stateless payload.** Add `last_query_type` to `ChatRequest`
  (main.py) and restore it in `/chat` alongside `history`/`last_sql`/`last_tables`, so follow-up turns
  can see the previous query type. (The planner already gets prior-turn context via `history`; this is
  an enhancement, not a correctness blocker.)

## Known gaps / edge cases (query-type classification & SQL shaping)

Non-blocking edge cases in the query-type pipeline, deferred for a dedicated pass:

1. **Mode / IQR / skewness misclassification.** These read like single-stat aggregates but have no
   portable scalar SQL form — they need Python. The planner may classify them as `aggregate` (→ SQL)
   instead of `analytical` (→ Python analysis). Teach the planner (and possibly `sql_gen`) to route
   these specific statistics to analysis.

2. **Grouped aggregates are ambiguous at the planner boundary.** "Average conductor for each rank" is
   multi-row and should be `tabular` (a `GROUP BY` result), but users phrase it exactly like an
   `aggregate` question. The aggregate (single row) vs grouped/tabular boundary is not crisp;
   misclassification returns a single number instead of the per-group table.

3. **`sql_hint` is unvalidated free text.** The planner's `sql_hint` is appended to the SQL prompt, but
   nothing verifies it actually influenced the generated SQL or is consistent with the chosen
   `query_type`. It can be silently ignored or contradict the query.

4. **`PERCENTILE_CONT` / numeric aggregates assume plain numeric columns.** Several LMFDB columns are
   text or jsonb (label-like fields, array/jsonb invariants). Aggregating those errors or, worse, fails
   silently / misleads. `sql_gen` restricts aggregates to "plain numeric columns" by instruction only —
   there is no type check against `schema/lmfdb_schema.json`.
