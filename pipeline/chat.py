"""
Description: This file holds all session state (history, last DataFrame, last SQL) and executes the chatbot functionality of this package. 

Error handling: 
- Every failure mode returns a structured, human-readable response.
- Errors are categorised: clarification needed, no data found, query generation failed, execution failed, analysis failed.
"""

import re
import json
from pathlib import Path
from anthropic import Anthropic
from pipeline.router import route
from pipeline.sql_gen import generate_sql
from pipeline.executor import execute_sql
from pipeline.lookup import resolve
from pipeline.analysis import generate_analysis, execute_analysis

_SYNTHESIS_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "synthesis_prompt.txt"


def _load_synthesis_prompt() -> str:
    with open(_SYNTHESIS_PROMPT_PATH) as f:
        return f.read()


_SYNTHESIS_PROMPT = _load_synthesis_prompt()

# ── System prompts ────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """You are a query classifier for Conductor, an LMFDB interface that CAN generate plots and analyses.

Classify the user message as either QUERY or CHAT.

Output QUERY if the message:
- Asks for data from the LMFDB database
- Asks for a plot, graph, chart, or visualisation
- Asks for analysis of data
- Refers to previous results and asks to filter, refine, or extend them
- Asks about a specific mathematical object (elliptic curve, modular form, L-function, etc.) by its properties or equation
- Asks to look up, find, count, or retrieve a mathematical invariant (rank, conductor, integral points, torsion, etc.)

Output CHAT if the message:
- Is a greeting, thanks, or small talk
- Asks something entirely unrelated to mathematics or the LMFDB
- Asks for a proof or theoretical explanation with no data lookup involved

When in doubt, output QUERY.

Respond with exactly one word: either QUERY or CHAT. Nothing else."""

_CHAT_SYSTEM = """You are Conductor, a mathematically knowledgeable assistant with access to the LMFDB (L-functions and Modular Forms Database). You have a warm, understated personality — think a good graduate student who knows their stuff and doesn't waste words.

You CAN generate plots and analyses of LMFDB data when asked.

If someone greets you, chats, or thanks you, respond naturally and briefly as yourself. You can show a little personality. Don't be formal.

If someone asks something unrelated to mathematics or the LMFDB — conferences, restaurants, life advice — gently let them know what you're here for, but without being robotic about it."""

_PLAN_SYSTEM = """You are the query planner for Conductor, a natural-language interface to the LMFDB.
Given the user's latest message and the conversation so far, decide whether the request is clear
enough to act on, restate it as a self-contained query, and classify how it should be answered.

Return ONLY a raw JSON object (no markdown, no backticks, no prose) with these keys:

{
  "action": "proceed" | "clarify",
  "question": "<one focused question, else \\"\\">",
  "refined_query": "<standalone restatement of what to query>",
  "query_type": "scalar" | "boolean" | "count" | "aggregate" | "tabular" | "analytical" | "prose",
  "output_format": "prose" | "table" | "plot",
  "needs_analysis": true | false,
  "analysis_instruction": "<what the Python analysis step should compute, else \\"\\">",
  "sql_hint": "<optional shaping hint for SQL generation, else \\"\\">"
}

REFINED QUERY (resolve context):
- Produce a fully self-contained refined_query. Resolve pronouns and ellipsis ("that", "those",
  "what about rank 3", "now restrict to prime conductor") against the conversation history.
- Carry forward every constraint from the previous turn that the new message does not change; only
  override the specific constraints the new message changes.
- Always name the mathematical object type explicitly (e.g. "elliptic curves over Q", "newforms").
- If a Weierstrass equation is given, identify it as an elliptic curve over Q.

QUERY TYPE (how the answer should be shaped):
- scalar: one value of one named object — "What is the rank of 11.a1?". output_format prose.
- boolean: a simple yes/no answerable from one stored property — "Is 11.a1 semistable?", "Is 389.a1 a
  CM curve?". output_format prose. (A yes/no needing several invariants and reasoning, like BSD
  verification, is prose, not boolean.)
- count: a cardinality — "How many elliptic curves have rank 2?". output_format prose.
- aggregate: a single statistic over a set, computable in one SQL query — average/min/max/sum/standard
  deviation/variance, or a single median or percentile of one numeric column. Also an extremal VALUE,
  "the largest/smallest X" -> MAX/MIN. output_format prose; needs_analysis false.
- tabular: a list of matching objects ("list elliptic curves with rank 2"); an extremal OBJECT, "the
  curve(s) with the largest/smallest X" -> ORDER BY X with a small LIMIT; or a grouped aggregate,
  "X for each Y" / "per Y" / "grouped by Y" / "count by Y" -> a GROUP BY result. output_format table;
  needs_analysis false. Put the ORDER BY / GROUP BY shape in sql_hint.
- analytical: needs computation beyond what one SQL query returns — a full distribution or several
  statistics at once, correlation between two invariants, histograms/binning, or cumulative/growth
  counts versus a bound. Set needs_analysis true and write a concrete, column-specific
  analysis_instruction. output_format plot for visual requests, otherwise prose.
- prose: a free-form question grounded in database invariants that needs a synthesized explanation,
  often combining several invariants — "Is BSD verified for 11.a1?", "What is the BSD prediction for
  37.a1?". output_format prose; set needs_analysis true if a computation is required.

GROUNDED FREE-FORM QUESTIONS (query_type "prose"):
- These ask a conceptual question whose answer must be derived from database invariants — e.g.
  "Is BSD verified for 11.a1?", "What is the BSD prediction for 37.a1?", "What does the rank of
  5077.a1 tell us?".
- Write refined_query so it explicitly requests EVERY invariant needed to answer, naming them so
  routing and SQL retrieve them (do not leave the needed data implicit). For BSD this is the curve's
  rank and analytic rank together with the Mordell-Weil / BSD data: regulator, real period, Tamagawa
  product, analytic order of Sha, special L-value, and torsion.
- If answering requires evaluating a formula or numerical comparison beyond plain retrieval, set
  needs_analysis true with a concrete analysis_instruction; if it only requires reading and comparing
  retrieved values, needs_analysis may be false and synthesis will explain.

NEEDS ANALYSIS:
- true when answering requires Python over the retrieved rows (statistics, correlation, binning,
  cumulative sums, grouped aggregation, plotting). Otherwise false.
- When true, analysis_instruction must state concretely what to compute (e.g. "compute mean, median
  and standard deviation of the rank column", or "plot a histogram of conductor").

SQL HINT (optional):
- count: "use SELECT COUNT(*)". aggregate: name the SQL aggregate and column. extremal object:
  "ORDER BY <col> DESC/ASC LIMIT k". grouped: "GROUP BY <col>, <aggregate>". Leave "" when no special
  shaping is needed.

Flag ambiguity (action "clarify") only when it would materially change what is queried.
Do not ask for clarification that is not mathematically necessary.

Conversation so far:
<<HISTORY>>"""

_PROVENANCE_SYSTEM = """You write the single closing line for a mathematical database query response.

The query returned data from these SQL tables: {tables}
The query returned {rows} rows.

Output EXACTLY one sentence that:
1. States the actual number of rows returned ({rows}) — commit to the number; do not use vague words like "some" or "several".
2. Names the data source in natural mathematical language (e.g. "elliptic curve data", not "ec_curvedata").
3. Briefly invites the user to refine the query if they need to.

Hard constraints on your output:
- Output ONLY that sentence — no text before or after it.
- Do NOT add any preamble or framing such as "Here's a closing line:", "Sure,", or similar.
- Do NOT wrap the sentence in quotation marks or backticks.
- Do NOT restate the user's question. Do NOT start with "I". Do NOT be sycophantic."""

# ── Error messages ────────────────────────────────────────────────────────────

_MSG_EMPTY_RESULT = (
    "The query executed successfully but returned no results. "
    "This may mean the data is not available on this mirror of the LMFDB, "
    "or the combination of filters you specified matches no objects. "
    "Try relaxing one of the constraints — for example, widening the conductor "
    "range or removing a secondary filter."
)

_MSG_SQL_FAILED = (
    "I was unable to generate a valid SQL query for your request. "
    "This sometimes happens for queries that span multiple tables in a complex way, "
    "or that reference invariants not directly stored in the database. "
    "Try rephrasing your question, or ask for a simpler version first."
)

_MSG_EXECUTION_FAILED = (
    "The query was generated but failed during execution. "
    "This is usually caused by a column name mismatch or an unsupported operation. "
    "The technical error was: {error}"
)

_MSG_ROUTER_FAILED = (
    "I was unable to identify which part of the LMFDB is relevant to your query. "
    "Try being more specific about the mathematical objects you are interested in — "
    "for example, 'elliptic curves over Q' rather than 'curves'."
)

_MSG_CLARIFY_FAILED = (
    "I had trouble interpreting your request. "
    "Please try rephrasing it — naming the mathematical object and the property "
    "you are interested in usually helps."
)

_MSG_ANALYSIS_FAILED = (
    "I retrieved the data successfully ({rows} rows) but was unable to generate "
    "the analysis or plot you requested. "
    "The technical error was: {error} "
    "You can still work with the data directly — it is available in the session."
)

_MSG_RATE_LIMITED = (
    "We apologize, but Conductor is temporarily experiencing high demand. "
    "Please wait a moment and try again."
)

_MSG_CREDITS = (
    "We apologize, but Conductor is temporarily unavailable due to a "
    "service issue on our end. Please check back later."
)

# Keywords that indicate the user wants a plot or visualisation
_PLOT_KEYWORDS = {
    "plot", "graph", "chart", "visuali", "scatter",
    "histogram", "draw", "visualise", "visualize"
}

# Keywords that indicate a mathematical/database query — bypass LLM classifier
_MATH_KEYWORDS = {
    # Object types
    "elliptic curve", "modular form", "l-function", "l function",
    "dirichlet character", "dirichlet", "newform", "oldform",
    "maass form", "maass", "hilbert modular", "bianchi",
    "siegel modular", "genus-2", "genus 2", "abelian variety",
    "hypergeometric motive", "artin representation", "artin",
    "number field", "local field", "finite field", "galois group",
    "isogeny class", "modular curve", "lattice",
    # Invariants and properties
    "conductor", "discriminant", "rank", "analytic rank",
    "integral point", "torsion", "regulator", "sha",
    "weierstrass", "isogeny", "bsd", "birch", "swinnerton",
    "j-invariant", "jinv", "cm field", "complex multiplication",
    "bad prime", "reduction", "kodaira", "tamagawa",
    "hecke", "eigenvalue", "eigenform", "level", "weight",
    "character", "nebentypus", "atkin", "fricke",
    "root number", "functional equation", "gamma factor",
    "zero", "zeros", "order of vanishing", "analytic conductor",
    "euler factor", "euler product", "dirichlet series",
    "symmetric square", "tensor product", "rankin",
    "galois representation", "mod-l", "adelic",
    "selmer group", "descent", "height", "canonical height",
    "mordell", "weil", "faltings", "szpiro",
    "p-adic", "iwasawa", "lambda invariant", "mu invariant",
    # Equation forms
    "y^2", "y²", "x^3", "x³", "weierstrass equation",
    "minimal model", "ainvs",
    # Actions implying lookup
    "how many", "find all", "list all", "give me", "fetch",
    "retrieve", "look up", "search for",
    "plot", "graph", "chart", "scatter", "histogram",
    "visuali", "draw", "analyse", "analyze",
    # Field/domain terms
    "over q", "rational point", "integer point", "integral",
    "lmfdb", "database",
}


class LMFDBChat:

    def __init__(self):
        self.history: list[dict] = []
        self.last_df = None
        self.last_sql: str | None = None
        self.last_tables: list[str] = []
        self.last_query_type: str | None = None

    def chat(self, message: str, verbose: bool = False) -> dict:
        """
        Process one conversational turn.
        Returns a dict with keys: message, sql, code, plot, df, error.
        Never raises.
        """
        self.history.append({"role": "user", "content": message})

        # Step 1: intent classification
        try:
            intent_response = self._classify_and_respond(message)
        except Exception as e:
            intent_response = None  # fail open — treat as query

        if intent_response is not None:
            self.history.append({"role": "assistant", "content": intent_response})
            return self._reply(intent_response)

        # Step 2: plan the query — clarity, history-resolved restatement, and query type
        try:
            plan = self._plan(message)
        except Exception as e:
            return self._error_response(_categorise(e, _MSG_CLARIFY_FAILED))

        if plan["action"] == "clarify":
            reply = plan["question"]
            self.history.append({"role": "assistant", "content": reply})
            return self._reply(reply)

        query = plan["refined_query"]
        self.last_query_type = plan["query_type"]

        if verbose:
            print(
                f"  Plan: type={plan['query_type']} output={plan['output_format']} "
                f"needs_analysis={plan['needs_analysis']}"
            )
            print(f"  Refined: {query}")

        # Step 2.5: resolve concrete mathematical objects (equations, labels, polynomials)
        query, lookup_info = resolve(query)

        if verbose:
            print(f"  Lookup: {lookup_info}")

        # Step 3: route to tables
        try:
            tables = route(query, history=self._history_str())
        except ValueError as e:
            reply = str(e)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)
        except Exception as e:
            reply = _categorise(e, _MSG_ROUTER_FAILED)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if verbose:
            print(f"  Tables: {tables}")

        # Step 4: generate SQL
        try:
            sql_result = generate_sql(
                query, tables, lookup_info=lookup_info,
                query_type=plan["query_type"], sql_hint=plan["sql_hint"],
            )
        except Exception as e:
            reply = _categorise(e, _MSG_SQL_FAILED)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        sql = sql_result.get("sql")
        if not sql:
            explanation = sql_result.get("explanation", "")
            reply = _MSG_SQL_FAILED + (f" Details: {explanation}" if explanation else "")
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if verbose:
            print(f"  SQL: {sql}")

        # Step 5: execute
        try:
            df = execute_sql(sql)
        except Exception as e:
            error_detail = str(e).split("\n")[0]
            reply = _MSG_EXECUTION_FAILED.format(error=error_detail)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        # Step 6: handle empty result
        if df.empty:
            self.history.append({"role": "assistant", "content": _MSG_EMPTY_RESULT})
            return {
                "message": _MSG_EMPTY_RESULT,
                "sql": sql,
                "code": None,
                "plot": None,
                "df": [],
                "error": None,
            }

        self.last_df = df
        self.last_sql = sql
        self.last_tables = _extract_tables(sql)

        # Step 7: shape the response by the planned output format.
        output_format = plan["output_format"]
        # An explicit visual request always means a plot, even if the planner missed it.
        if any(kw in message.lower() for kw in _PLOT_KEYWORDS):
            output_format = "plot"

        # Run the Python analysis step whenever the plan needs computation beyond SQL
        # (statistics, correlation, binning, growth, grouped analysis) or a plot.
        needs_analysis = plan["needs_analysis"] or output_format == "plot"
        analysis_out = (
            self.run_analysis_turn(_analysis_instruction(plan, query))
            if needs_analysis else None
        )

        # Plot output: return the figure with the analysis explanation.
        if output_format == "plot":
            reply = (analysis_out or {}).get("message") or f"Query returned {len(df)} rows."
            self.history.append({"role": "assistant", "content": reply})
            return {
                "message": reply,
                "sql": sql,
                "code": (analysis_out or {}).get("code"),
                "plot": (analysis_out or {}).get("plot"),
                "df": df.to_dict(orient="records"),
                "error": (analysis_out or {}).get("error"),
            }

        # Prose output: synthesise from the data, plus any computed analysis result.
        if output_format == "prose":
            analysis_text = (analysis_out or {}).get("result") or ""
            try:
                reply = self._synthesize_response(
                    query, plan["query_type"], df, sql, analysis_text
                )
            except Exception:
                reply = self._safe_provenance(self.last_tables, len(df))
            self.history.append({"role": "assistant", "content": reply})
            return {
                "message": reply,
                "sql": sql,
                "code": (analysis_out or {}).get("code"),
                "plot": None,
                "df": df.to_dict(orient="records"),
                "error": None,
            }

        # Table output (default): provenance line plus tabular data.
        reply = self._safe_provenance(self.last_tables, len(df))
        self.history.append({"role": "assistant", "content": reply})
        return {
            "message": reply,
            "sql": sql,
            "code": None,
            "plot": None,
            "df": df.to_dict(orient="records"),
            "error": None,
        }

    def _ensure_dataframe(self) -> None:
        """
        Rehydrate self.last_df from self.last_sql when the DataFrame is missing.

        last_df lives only in server memory; history/last_sql are restored from the
        stateless frontend payload. On a fresh worker a follow-up analysis would
        otherwise have no data, so re-execute the last SQL to recover it.
        """
        if self.last_df is None and self.last_sql:
            try:
                self.last_df = execute_sql(self.last_sql)
            except Exception:
                pass

    def run_analysis_turn(self, instruction: str) -> dict:
        """
        Process a follow-up analysis instruction on the current DataFrame.
        Returns a dict with keys: message, code, plot, result, error.
        Never raises.
        """
        # Rehydrate the DataFrame from the last SQL if it was lost (e.g. a follow-up
        # request landed on a worker without the in-memory result).
        self._ensure_dataframe()

        if self.last_df is None:
            return {
                "message": (
                    "There is no data in the current session to analyse. "
                    "Please submit a query first to retrieve some data."
                ),
                "code": None,
                "plot": None,
                "result": None,
                "error": "no_dataframe",
            }

        try:
            result = generate_analysis(instruction, self.last_df)
        except Exception as e:
            error_msg = _MSG_ANALYSIS_FAILED.format(
                rows=len(self.last_df),
                error=_categorise(e, str(e).split("\n")[0])
            )
            return {"message": error_msg, "code": None, "plot": None, "result": None, "error": str(e)}

        code = result.get("code")
        explanation = result.get("explanation", "")

        if not code:
            msg = explanation or (
                "I was unable to generate analysis code for this request. "
                "The data may not contain the columns needed. "
                f"Available columns: {', '.join(self.last_df.columns.tolist())}."
            )
            return {"message": msg, "code": None, "plot": None, "result": None, "error": None}

        try:
            analysis_out = execute_analysis(code, self.last_df)
        except TimeoutError:
            return {
                "message": (
                    "The analysis code took too long to execute and was stopped. "
                    "Try requesting a simpler analysis or a smaller dataset."
                ),
                "code": code,
                "plot": None,
                "result": None,
                "error": "timeout",
            }
        except Exception as e:
            error_detail = str(e).split("\n")[0]
            return {
                "message": (
                    f"The analysis code encountered an error: {error_detail}. "
                    "The generated code is included so you can inspect or modify it."
                ),
                "code": code,
                "plot": None,
                "result": None,
                "error": error_detail,
            }

        msg = explanation
        if self.last_tables:
            msg = explanation + (
                f" Data sourced from: {', '.join(self.last_tables)}. "
                "Does this look right, or would you like to adjust the query or analysis?"
            )

        return {
            "message": msg,
            "code": code,
            "plot": analysis_out.get("plot"),
            "result": analysis_out.get("result"),
            "error": None,
        }

    def state(self) -> dict:
        """Return serialisable session state for the GET /session endpoint."""
        return {
            "history": self.history,
            "last_sql": self.last_sql,
            "last_tables": self.last_tables,
            "last_query_type": self.last_query_type,
            "has_dataframe": self.last_df is not None,
            "dataframe_shape": (
                list(self.last_df.shape) if self.last_df is not None else None
            ),
            "dataframe_columns": (
                self.last_df.columns.tolist() if self.last_df is not None else None
            ),
        }

    # ── Private methods ───────────────────────────────────────────────────────

    def _classify_and_respond(self, message: str) -> str | None:
        """
        Returns None if the message is a database/analysis query (proceed through pipeline).
        Returns a response string if conversational or off-topic.

        First checks a keyword list — if any math keyword is present, routes directly
        to the pipeline without calling the LLM classifier. This prevents the LLM from
        "solving" mathematical questions itself rather than looking them up.

        Falls back to LLM classification for ambiguous messages.
        Fails open: returns None on any exception so queries are never blocked.
        """
        # Fast path: if any math keyword present, always route to pipeline
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in _MATH_KEYWORDS):
            return None

        client = Anthropic()

        # Build recent history for context
        messages = []
        for m in self.history[:-1][-6:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        # Classify with LLM
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            system=_INTENT_SYSTEM,
            messages=messages,
        )
        result = r.content[0].text.strip().upper()

        if "QUERY" in result:
            return None

        # It's CHAT — generate a natural response
        return self._chat_response(message, messages)

    def _chat_response(self, message: str, messages: list[dict]) -> str:
        """Generate a natural conversational response with full conversation context."""
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            system=_CHAT_SYSTEM,
            messages=messages,
        )
        return _strip_meta(r.content[0].text)

    def _plan(self, message: str) -> dict:
        """
        Plan the query: clarity decision, history-resolved standalone restatement,
        and query-type classification. Returns a normalized plan dict (see
        _normalize_plan) so downstream stages always get complete, valid fields.
        """
        system = _PLAN_SYSTEM.replace("<<HISTORY>>", self._history_str())
        client = Anthropic()
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": message}]
        )
        return _normalize_plan(_parse(r.content[0].text), fallback_query=message)

    def _provenance_message(self, tables: list[str], rows: int) -> str:
        """Generate a natural provenance + confirmation message via Haiku."""
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=_PROVENANCE_SYSTEM.format(
                tables=", ".join(tables) if tables else "the LMFDB",
                rows=rows,
            ),
            messages=[{"role": "user", "content": "Generate the closing line."}]
        )
        return _strip_meta(r.content[0].text)

    def _safe_provenance(self, tables: list[str], rows: int) -> str:
        """Provenance line, falling back to a plain count if the LLM call fails."""
        try:
            return self._provenance_message(tables, rows)
        except Exception:
            return f"Query returned {rows} rows."

    def _synthesize_response(
        self, query: str, query_type: str, df, sql: str, analysis_text: str = ""
    ) -> str:
        """
        Produce a natural-prose answer from the result DataFrame (and optional
        analysis output). Used for prose query types (scalar/boolean/count/
        aggregate/prose) instead of returning a raw table.

        Runs on Sonnet for answer reliability, passes the executed SQL so the
        model can detect LIMIT truncation, and bounds tokens by sending only the
        first rows of df alongside the true row count.
        """
        analysis_block = (
            f"Computed analysis result:\n{analysis_text}\n" if analysis_text else ""
        )
        system = (
            _SYNTHESIS_PROMPT
            .replace("<<QUERY>>", query)
            .replace("<<QUERY_TYPE>>", query_type)
            .replace("<<ROW_COUNT>>", str(len(df)))
            .replace("<<COLUMNS>>", ", ".join(map(str, df.columns)))
            .replace("<<SQL>>", sql or "(none)")
            .replace("<<DATA>>", df.head(20).to_json(orient="records"))
            .replace("<<ANALYSIS>>", analysis_block)
        )
        client = Anthropic()
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": query}]
        )
        return _strip_meta(r.content[0].text)

    def _history_str(self) -> str:
        if not self.history:
            return "(none)"
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self.history[-6:]
        )

    def _reply(self, message: str) -> dict:
        return {
            "message": message,
            "sql": None,
            "code": None,
            "plot": None,
            "df": None,
            "error": None,
        }

    def _error_response(self, message: str) -> dict:
        return {
            "message": message,
            "sql": None,
            "code": None,
            "plot": None,
            "df": None,
            "error": message,
        }


# ── Module-level helpers ──────────────────────────────────────────────────────

def _extract_tables(sql: str) -> list[str]:
    """Extract table names from a SQL query (after FROM and JOIN keywords)."""
    pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?\w+)?'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    seen = set()
    result = []
    for m in matches:
        if m.lower() not in seen:
            seen.add(m.lower())
            result.append(m)
    return result


# Leading filler ("Sure,", "Certainly.") and meta framing clauses ("Here's a
# natural closing line:") that the model sometimes prepends instead of returning
# only the answer. Kept deliberately narrow so it never eats real content.
_QUOTE_CHARS = "\"'“”"
_FILLER_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|okay|ok|got it|here you go)\b[,.!:]*\s+",
    re.IGNORECASE,
)
_FRAMING_RE = re.compile(
    r"^\s*here(?:'s| is)\b[^:\n]*\b(?:line|sentence|response|answer|message|closing)\b[^:\n]*:\s+",
    re.IGNORECASE,
)


def _strip_meta(text: str) -> str:
    """
    Remove LLM framing/preamble and wrapping quotes so only the answer remains.

    Strips two artifacts: leading filler ("Sure, ...") and meta framing clauses
    ("Here's a natural closing line: ..."), plus quotes/backticks wrapping the
    whole message. Conservative — the framing pattern only fires when a meta noun
    (line/sentence/response/answer/message/closing) precedes the colon, so real
    content like "Here is the rank: 2" is left untouched.
    """
    t = text.strip()
    # Strip surrounding code fences.
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    # Strip leading filler, then a framing clause (filler may precede framing).
    for pattern in (_FILLER_RE, _FRAMING_RE):
        stripped = pattern.sub("", t, count=1).strip()
        if stripped:
            t = stripped
    # Strip symmetric wrapping quotes.
    if len(t) >= 2 and t[0] in _QUOTE_CHARS and t[-1] in _QUOTE_CHARS:
        t = t[1:-1].strip()
    return t


_QUERY_TYPES = {"scalar", "boolean", "count", "aggregate", "tabular", "analytical", "prose"}
_OUTPUT_FORMATS = {"prose", "table", "plot"}
# Types whose natural output is prose rather than a table.
_PROSE_TYPES = {"scalar", "boolean", "count", "aggregate", "prose"}


def _normalize_plan(plan: dict, fallback_query: str) -> dict:
    """
    Coerce a raw planner response into a complete, valid plan dict.

    Fills defaults and rejects unknown enum values so downstream stages never see
    missing keys or bad values. Defaults degrade to today's behaviour: a
    proceed / tabular / table query that needs no analysis.
    """
    action = plan.get("action")
    if action not in ("proceed", "clarify"):
        action = "proceed"

    query_type = plan.get("query_type")
    if query_type not in _QUERY_TYPES:
        query_type = "tabular"

    output_format = plan.get("output_format")
    if output_format not in _OUTPUT_FORMATS:
        output_format = "prose" if query_type in _PROSE_TYPES else "table"

    return {
        "action": action,
        "question": (plan.get("question") or "").strip(),
        "refined_query": plan.get("refined_query") or fallback_query,
        "query_type": query_type,
        "output_format": output_format,
        "needs_analysis": bool(plan.get("needs_analysis", False)),
        "analysis_instruction": (plan.get("analysis_instruction") or "").strip(),
        "sql_hint": (plan.get("sql_hint") or "").strip(),
    }


def _analysis_instruction(plan: dict, query: str) -> str:
    """
    Instruction for the analysis step. Use the planner's explicit instruction when
    present; otherwise (e.g. a plot triggered by a keyword rather than by the planner
    setting needs_analysis) fall back to the standalone refined query and let the
    analysis step infer the right computation/visualisation for it.
    """
    if plan["analysis_instruction"]:
        return plan["analysis_instruction"]
    return (
        f'For this request: "{query}", produce the most informative analysis or '
        "visualisation of the retrieved data, inferring sensible columns and method "
        "from the request and the DataFrame."
    )


def _categorise(exc: Exception, default: str = _MSG_ROUTER_FAILED) -> str:
    """
    Map an exception to a human-readable error message.

    Global API conditions (rate limiting, exhausted credits) are detected and
    returned regardless of which stage raised. Otherwise the caller-supplied
    `default` is returned, so the message reflects the pipeline stage that
    actually failed rather than always blaming routing.
    """
    msg = str(exc)
    if "rate_limit" in msg.lower() or "429" in msg:
        return _MSG_RATE_LIMITED
    if "credit balance" in msg.lower() or "too low" in msg.lower() or "402" in msg:
        return _MSG_CREDITS
    return default


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
