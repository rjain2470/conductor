"""
Description: This file holds all session state (history, last DataFrame, last SQL) and executes the chatbot functionality of this package. 

Error handling: 
- Every failure mode returns a structured, human-readable response.
- Errors are categorised: clarification needed, no data found, query generation failed, execution failed, analysis failed.
"""

import re
import json
from anthropic import Anthropic
from pipeline.router import route
from pipeline.sql_gen import generate_sql
from pipeline.executor import execute_sql
from pipeline.analysis import generate_analysis, execute_analysis

# ── System prompts ────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """You are a query classifier for Conductor, an LMFDB interface that CAN generate plots and analyses.

Classify the user message as either QUERY or CHAT.

Output QUERY if the message:
- Asks for data from the LMFDB database
- Asks for a plot, graph, chart, or visualisation
- Asks for analysis of data
- Refers to previous results and asks to filter, refine, or extend them
- Is any mathematical question that could be answered with database data

Output CHAT if the message:
- Is a greeting, thanks, or small talk
- Asks something unrelated to mathematics or the LMFDB
- Is a general mathematical question not requiring database lookup

Respond with exactly one word: either QUERY or CHAT. Nothing else."""

_CHAT_SYSTEM = """You are Conductor, a mathematically knowledgeable assistant with access to the LMFDB (L-functions and Modular Forms Database). You have a warm, understated personality — think a good graduate student who knows their stuff and doesn't waste words.

You CAN generate plots and analyses of LMFDB data when asked.

If someone greets you, chats, or thanks you, respond naturally and briefly as yourself. You can show a little personality. Don't be formal.

If someone asks something unrelated to mathematics or the LMFDB — conferences, restaurants, life advice — gently let them know what you're here for, but without being robotic about it."""

_CLARIFY_SYSTEM = """You are a mathematical assistant specialising in the LMFDB database.
Assess whether the user query is clear enough to act on.

Return ONLY a raw JSON object:
- If clear: {"action": "proceed", "refined_query": "<restate precisely>"}
- If ambiguous: {"action": "clarify", "question": "<one focused question>"}

Flag ambiguity only when it would materially change what is queried.
Do not ask for clarification that is not mathematically necessary.

Conversation history so far:
<<HISTORY>>"""

_PROVENANCE_SYSTEM = """You are writing a brief, natural closing line for a mathematical
database query response.

The query returned data from the following SQL tables: {tables}
The query returned {rows} rows.

Write one sentence that:
1. Mentions which table or tables the data came from, using natural mathematical
   language (e.g. "elliptic curve data" rather than "ec_curvedata")
2. Asks whether this is what the user was looking for, or whether they would
   like to refine the query

Be concise and natural. Do not be sycophantic. Do not start with "I"."""

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
_PLOT_KEYWORDS = {"plot", "graph", "chart", "visuali", "scatter", "histogram", "draw", "visualise", "visualize"}


class LMFDBChat:

    def __init__(self):
        self.history: list[dict] = []
        self.last_df = None
        self.last_sql: str | None = None
        self.last_tables: list[str] = []

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

        # Step 2: mathematical clarification
        try:
            clarification = self._clarify(message)
        except Exception as e:
            return self._error_response(_categorise(e))

        if clarification["action"] == "clarify":
            reply = clarification["question"]
            self.history.append({"role": "assistant", "content": reply})
            return self._reply(reply)

        query = clarification["refined_query"]

        # Step 3: route to tables
        try:
            tables = route(query, history=self._history_str())
        except ValueError as e:
            reply = str(e)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)
        except Exception as e:
            reply = _categorise(e)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if verbose:
            print(f"  Tables: {tables}")

        # Step 4: generate SQL
        try:
            sql_result = generate_sql(query, tables)
        except Exception as e:
            reply = _categorise(e)
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

        # Step 7: generate provenance + confirmation message
        try:
            reply = self._provenance_message(self.last_tables, len(df))
        except Exception:
            reply = f"Query returned {len(df)} rows."

        self.history.append({"role": "assistant", "content": reply})

        # Step 8: auto-run analysis if the original message requests a plot
        wants_plot = any(kw in message.lower() for kw in _PLOT_KEYWORDS)

        if wants_plot:
            analysis_result = self.run_analysis_turn(message)
            return {
                "message": reply,
                "sql": sql,
                "code": analysis_result.get("code"),
                "plot": analysis_result.get("plot"),
                "df": df.to_dict(orient="records"),
                "error": analysis_result.get("error"),
            }

        return {
            "message": reply,
            "sql": sql,
            "code": None,
            "plot": None,
            "df": df.to_dict(orient="records"),
            "error": None,
        }

    def run_analysis_turn(self, instruction: str) -> dict:
        """
        Process a follow-up analysis instruction on the current DataFrame.
        Returns a dict with keys: message, code, plot, error.
        Never raises.
        """
        if self.last_df is None:
            return {
                "message": (
                    "There is no data in the current session to analyse. "
                    "Please submit a query first to retrieve some data."
                ),
                "code": None,
                "plot": None,
                "error": "no_dataframe",
            }

        try:
            result = generate_analysis(instruction, self.last_df)
        except Exception as e:
            error_msg = _MSG_ANALYSIS_FAILED.format(
                rows=len(self.last_df),
                error=_categorise(e)
            )
            return {"message": error_msg, "code": None, "plot": None, "error": str(e)}

        code = result.get("code")
        explanation = result.get("explanation", "")

        if not code:
            msg = explanation or (
                "I was unable to generate analysis code for this request. "
                "The data may not contain the columns needed. "
                f"Available columns: {', '.join(self.last_df.columns.tolist())}."
            )
            return {"message": msg, "code": None, "plot": None, "error": None}

        try:
            plot_b64 = execute_analysis(code, self.last_df)
        except TimeoutError:
            return {
                "message": (
                    "The analysis code took too long to execute and was stopped. "
                    "Try requesting a simpler analysis or a smaller dataset."
                ),
                "code": code,
                "plot": None,
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
                "error": error_detail,
            }

        # Append provenance to analysis response if tables are known
        msg = explanation
        if self.last_tables:
            msg = explanation + (
                f" Data sourced from: {', '.join(self.last_tables)}. "
                "Does this look right, or would you like to adjust the query or analysis?"
            )

        return {
            "message": msg,
            "code": code,
            "plot": plot_b64,
            "error": None,
        }

    def state(self) -> dict:
        """Return serialisable session state for the GET /session endpoint."""
        return {
            "history": self.history,
            "last_sql": self.last_sql,
            "last_tables": self.last_tables,
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
        Uses a two-call approach: classify first (max_tokens=10), then respond if CHAT.
        Fails open: returns None on any exception so queries are never blocked.
        """
        client = Anthropic()

        # Build recent history for context
        messages = []
        for m in self.history[:-1][-6:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        # Step 1: classify
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            system=_INTENT_SYSTEM,
            messages=messages,
        )
        result = r.content[0].text.strip().upper()

        if "QUERY" in result:
            return None

        # Step 2: it's CHAT — generate a natural response with full context
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
        return r.content[0].text.strip()

    def _clarify(self, message: str) -> dict:
        system = _CLARIFY_SYSTEM.replace("<<HISTORY>>", self._history_str())
        client = Anthropic()
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": message}]
        )
        return _parse(r.content[0].text)

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
        return r.content[0].text.strip()

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
    """
    Extract table names from a SQL query.
    Finds names after FROM and JOIN keywords.
    """
    pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?\w+)?'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    seen = set()
    result = []
    for m in matches:
        if m.lower() not in seen:
            seen.add(m.lower())
            result.append(m)
    return result


def _categorise(exc: Exception) -> str:
    """Map an exception to a human-readable error message."""
    msg = str(exc)
    if "rate_limit" in msg.lower() or "429" in msg:
        return _MSG_RATE_LIMITED
    if "credit balance" in msg.lower() or "too low" in msg.lower() or "402" in msg:
        return _MSG_CREDITS
    return _MSG_ROUTER_FAILED


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
