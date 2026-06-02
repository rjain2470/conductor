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
from pipeline.lookup import resolve

# ── System prompts ────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """You are Conductor, a mathematically knowledgeable assistant with access to the LMFDB (L-functions and Modular Forms Database). You have a warm, understated personality — think a good graduate student who knows their stuff and doesn't waste words.

If someone greets you, chats, or thanks you, respond naturally and briefly as yourself. You can show a little personality. Don't be formal.

If someone asks something unrelated to mathematics or the LMFDB — conferences, restaurants, life advice — gently let them know what you're here for, but without being robotic about it. Do not attempt to partially answer off-topic questions — just redirect.

If you genuinely can't tell whether something is a database query or something else, ask a short natural question to find out.

If the message is a mathematical or database query, respond with exactly the single word: QUERY

Never explain your reasoning. Either respond as yourself, or output QUERY."""

_CLARIFY_SYSTEM = """You are a mathematical assistant specialising in the LMFDB database.
Assess whether the user query is clear enough to act on.

Return ONLY a raw JSON object:
- If clear: {"action": "proceed", "refined_query": "<restate precisely>"}
- If ambiguous: {"action": "clarify", "question": "<one focused question>"}

Flag ambiguity only when it would materially change what is queried.
Do not ask for clarification that is not mathematically necessary.

Conversation history so far:
<<HISTORY>>"""

_PROVENANCE_SYSTEM = """You are Conductor, a mathematically knowledgeable assistant with a warm, understated personality.

You have just returned {rows} rows of data to the user. Write a single short closing line that:
- Confirms the result naturally (e.g. mentions the row count if interesting)
- Asks if it's what they were looking for, or if there's anything else

Keep it to one sentence. Vary the phrasing. Don't be formal. Don't mention table names.
Examples of good closing lines:
- "Found {rows} results — does that cover what you needed?"
- "That's everything matching your query. Anything else?"
- "Got {rows} — is that what you had in mind?"
"""

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
    "Could you rephrase your question, or ask for a simpler version to start?"
)

_MSG_EXECUTION_FAILED = (
    "The query was generated but failed during execution. "
    "This is usually caused by a column name mismatch or an unsupported operation. "
    "The error message was: {error}"
)

_MSG_ROUTER_FAILED = (
    "I was unable to identify which part of the LMFDB is relevant to your query. "
    "Could you be a bit more specific?"
)

_MSG_ANALYSIS_FAILED = (
    "I retrieved the data successfully ({rows} rows) but was unable to generate "
    "the analysis or plot you requested. "
    "The technical error was: {error} "
    "You can still work with the data directly if you'd like — it is available in the session."
)

_MSG_RATE_LIMITED = (
    "We apologize, but Conductor is temporarily experiencing high demand. "
    "Please wait a moment and try again."
)

_MSG_CREDITS = (
    "We apologize, but Conductor is temporarily unavailable due to a "
    "service issue on our end. Please check back later."
)


class LMFDBChat:

    def __init__(self):
        self.history: list[dict] = []
        self.last_df = None
        self.last_sql: str | None = None
        self.last_tables: list[str] = []
        self.last_lookup: dict | None = None

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
        except Exception:
            intent_response = None  # fail open — treat as query

        if intent_response is not None:
            self.history.append({"role": "assistant", "content": intent_response})
            return self._reply(intent_response)

        # Step 2: lookup — resolve concrete mathematical objects
        try:
            query_with_lookup, lookup_info = resolve(message)
            self.last_lookup = lookup_info
        except Exception:
            query_with_lookup = message
            lookup_info = None
            self.last_lookup = None

        # Step 3: mathematical clarification
        try:
            clarification = self._clarify(query_with_lookup)
        except Exception as e:
            return self._error_response(_categorise(e))

        if clarification["action"] == "clarify":
            reply = clarification["question"]
            self.history.append({"role": "assistant", "content": reply})
            return self._reply(reply)

        query = clarification["refined_query"]

        # Step 4: route to tables
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

        # Step 5: generate SQL
        try:
            sql_result = generate_sql(query, tables, lookup_info=lookup_info)
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

        # Step 6: execute
        try:
            df = execute_sql(sql)
        except Exception as e:
            error_detail = str(e).split("\n")[0]
            reply = _MSG_EXECUTION_FAILED.format(error=error_detail)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        # Step 7: handle empty result
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

        # Step 8: generate closing message
        try:
            reply = self._provenance_message(len(df))
        except Exception:
            reply = f"Found {len(df)} results. Is that what you were looking for?"

        self.history.append({"role": "assistant", "content": reply})

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

        # Append closing line to analysis response
        try:
            closing = self._provenance_message(len(self.last_df))
        except Exception:
            closing = "Is there anything else I can help you with?"

        msg = explanation + " " + closing if explanation else closing

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
        Returns None if the message is a database query (proceed through pipeline).
        Returns a response string if conversational, off-topic, or ambiguous.
        Fails open: returns None on any exception so queries are never blocked.
        """
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": message}]
        )
        result = r.content[0].text.strip()
        if result == "QUERY":
            return None
        return result

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

    def _provenance_message(self, rows: int) -> str:
        """Generate a short, varied closing line via Haiku."""
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=60,
            system=_PROVENANCE_SYSTEM.format(rows=rows),
            messages=[{"role": "user", "content": "Write the closing line."}]
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
    """Extract table names from a SQL query after FROM and JOIN keywords."""
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
