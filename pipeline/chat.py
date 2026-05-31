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

_CLARIFY_SYSTEM = """You are a mathematical assistant specialising in the LMFDB database.
Assess whether the user query is clear enough to act on.

Return ONLY a raw JSON object:
- If clear: {"action": "proceed", "refined_query": "<restate precisely>"}
- If ambiguous: {"action": "clarify", "question": "<one focused question>"}

Flag ambiguity only when it would materially change what is queried.
Do not ask for clarification that is not mathematically necessary.

Conversation history so far:
<<HISTORY>>"""

# Human-readable messages for each failure mode
_MSG_EMPTY_RESULT = (
    "The query executed successfully but returned no results. "
    "This may mean the data is not available on this mirror of the LMFDB, "
    "or the combination of filters you specified matches no objects. "
    "Try relaxing one of the constraints — for example, widening the conductor range "
    "or removing a secondary filter."
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
    "The Anthropic API rate limit was reached. "
    "Please wait a moment and try again."
)


class LMFDBChat:

    def __init__(self):
        self.history: list[dict] = []
        self.last_df = None
        self.last_sql: str | None = None

    def chat(self, message: str, verbose: bool = False) -> dict:
        """
        Process one conversational turn.
        Returns a dict with keys: message, sql, code, plot, df, error.
        Never raises — all failures return a structured response.
        """
        self.history.append({"role": "user", "content": message})

        # Step 1: clarify if needed
        try:
            clarification = self._clarify(message)
        except Exception as e:
            return self._error_response(_rate_limit_or(e, _MSG_ROUTER_FAILED))

        if clarification["action"] == "clarify":
            reply = clarification["question"]
            self.history.append({"role": "assistant", "content": reply})
            return self._reply(reply)

        query = clarification["refined_query"]

        # Step 2: route to tables
        try:
            tables = route(query, history=self._history_str())
        except ValueError as e:
            reply = str(e)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)
        except Exception as e:
            reply = _rate_limit_or(e, _MSG_ROUTER_FAILED)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if verbose:
            print(f"  Tables: {tables}")

        # Step 3: generate SQL
        try:
            sql_result = generate_sql(query, tables)
        except Exception as e:
            reply = _rate_limit_or(e, _MSG_SQL_FAILED)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        sql = sql_result.get("sql")
        if not sql:
            explanation = sql_result.get("explanation", "")
            reply = (
                f"{_MSG_SQL_FAILED}"
                + (f" Details: {explanation}" if explanation else "")
            )
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if verbose:
            print(f"  SQL: {sql}")

        # Step 4: execute
        try:
            df = execute_sql(sql)
        except Exception as e:
            error_detail = str(e).split("\n")[0]  # first line only
            reply = _MSG_EXECUTION_FAILED.format(error=error_detail)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        # Step 5: handle empty result
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

        reply = f"Query returned {len(df)} rows."
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

        # Generate analysis code
        try:
            result = generate_analysis(instruction, self.last_df)
        except Exception as e:
            error_msg = _MSG_ANALYSIS_FAILED.format(
                rows=len(self.last_df),
                error=_rate_limit_or(e, str(e).split("\n")[0])
            )
            return {"message": error_msg, "code": None, "plot": None, "error": str(e)}

        code = result.get("code")
        explanation = result.get("explanation", "")

        if not code:
            msg = explanation or (
                "I was unable to generate analysis code for this request. "
                "The data may not contain the columns needed for this analysis. "
                f"Available columns: {', '.join(self.last_df.columns.tolist())}."
            )
            return {"message": msg, "code": None, "plot": None, "error": None}

        # Execute analysis code
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
                    f"The analysis code ran but encountered an error: {error_detail}. "
                    "The generated code is included so you can inspect or modify it."
                ),
                "code": code,
                "plot": None,
                "error": error_detail,
            }

        return {
            "message": explanation,
            "code": code,
            "plot": plot_b64,
            "error": None,
        }

    def state(self) -> dict:
        """Return serialisable session state for the GET /session endpoint."""
        return {
            "history": self.history,
            "last_sql": self.last_sql,
            "has_dataframe": self.last_df is not None,
            "dataframe_shape": (
                list(self.last_df.shape) if self.last_df is not None else None
            ),
            "dataframe_columns": (
                self.last_df.columns.tolist() if self.last_df is not None else None
            ),
        }

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


def _rate_limit_or(exc: Exception, fallback: str) -> str:
    """Return a rate limit message if the exception is a rate limit error,
    otherwise return the fallback message."""
    msg = str(exc)
    if "rate_limit" in msg.lower() or "429" in msg:
        return _MSG_RATE_LIMITED
    return fallback


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
