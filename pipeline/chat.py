"""
Description: This file holds all session state (history, last DataFrame, last SQL) and executes the chatbot functionality of this package. 
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
Do not ask unnecessary clarifying questions.

Conversation history:
<<HISTORY>>"""


class LMFDBChat:

    def __init__(self):
        self.history: list[dict] = []
        self.last_df = None
        self.last_sql: str | None = None

    def chat(self, message: str, verbose: bool = False) -> dict:
        """
        Process one conversational turn.
        Returns a dict with keys: message, sql, code, plot, df, error.
        """
        self.history.append({"role": "user", "content": message})

        clarification = self._clarify(message)
        if clarification["action"] == "clarify":
            reply = clarification["question"]
            self.history.append({"role": "assistant", "content": reply})
            return {"message": reply, "sql": None, "code": None, "plot": None, "df": None, "error": None}

        query = clarification["refined_query"]
        tables = route(query, history=self._history_str())
        sql_result = generate_sql(query, tables)
        sql = sql_result.get("sql")

        if not sql:
            reply = f"I could not generate a query: {sql_result.get('explanation')}"
            self.history.append({"role": "assistant", "content": reply})
            return {"message": reply, "sql": None, "code": None, "plot": None, "df": None, "error": None}

        df = execute_sql(sql)
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
        """
        if self.last_df is None:
            return {"message": None, "code": None, "plot": None, "error": "No dataframe in session."}

        result = generate_analysis(instruction, self.last_df)
        code = result.get("code")
        explanation = result.get("explanation")

        if not code:
            return {"message": explanation, "code": None, "plot": None, "error": None}

        plot_b64 = execute_analysis(code, self.last_df)

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
            "dataframe_shape": list(self.last_df.shape) if self.last_df is not None else None,
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


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
