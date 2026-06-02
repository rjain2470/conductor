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

# ── Prompts ───────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """You are Conductor, an assistant for the LMFDB mathematical database. Warm, understated, knowledgeable — like a good graduate student.

- Greetings/thanks/small talk: respond briefly and naturally.
- Off-topic questions: redirect without partially answering.
- Ambiguous messages: ask one short clarifying question.
- Mathematical or database queries: respond with exactly: QUERY"""

_CLARIFY_SYSTEM = """You are a mathematical assistant for the LMFDB database.
Only ask for clarification if the ambiguity would fundamentally change which table or filter is used.
Do not ask about plot types, axis choices, or standard mathematical terminology.
Do not ask if the user means the standard interpretation of a term.
When plot type or analysis method is unspecified, proceed with a sensible default.

Return ONLY a JSON object:
- If clear enough to query: {"action": "proceed", "refined_query": "<restate precisely, no column or table names>"}
- If genuinely ambiguous: {"action": "clarify", "question": "<one focused question>"}

History: <<HISTORY>>"""

_SUMMARISE_SYSTEM = """You are Conductor, a mathematical database assistant.
The user asked a factual question. Answer it directly in one or two sentences of precise mathematical language using the data below. Do not describe what you did.
Then on a new sentence, add a brief professional closing (e.g. "Let me know if you need anything further.").

Data: {data}"""

_PROVENANCE_SYSTEM = """You are Conductor, a warm and professional mathematical database assistant.
You returned {rows} result(s). Write one closing sentence:
- 1 result: confirm the object was found, invite follow-up.
- Multiple results: note the count, invite refinement or follow-up.
No table names. Be concise and professional."""

# ── Error messages ────────────────────────────────────────────────────────────

_MSG_EMPTY = (
    "The query returned no results. The data may not be available on this mirror, "
    "or the filters match no objects. Try relaxing a constraint such as widening "
    "the conductor range."
)
_MSG_SQL_FAILED = (
    "I was unable to generate a valid query for this request. Maybe"
    "try rephrasing or asking for a simpler version first?"
)
_MSG_EXEC_FAILED = (
    "The query failed during execution. Technical error: {error}"
)
_MSG_ROUTER_FAILED = (
    "I could not identify the relevant part of the LMFDB for this query. "
    "Could you be a bit more specific?"
)
_MSG_ANALYSIS_FAILED = (
    "Data retrieved ({rows} rows) but analysis failed. Error: {error}."
    "The data is still available in the session."
)
_MSG_RATE_LIMITED = (
    "Conductor is temporarily experiencing high demand. Please try again in a moment."
)
_MSG_CREDITS = (
    "Conductor is temporarily unavailable due to a service issue. Please check back later."
)

# Factual question patterns — use summarise instead of provenance
_FACTUAL_RE = re.compile(
    r'\b(how many|what is|what are|what was|find the|compute|calculate|'
    r'does|is it|is there|is this|give me the)\b',
    re.IGNORECASE
)


class LMFDBChat:

    def __init__(self):
        self.history: list[dict] = []
        self.last_df = None
        self.last_sql: str | None = None
        self.last_tables: list[str] = []
        self.last_lookup: dict | None = None

    def chat(self, message: str, verbose: bool = False) -> dict:
        self.history.append({"role": "user", "content": message})

        # Step 1: intent classification
        try:
            intent_response = self._classify_and_respond(message)
        except Exception:
            intent_response = None

        if intent_response is not None:
            self.history.append({"role": "assistant", "content": intent_response})
            return self._reply(intent_response)

        # Step 2: clarification — on the original message before lookup
        try:
            clarification = self._clarify(message)
        except Exception as e:
            return self._error_response(_categorise(e))

        if clarification["action"] == "clarify":
            reply = clarification["question"]
            self.history.append({"role": "assistant", "content": reply})
            return self._reply(reply)

        query = clarification["refined_query"]

        # Step 3: lookup — on the clarified query
        try:
            query_with_lookup, lookup_info = resolve(query)
            self.last_lookup = lookup_info
        except Exception:
            query_with_lookup = query
            lookup_info = None
            self.last_lookup = None

        # Step 4: route
        try:
            tables = route(query_with_lookup, history=self._history_str())
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
            sql_result = generate_sql(
                query_with_lookup, tables, lookup_info=lookup_info
            )
        except Exception as e:
            reply = _categorise(e)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        sql = sql_result.get("sql")
        if not sql:
            explanation = sql_result.get("explanation", "")
            reply = _MSG_SQL_FAILED + (f" {explanation}" if explanation else "")
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if verbose:
            print(f"  SQL: {sql}")

        # Step 6: execute
        try:
            df = execute_sql(sql)
        except Exception as e:
            error_detail = str(e).split("\n")[0]
            reply = _MSG_EXEC_FAILED.format(error=error_detail)
            self.history.append({"role": "assistant", "content": reply})
            return self._error_response(reply)

        if df.empty:
            self.history.append({"role": "assistant", "content": _MSG_EMPTY})
            return {
                "message": _MSG_EMPTY,
                "sql": sql,
                "code": None,
                "plot": None,
                "df": [],
                "error": None,
            }

        self.last_df = df
        self.last_sql = sql
        self.last_tables = _extract_tables(sql)

        # Step 7: generate closing message
        # Use summarise for factual questions, provenance for data retrieval
        try:
            if _FACTUAL_RE.search(message):
                reply = self._summarise(message, df)
            else:
                reply = self._provenance_message(len(df))
        except Exception:
            reply = f"Returned {len(df)} result(s). Let me know if you need anything further."

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
        if self.last_df is None:
            return {
                "message": (
                    "No data in the current session. "
                    "Please submit a query first."
                ),
                "code": None,
                "plot": None,
                "error": "no_dataframe",
            }

        try:
            result = generate_analysis(instruction, self.last_df)
        except Exception as e:
            return {
                "message": _MSG_ANALYSIS_FAILED.format(
                    rows=len(self.last_df), error=_categorise(e)
                ),
                "code": None,
                "plot": None,
                "error": str(e),
            }

        code = result.get("code")
        explanation = result.get("explanation", "")

        if not code:
            msg = explanation or (
                "Unable to generate analysis code. "
                f"Available columns: {', '.join(self.last_df.columns.tolist())}."
            )
            return {"message": msg, "code": None, "plot": None, "error": None}

        try:
            plot_b64 = execute_analysis(code, self.last_df)
        except TimeoutError:
            return {
                "message": (
                    "Analysis timed out. "
                    "Try a simpler analysis or a smaller dataset."
                ),
                "code": code,
                "plot": None,
                "error": "timeout",
            }
        except Exception as e:
            return {
                "message": (
                    f"Analysis encountered an error: {str(e).split(chr(10))[0]}. "
                    "The generated code is included for inspection."
                ),
                "code": code,
                "plot": None,
                "error": str(e).split("\n")[0],
            }

        try:
            closing = self._provenance_message(len(self.last_df))
        except Exception:
            closing = "Let me know if you need anything further."

        msg = (explanation + " " + closing).strip() if explanation else closing

        return {
            "message": msg,
            "code": code,
            "plot": plot_b64,
            "error": None,
        }

    def state(self) -> dict:
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

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify_and_respond(self, message: str) -> str | None:
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": message}]
        )
        result = r.content[0].text.strip()
        return None if result == "QUERY" else result

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

    def _summarise(self, question: str, df) -> str:
        data_str = df.head(5).to_json(orient="records", indent=2)
        client = Anthropic()
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=120,
            system=_SUMMARISE_SYSTEM.format(data=data_str),
            messages=[{"role": "user", "content": question}]
        )
        return r.content[0].text.strip()

    def _provenance_message(self, rows: int) -> str:
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
            "message": message, "sql": None, "code": None,
            "plot": None, "df": None, "error": None,
        }

    def _error_response(self, message: str) -> dict:
        return {
            "message": message, "sql": None, "code": None,
            "plot": None, "df": None, "error": message,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_tables(sql: str) -> list[str]:
    pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?\w+)?'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    seen, result = set(), []
    for m in matches:
        if m.lower() not in seen:
            seen.add(m.lower())
            result.append(m)
    return result


def _categorise(exc: Exception) -> str:
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
