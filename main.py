"""
Description: This is the FastAPI application entry point. 
- Session management: TTL eviction for 30 min inactivity.
- Rate limiting: 20 requests per minute per session, sliding window.

NOTE: CORS is open for now; tighten allow_origins before public release once frontend is set.
"""

import os
import time
import uuid
from collections import defaultdict
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Cookie, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.chat import LMFDBChat

app = FastAPI(title="Conductor")

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("CONDUCTOR_ALLOWED_ORIGINS", "*")
_allowed_origins = (
    [o.strip() for o in _raw_origins.split(",")]
    if _raw_origins != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Authentication ─────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    expected = os.getenv("CONDUCTOR_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API key not set.",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )


# ── Session store ─────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes of inactivity


class SessionEntry:
    def __init__(self):
        self.chat = LMFDBChat()
        self.last_access = time.time()

    def touch(self):
        self.last_access = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_access) > SESSION_TTL_SECONDS


_sessions: dict[str, SessionEntry] = {}


def _evict_expired_sessions():
    expired = [sid for sid, entry in _sessions.items() if entry.is_expired()]
    for sid in expired:
        del _sessions[sid]


def get_or_create_session(session_id: Optional[str]) -> tuple[str, LMFDBChat]:
    _evict_expired_sessions()
    if session_id and session_id in _sessions:
        entry = _sessions[session_id]
        entry.touch()
        return session_id, entry.chat
    new_id = str(uuid.uuid4())
    _sessions[new_id] = SessionEntry()
    return new_id, _sessions[new_id].chat


# ── Rate limiting ─────────────────────────────────────────────────────────────

RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW = 60  # seconds

_request_timestamps: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(session_id: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    _request_timestamps[session_id] = [
        t for t in _request_timestamps[session_id] if t > window_start
    ]
    if len(_request_timestamps[session_id]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: maximum {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW} seconds. Please wait before retrying."
            ),
        )
    _request_timestamps[session_id].append(now)


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    verbose: bool = False
    # Stateless history — passed from frontend on every request
    history: list[dict] = []
    last_sql: str | None = None
    last_tables: list[str] = []


class AnalysisRequest(BaseModel):
    instruction: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Public endpoint — no auth required. Used for deployment health checks."""
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
    }


@app.post("/chat")
def chat(
    req: ChatRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None),
    _: None = Depends(verify_api_key),
):
    sid, session = get_or_create_session(session_id)
    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    check_rate_limit(sid)
    # Restore stateless text state from frontend.
    # last_df is intentionally NOT restored here — it lives server-side only
    # so follow-up analysis always has access to the full DataFrame.
    if req.history:
        session.history = req.history
    if req.last_sql:
        session.last_sql = req.last_sql
    if req.last_tables:
        session.last_tables = req.last_tables
    return session.chat(req.message, verbose=req.verbose)


@app.post("/analysis")
def analysis(
    req: AnalysisRequest,
    session_id: Optional[str] = Cookie(None),
    _: None = Depends(verify_api_key),
):
    if not session_id or session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Submit a /chat query first.",
        )
    entry = _sessions[session_id]
    entry.touch()
    check_rate_limit(session_id)
    return entry.chat.run_analysis_turn(req.instruction)


@app.get("/session")
def get_session(
    session_id: Optional[str] = Cookie(None),
    _: None = Depends(verify_api_key),
):
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    _sessions[session_id].touch()
    return _sessions[session_id].chat.state()


@app.delete("/session")
def delete_session(
    session_id: Optional[str] = Cookie(None),
    _: None = Depends(verify_api_key),
):
    if session_id and session_id in _sessions:
        del _sessions[session_id]
        if session_id in _request_timestamps:
            del _request_timestamps[session_id]
    return {"status": "cleared"}
