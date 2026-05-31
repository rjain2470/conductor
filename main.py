"""
Description: This is the FastAPI application entry point. 

NOTE: CORS is open for now; tighten allow_origins before public release.
"""

import time
import uuid
from collections import defaultdict
from typing import Optional
 
from fastapi import FastAPI, HTTPException, Cookie, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from pipeline.chat import LMFDBChat
 
app = FastAPI(title="Conductor")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before public release
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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
    """Remove sessions that have been inactive for SESSION_TTL_SECONDS."""
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
 
RATE_LIMIT_REQUESTS = 20   # max requests
RATE_LIMIT_WINDOW = 60     # per this many seconds
 
# {session_id: [timestamp, timestamp, ...]}
_request_timestamps: dict[str, list[float]] = defaultdict(list)
 
 
def check_rate_limit(session_id: str):
    """
    Sliding window rate limiter.
    Raises HTTP 429 if the session has exceeded RATE_LIMIT_REQUESTS
    in the last RATE_LIMIT_WINDOW seconds.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    timestamps = _request_timestamps[session_id]
 
    # Drop timestamps outside the window
    _request_timestamps[session_id] = [t for t in timestamps if t > window_start]
 
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
 
 
class AnalysisRequest(BaseModel):
    instruction: str
 
 
# ── Endpoints ─────────────────────────────────────────────────────────────────
 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
    }
 
 
@app.post("/chat")
def chat(
    req: ChatRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None),
):
    sid, session = get_or_create_session(session_id)
    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    check_rate_limit(sid)
 
    result = session.chat(req.message, verbose=req.verbose)
    return result
 
 
@app.post("/analysis")
def analysis(
    req: AnalysisRequest,
    session_id: Optional[str] = Cookie(None),
):
    if not session_id or session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Submit a /chat query first.",
        )
    entry = _sessions[session_id]
    entry.touch()
    check_rate_limit(session_id)
 
    result = entry.chat.run_analysis_turn(req.instruction)
    return result
 
 
@app.get("/session")
def get_session(session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="No active session.",
        )
    _sessions[session_id].touch()
    return _sessions[session_id].chat.state()
 
 
@app.delete("/session")
def delete_session(session_id: Optional[str] = Cookie(None)):
    if session_id and session_id in _sessions:
        del _sessions[session_id]
        if session_id in _request_timestamps:
            del _request_timestamps[session_id]
    return {"status": "cleared"}
