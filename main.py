"""
Description: This is the FastAPI application entry point. 

NOTE: CORS is open for now; tighten allow_origins before public release.
"""

from fastapi import FastAPI, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from pipeline.chat import LMFDBChat

app = FastAPI(title="Conductor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before public release
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: {session_id: LMFDBChat}
sessions: dict[str, LMFDBChat] = {}


def get_or_create_session(session_id: Optional[str]) -> tuple[str, LMFDBChat]:
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = str(uuid.uuid4())
    sessions[new_id] = LMFDBChat()
    return new_id, sessions[new_id]


class ChatRequest(BaseModel):
    message: str
    verbose: bool = False


class AnalysisRequest(BaseModel):
    instruction: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest, response: Response, session_id: Optional[str] = Cookie(None)):
    sid, session = get_or_create_session(session_id)
    response.set_cookie("session_id", sid)
    result = session.chat(req.message, verbose=req.verbose)
    return result


@app.post("/analysis")
def analysis(req: AnalysisRequest, session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="No active session.")
    session = sessions[session_id]
    result = session.run_analysis_turn(req.instruction)
    return result


@app.get("/session")
def get_session(session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    return sessions[session_id].state()


@app.delete("/session")
def delete_session(session_id: Optional[str] = Cookie(None)):
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}
