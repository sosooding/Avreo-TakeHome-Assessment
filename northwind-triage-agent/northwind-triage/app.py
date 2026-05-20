"""
FastAPI backend for the Northwind Triage Agent.
Powered by Gemini 3.1 Flash Lite.

Endpoints:
  POST /triage      — Triage a single message, returns structured JSON decision.
  POST /evaluate    — Run batch evaluation across all 20 benchmark messages.
  GET  /messages    — Return the 20 inbound messages (for the UI test panel).
  GET  /benchmark   — Return benchmark decisions (for the UI evaluation view).
  GET  /            — Serve the frontend UI.

Start with: python app.py   (or: uvicorn app:app --reload)
"""

import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Force stdout/stderr to UTF-8. On Windows, console default is cp1252, which
# garbles message bodies that contain ² ° — and other non-ASCII chars when we
# print them for debugging. This is a no-op on Linux/macOS.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

# Load .env before importing agent (which reads env vars at init)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from agent import TriageAgent
from evaluate import load_benchmark, load_messages, run_evaluation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

_agent: Optional[TriageAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    print("[startup] Initialising Northwind Triage Agent (Gemini)...")
    try:
        _agent = TriageAgent()
        print("[startup] Agent ready.")
    except Exception as e:
        print(f"[startup] WARNING: Agent failed to initialise: {e}")
        print("[startup] Make sure GEMINI_API_KEY is set in your .env file.")
    yield


app = FastAPI(
    title="Northwind Triage Agent",
    description="AI-powered customer message triage for Northwind Home Services",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class MessageRequest(BaseModel):
    id: str = "MSG-CUSTOM"
    channel: str = "webform"
    received_at: str = ""
    sender_name: str = ""
    sender_address: str = ""
    subject: Optional[str] = None
    body: str


class TriageResponse(BaseModel):
    category: str
    priority: str
    route_to: str
    draft_reply: str
    needs_human_review: bool
    reasoning: str


# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────

@app.post("/triage", response_model=TriageResponse, summary="Triage a single message")
async def triage(req: MessageRequest):
    """
    Accepts a raw customer message and returns a structured triage decision.
    All fields are required in the response:
      category, priority, route_to, draft_reply, needs_human_review, reasoning
    """
    if _agent is None:
        raise HTTPException(503, detail="Agent not ready — check GEMINI_API_KEY in .env")
    try:
        result = _agent.triage(req.model_dump())
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/evaluate", summary="Run batch evaluation across all 20 benchmark messages")
async def evaluate():
    """
    Runs the agent against all 20 messages in 05_Inbound_Messages.json,
    scores against 06_Benchmark.json, and returns the full evaluation report.

    Note: This takes ~1.5 minutes on the Gemini free tier (15 RPM = one call
    every ~4.5 seconds). Set GEMINI_REQUEST_DELAY_S
    in your .env to tune this — lower for paid tiers, 0 to send back-to-back
    and rely on the agent's built-in retry/backoff.
    """
    if _agent is None:
        raise HTTPException(503, detail="Agent not ready — check GEMINI_API_KEY in .env")
    try:
        report = run_evaluation(_agent)
        return JSONResponse(report)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/messages", summary="Get the 20 benchmark inbound messages")
async def get_messages():
    return JSONResponse(load_messages())


@app.get("/benchmark", summary="Get the gold-standard benchmark decisions")
async def get_benchmark():
    bm = load_benchmark()
    return JSONResponse(list(bm.values()))


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "agent_ready": _agent is not None}


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"\n🟢 Northwind Triage Agent starting on http://localhost:{port}")
    print("   Press Ctrl+C to stop.\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
