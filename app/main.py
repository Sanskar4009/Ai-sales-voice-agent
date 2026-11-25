"""FastAPI application entry point for the Autonomous AI Sales Voice Agent."""

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.models import Appointment, AvailableSlot, Lead
from app.schemas import (
    AppointmentResponse,
    AvailableSlotResponse,
    CallEndResponse,
    ChatRequest,
    ChatResponse,
    LeadResponse,
)
from app.services.agent_service import analyze_call_end, run_sales_agent, warm_groq_chat_model
from app.services.rag_service import init_rag
from app.services.voice_service import synthesize_speech, transcribe_audio, warm_whisper_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize database schema, seed initial slots, warm up RAG index and Groq models."""
    init_db()
    init_rag()
    warm_groq_chat_model()
    warm_whisper_model()
    yield


app = FastAPI(
    title="Fortivex Autonomous AI Sales Voice Agent API",
    description="Backend API for lead qualification, appointment scheduling, and voice sales workflows.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Serve the single-page dark-mode sales voice agent console."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Lightweight health check endpoint."""
    return {"status": "ok"}


@app.get("/api/leads", response_model=list[LeadResponse], tags=["Leads"])
def list_leads(db: Session = Depends(get_db)) -> list[Lead]:
    """Return all captured leads ordered from newest to oldest."""
    return db.query(Lead).order_by(Lead.created_at.desc(), Lead.id.desc()).all()


@app.get("/api/appointments", response_model=list[AppointmentResponse], tags=["Appointments"])
def list_appointments(db: Session = Depends(get_db)) -> list[Appointment]:
    """Return all scheduled appointments ordered by slot time."""
    return (
        db.query(Appointment)
        .order_by(Appointment.slot_datetime.asc(), Appointment.id.asc())
        .all()
    )


@app.get("/api/slots", response_model=list[AvailableSlotResponse], tags=["Slots"])
def list_slots(db: Session = Depends(get_db)) -> list[AvailableSlot]:
    """Return all consultation slots ordered chronologically."""
    return (
        db.query(AvailableSlot)
        .order_by(AvailableSlot.slot_datetime.asc(), AvailableSlot.id.asc())
        .all()
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Agent"])
def chat_with_agent(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Test endpoint to send a message to Forvi and inspect reply and tool events."""
    try:
        result = run_sales_agent(
            session_id=payload.session_id,
            user_text=payload.message,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(**result)


@app.post("/api/call/end/{session_id}", response_model=CallEndResponse, tags=["Agent"])
def end_call_and_analyze(
    session_id: str,
    db: Session = Depends(get_db),
) -> CallEndResponse:
    """Summarize the completed call session, update the lead record, and return analytics."""
    try:
        analytics = analyze_call_end(session_id=session_id, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return CallEndResponse(**analytics)


def _process_voice_turn(
    session_id: str,
    user_text: str,
) -> tuple[dict[str, object], bytes]:
    """Run the sales agent with a scoped DB session and synthesize TTS audio."""
    db_provider = app.dependency_overrides.get(get_db, get_db)
    db_gen = db_provider()
    db = next(db_gen)
    try:
        agent_result = run_sales_agent(session_id=session_id, user_text=user_text, db=db)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

    audio_out = synthesize_speech(agent_result["reply"])
    return agent_result, audio_out


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str) -> None:
    """Real-time voice and text WebSocket endpoint for Forvi."""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            user_text = ""
            if message.get("bytes") is not None:
                audio_bytes: bytes = message["bytes"]
                try:
                    user_text = await run_in_threadpool(transcribe_audio, audio_bytes)
                except Exception as exc:
                    await websocket.send_json({"error": f"Transcription failed: {exc}"})
                    continue
            elif message.get("text") is not None:
                raw_text: str = message["text"]
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON message."})
                    continue

                if not isinstance(payload, dict) or payload.get("type") != "text":
                    await websocket.send_json(
                        {"error": "Unsupported message format. Expected {'type': 'text', 'text': '...'}."}
                    )
                    continue
                user_text = str(payload.get("text") or "").strip()
            else:
                continue

            if not user_text:
                await websocket.send_json({"error": "Empty audio transcript or text message."})
                continue

            try:
                agent_result, audio_out = await run_in_threadpool(
                    _process_voice_turn, session_id, user_text
                )
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            audio_base64 = base64.b64encode(audio_out).decode("ascii") if audio_out else ""
            await websocket.send_json(
                {
                    "user_text": user_text,
                    "agent_text": agent_result["reply"],
                    "audio_base64": audio_base64,
                    "tool_events": agent_result["tool_events"],
                }
            )
    except WebSocketDisconnect:
        return
