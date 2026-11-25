"""Groq-powered AI Sales Agent ('Forvi') with RAG context injection and tool calling."""

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy.orm import Session

from app.models import Appointment, AvailableSlot, Lead
from app.services.rag_service import retrieve_context

load_dotenv()

DEFAULT_FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_MEMORY_MESSAGES = 30
MAX_TOOL_ROUNDS = 4
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))

PREFERRED_CHAT_MODEL_KEYWORDS = ("llama-4", "qwen", "gpt-oss", "gemma")
EXCLUDED_CHAT_MODEL_KEYWORDS = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard")

# Per-session conversation history: dict[session_id, list[messages]]
session_memory: dict[str, list[dict[str, Any]]] = {}
_memory_lock = threading.Lock()
_model_lock = threading.Lock()
_cached_chat_model: Optional[str] = None


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "upsert_lead",
            "description": (
                "Create or update the prospect's lead profile in the CRM database as you "
                "discover their name, phone, email, company, budget/plan interest, and lead score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The active conversation session ID.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Prospect's full name.",
                    },
                    "phone": {
                        "type": "string",
                        "description": "Prospect's phone number.",
                    },
                    "email": {
                        "type": "string",
                        "description": "Prospect's work email address.",
                    },
                    "company": {
                        "type": "string",
                        "description": "Prospect's company or organization name.",
                    },
                    "budget": {
                        "type": "string",
                        "description": "Stated budget or Fortivex plan interest (e.g., Starter, Growth, Enterprise, or amount).",
                    },
                    "lead_score": {
                        "type": "integer",
                        "description": "Estimated qualification score from 0 to 100 based on fit, budget, and buying intent.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book an available Fortivex demo/discovery slot for the prospect. "
                "Marks the slot as booked and records the appointment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The active conversation session ID.",
                    },
                    "lead_name": {
                        "type": "string",
                        "description": "Full name of the prospect booking the demo.",
                    },
                    "slot_datetime": {
                        "type": "string",
                        "description": "Exact ISO-8601 slot_datetime string from the available slots list.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Brief summary of prospect requirements, plan interest, or questions.",
                    },
                },
                "required": ["session_id", "lead_name", "slot_datetime"],
            },
        },
    },
]


def _normalize_utc_naive(dt: datetime) -> datetime:
    """Normalize a datetime to naive UTC for consistent SQLite comparisons."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string into naive UTC, returning None if invalid."""
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    return _normalize_utc_naive(parsed)


def _format_available_slots(db: Session) -> str:
    """Format currently unbooked slots for injection into the system prompt."""
    slots = (
        db.query(AvailableSlot)
        .filter(AvailableSlot.is_booked.is_(False))
        .order_by(AvailableSlot.slot_datetime.asc())
        .all()
    )
    if not slots:
        return "No open demo slots are currently available."

    lines: list[str] = []
    for slot in slots:
        utc_dt = _normalize_utc_naive(slot.slot_datetime).replace(tzinfo=timezone.utc)
        ist_dt = utc_dt.astimezone(IST_TIMEZONE)
        lines.append(
            f"- slot_datetime=\"{_normalize_utc_naive(slot.slot_datetime).isoformat()}\" "
            f"({ist_dt.strftime('%a, %d %b %Y at %I:%M %p IST')})"
        )
    return "\n".join(lines)


def _build_system_prompt(session_id: str, context: str, available_slots_text: str) -> str:
    """Construct the dynamic system prompt with persona rules, RAG context, and open slots."""
    kb_section = context.strip() if context.strip() else "No matching product details found."
    return (
        "You are 'Forvi', a sharp, polite AI Sales Consultant for Fortivex. Introduce and refer to yourself as Forvi.\n"
        "Conversation rules:\n"
        "- Speak concisely: maximum 2-3 sentences per turn, natural and optimized for voice.\n"
        "- Automatically match the user's language (English, Hindi, or Hinglish).\n"
        "- Ground every product, pricing, security, onboarding, and trial claim strictly in the Retrieved Knowledge Base Context below. Never invent pricing or features outside the retrieved context.\n"
        "- Naturally qualify the lead during the conversation by asking for their Name, Contact/Email, and Requirement/Budget, and offer to book a demo slot.\n"
        "- Whenever you learn or update lead details (name, phone, email, company, budget), call the `upsert_lead` tool immediately.\n"
        "- When the user confirms a demo slot, call `book_appointment` using the exact `slot_datetime` value from Available Demo Slots.\n"
        f"- Active session_id: {session_id}\n\n"
        "Retrieved Knowledge Base Context:\n"
        f"{kb_section}\n\n"
        "Available Demo Slots:\n"
        f"{available_slots_text}"
    )


def upsert_lead(
    db: Session,
    session_id: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    company: Optional[str] = None,
    budget: Optional[str] = None,
    lead_score: Optional[int] = None,
) -> dict[str, Any]:
    """Save or update a lead record in SQLite for the given session_id."""
    lead = db.query(Lead).filter(Lead.session_id == session_id).first()
    action = "updated"
    if lead is None:
        lead = Lead(session_id=session_id)
        db.add(lead)
        action = "created"

    if name and name.strip():
        lead.name = name.strip()
    if phone and phone.strip():
        lead.phone = phone.strip()
    if email and email.strip():
        lead.email = email.strip()
    if company and company.strip():
        lead.company = company.strip()
    if budget and budget.strip():
        lead.budget = budget.strip()
    if lead_score is not None:
        try:
            score_int = int(lead_score)
            lead.lead_score = max(0, min(100, score_int))
        except (TypeError, ValueError):
            pass

    db.commit()
    db.refresh(lead)

    return {
        "status": action,
        "lead_id": lead.id,
        "session_id": lead.session_id,
        "name": lead.name,
        "phone": lead.phone,
        "email": lead.email,
        "company": lead.company,
        "budget": lead.budget,
        "lead_score": lead.lead_score,
    }


def book_appointment(
    db: Session,
    session_id: str,
    lead_name: str,
    slot_datetime: str,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Book an available consultation slot and create an Appointment record."""
    target_dt = _parse_iso_datetime(slot_datetime)
    if target_dt is None:
        return {
            "status": "error",
            "message": f"Invalid slot_datetime format: '{slot_datetime}'.",
            "available_slots": _format_available_slots(db),
        }

    all_slots = db.query(AvailableSlot).order_by(AvailableSlot.slot_datetime.asc()).all()
    matched_slot: Optional[AvailableSlot] = None
    for slot in all_slots:
        if _normalize_utc_naive(slot.slot_datetime) == target_dt:
            matched_slot = slot
            break

    if matched_slot is None:
        return {
            "status": "error",
            "message": "Requested slot does not exist.",
            "available_slots": _format_available_slots(db),
        }

    if matched_slot.is_booked:
        return {
            "status": "error",
            "message": "That slot is already booked. Please choose another available slot.",
            "available_slots": _format_available_slots(db),
        }

    lead = db.query(Lead).filter(Lead.session_id == session_id).first()
    resolved_name = (lead_name or "").strip() or (lead.name if lead and lead.name else "Prospect")

    if lead is not None and not lead.name and resolved_name:
        lead.name = resolved_name

    matched_slot.is_booked = True
    appointment = Appointment(
        lead_id=lead.id if lead else None,
        lead_name=resolved_name,
        slot_datetime=matched_slot.slot_datetime,
        status="scheduled",
        notes=notes.strip() if notes and notes.strip() else None,
    )
    db.add(appointment)
    db.commit()
    db.refresh(appointment)

    return {
        "status": "booked",
        "appointment_id": appointment.id,
        "lead_id": appointment.lead_id,
        "lead_name": appointment.lead_name,
        "slot_datetime": _normalize_utc_naive(appointment.slot_datetime).isoformat(),
        "appointment_status": appointment.status,
        "notes": appointment.notes,
    }


def _execute_tool(
    db: Session,
    session_id: str,
    tool_name: str,
    raw_arguments: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse tool arguments, execute the requested DB tool, and return (parsed_args, result)."""
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(args, dict):
            args = {}
    except json.JSONDecodeError:
        args = {}

    if tool_name == "upsert_lead":
        result = upsert_lead(
            db=db,
            session_id=session_id,
            name=args.get("name"),
            phone=args.get("phone"),
            email=args.get("email"),
            company=args.get("company"),
            budget=args.get("budget"),
            lead_score=args.get("lead_score"),
        )
        return args, result

    if tool_name == "book_appointment":
        result = book_appointment(
            db=db,
            session_id=session_id,
            lead_name=args.get("lead_name", ""),
            slot_datetime=args.get("slot_datetime", ""),
            notes=args.get("notes"),
        )
        return args, result

    return args, {"status": "error", "message": f"Unknown tool: {tool_name}"}


def _get_groq_client() -> Groq:
    """Create a Groq client using GROQ_API_KEY from the environment."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set a valid key in .env before calling /api/chat."
        )
    return Groq(api_key=api_key)


def _select_active_chat_model(model_ids: list[str]) -> str:
    """Pick the best active chat model from a list of Groq model IDs."""
    candidates = [
        mid
        for mid in model_ids
        if mid and not any(ex in mid.lower() for ex in EXCLUDED_CHAT_MODEL_KEYWORDS)
    ]
    for keyword in PREFERRED_CHAT_MODEL_KEYWORDS:
        for mid in candidates:
            if keyword in mid.lower():
                return mid
    if candidates:
        return candidates[0]
    return DEFAULT_FALLBACK_MODEL


def resolve_groq_chat_model(client: Groq) -> str:
    """Resolve and cache an active Groq chat model, honoring GROQ_MODEL if set."""
    global _cached_chat_model

    env_model = os.getenv("GROQ_MODEL", "").strip()
    if env_model:
        return env_model

    if _cached_chat_model:
        return _cached_chat_model

    with _model_lock:
        if _cached_chat_model:
            return _cached_chat_model

        try:
            models_page = client.models.list()
            active_ids = [
                m.id
                for m in (getattr(models_page, "data", None) or [])
                if getattr(m, "id", None) and getattr(m, "active", True) is not False
            ]
            _cached_chat_model = _select_active_chat_model(active_ids)
        except Exception:
            _cached_chat_model = DEFAULT_FALLBACK_MODEL

        return _cached_chat_model


def reset_cached_chat_model() -> None:
    """Clear the cached Groq chat model selection (useful for tests)."""
    global _cached_chat_model
    with _model_lock:
        _cached_chat_model = None


def warm_groq_chat_model() -> None:
    """Pre-resolve the active Groq chat model on startup if GROQ_API_KEY is configured."""
    try:
        resolve_groq_chat_model(_get_groq_client())
    except Exception:
        pass


def _get_session_history(session_id: str) -> list[dict[str, Any]]:
    """Return a copy of the conversation history for a session."""
    with _memory_lock:
        return list(session_memory.get(session_id, []))


def _save_session_history(session_id: str, history: list[dict[str, Any]]) -> None:
    """Persist trimmed conversation history for a session."""
    trimmed = history[-MAX_MEMORY_MESSAGES:]
    with _memory_lock:
        session_memory[session_id] = trimmed


def clear_session_memory(session_id: str) -> None:
    """Remove conversation history for a given session_id."""
    with _memory_lock:
        session_memory.pop(session_id, None)


def run_sales_agent(
    session_id: str,
    user_text: str,
    db: Session,
) -> dict[str, Any]:
    """Process one user turn through RAG retrieval, Groq LLM, and tool execution."""
    cleaned_session_id = session_id.strip()
    cleaned_user_text = user_text.strip()
    if not cleaned_session_id:
        raise ValueError("session_id must not be empty.")
    if not cleaned_user_text:
        raise ValueError("message must not be empty.")

    client = _get_groq_client()
    chat_model = resolve_groq_chat_model(client)
    history = _get_session_history(cleaned_session_id)
    history.append({"role": "user", "content": cleaned_user_text})

    tool_events: list[dict[str, Any]] = []
    final_reply = ""

    for _ in range(MAX_TOOL_ROUNDS):
        # Fetch fresh RAG context and open demo slots before every LLM call
        rag_context = retrieve_context(cleaned_user_text, top_k=3)
        slots_text = _format_available_slots(db)
        system_prompt = _build_system_prompt(
            session_id=cleaned_session_id,
            context=rag_context,
            available_slots_text=slots_text,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history,
        ]

        completion = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=300,
        )

        assistant_msg = completion.choices[0].message
        tool_calls = assistant_msg.tool_calls or []

        if not tool_calls:
            final_reply = (assistant_msg.content or "").strip()
            history.append({"role": "assistant", "content": final_reply})
            break

        serialized_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        history.append(
            {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": serialized_tool_calls,
            }
        )

        for tc in tool_calls:
            parsed_args, tool_result = _execute_tool(
                db=db,
                session_id=cleaned_session_id,
                tool_name=tc.function.name,
                raw_arguments=tc.function.arguments,
            )
            tool_events.append(
                {
                    "tool": tc.function.name,
                    "arguments": parsed_args,
                    "result": tool_result,
                }
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(tool_result),
                }
            )
    else:
        final_reply = (
            "I've updated your details. Could you please repeat or confirm your last request?"
        )
        history.append({"role": "assistant", "content": final_reply})

    _save_session_history(cleaned_session_id, history)
    return {"reply": final_reply, "tool_events": tool_events}


_SCORE_BUCKET_TO_INT = {
    "Hot": 90,
    "Warm": 60,
    "Cold": 25,
}


def _normalize_category(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """Normalize LLM enum-like string output to one of the allowed canonical labels."""
    if not isinstance(value, str):
        return default
    lowered = value.strip().lower()
    for option in allowed:
        if lowered == option.lower():
            return option
    return default


def analyze_call_end(session_id: str, db: Session) -> dict[str, Any]:
    """Summarize a completed call session with Groq JSON mode and update the lead record."""
    cleaned_session_id = session_id.strip()
    if not cleaned_session_id:
        raise ValueError("session_id must not be empty.")

    history = _get_session_history(cleaned_session_id)
    transcript_lines: list[str] = []
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            speaker = "Prospect" if role == "user" else "Forvi"
            transcript_lines.append(f"{speaker}: {content}")

    if not transcript_lines:
        analytics: dict[str, Any] = {
            "summary_bullets": ["No conversation transcript was recorded for this session."],
            "lead_score": "Cold",
            "sentiment": "Neutral",
            "key_objections": "None",
            "next_action": "Reach out to reschedule a discovery call.",
        }
    else:
        client = _get_groq_client()
        chat_model = resolve_groq_chat_model(client)
        transcript_text = "\n".join(transcript_lines)
        prompt = (
            "Analyze the following B2B sales call transcript for Fortivex AI Suite and return "
            "a strict JSON object with these exact keys:\n"
            '- "summary_bullets": array of 2 to 4 concise bullet point strings summarizing the call\n'
            '- "lead_score": exactly one of "Hot", "Warm", or "Cold"\n'
            '- "sentiment": exactly one of "Positive", "Neutral", or "Negative"\n'
            '- "key_objections": concise string describing pricing, security, or timeline objections (or "None")\n'
            '- "next_action": concise recommended follow-up action for the sales team\n\n'
            f"Transcript:\n{transcript_text}"
        )

        completion = client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a B2B RevOps analyst. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=400,
        )

        raw_json = (completion.choices[0].message.content or "{}").strip()
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            parsed = {}

        raw_bullets = parsed.get("summary_bullets")
        if isinstance(raw_bullets, list):
            summary_bullets = [str(item).strip() for item in raw_bullets if str(item).strip()]
        else:
            summary_bullets = []
        if not summary_bullets:
            summary_bullets = ["Discovery call completed with prospect."]

        lead_score_label = _normalize_category(
            parsed.get("lead_score"), ("Hot", "Warm", "Cold"), "Warm"
        )
        sentiment_label = _normalize_category(
            parsed.get("sentiment"), ("Positive", "Neutral", "Negative"), "Neutral"
        )
        key_objections = str(parsed.get("key_objections") or "None").strip() or "None"
        next_action = (
            str(parsed.get("next_action") or "Send follow-up email with product details.").strip()
            or "Send follow-up email with product details."
        )

        analytics = {
            "summary_bullets": summary_bullets,
            "lead_score": lead_score_label,
            "sentiment": sentiment_label,
            "key_objections": key_objections,
            "next_action": next_action,
        }

    lead = db.query(Lead).filter(Lead.session_id == cleaned_session_id).first()
    if lead is not None:
        lead.sentiment = analytics["sentiment"]
        lead.lead_score = _SCORE_BUCKET_TO_INT[analytics["lead_score"]]
        formatted_bullets = "\n".join(f"- {bullet}" for bullet in analytics["summary_bullets"])
        lead.summary = (
            f"{formatted_bullets}\n"
            f"Key Objections: {analytics['key_objections']}\n"
            f"Next Action: {analytics['next_action']}"
        )
        db.commit()
        db.refresh(lead)

    return analytics
