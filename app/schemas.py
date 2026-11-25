"""Pydantic response schemas for the Fortivex Sales Voice Agent API."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LeadResponse(BaseModel):
    """Serialized representation of a sales lead."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    budget: Optional[str] = None
    lead_score: int
    sentiment: str
    summary: Optional[str] = None
    created_at: datetime


class AppointmentResponse(BaseModel):
    """Serialized representation of a booked appointment."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    lead_id: Optional[int] = None
    lead_name: str
    slot_datetime: datetime
    status: str
    notes: Optional[str] = None
    created_at: datetime


class AvailableSlotResponse(BaseModel):
    """Serialized representation of an available consultation slot."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    slot_datetime: datetime
    is_booked: bool


class ChatRequest(BaseModel):
    """Request payload for the text chat test endpoint."""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)


class ToolEvent(BaseModel):
    """Record of a function call executed during an agent turn."""

    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    """Response payload returned by the AI sales agent chat endpoint."""

    reply: str
    tool_events: list[ToolEvent] = Field(default_factory=list)


class CallEndResponse(BaseModel):
    """Post-call analytics extracted from a completed sales conversation."""

    summary_bullets: list[str]
    lead_score: Literal["Hot", "Warm", "Cold"]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    key_objections: str
    next_action: str
