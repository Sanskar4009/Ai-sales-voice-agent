"""SQLAlchemy ORM models for leads, appointments, and available slots."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Lead(Base):
    """Represents a qualified or in-progress sales lead captured during a voice session."""

    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), unique=True, index=True, nullable=False)
    name = Column(String(120), nullable=True)
    phone = Column(String(32), nullable=True)
    email = Column(String(255), nullable=True)
    company = Column(String(180), nullable=True)
    budget = Column(String(100), nullable=True)
    lead_score = Column(Integer, default=0, nullable=False)
    sentiment = Column(String(40), default="neutral", nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    appointments = relationship("Appointment", back_populates="lead")


class Appointment(Base):
    """Represents a booked discovery or product demo appointment."""

    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="SET NULL"), nullable=True, index=True)
    lead_name = Column(String(120), nullable=False)
    slot_datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    status = Column(String(40), default="scheduled", nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    lead = relationship("Lead", back_populates="appointments")


class AvailableSlot(Base):
    """Represents a bookable calendar slot for sales consultations."""

    __tablename__ = "available_slots"

    id = Column(Integer, primary_key=True, index=True)
    slot_datetime = Column(DateTime(timezone=True), unique=True, nullable=False, index=True)
    is_booked = Column(Boolean, default=False, nullable=False)
