"""Database configuration, session management, and startup seeding."""

import os
from datetime import datetime, timedelta, timezone
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sales_agent.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy database session and close it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_available_slots(db: Session) -> None:
    """Seed 6 upcoming demo/discovery call slots if available_slots is empty."""
    from app.models import AvailableSlot

    if db.query(AvailableSlot).count() > 0:
        return

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    ist_offset = timezone(timedelta(hours=5, minutes=30))
    now_ist = now.astimezone(ist_offset)

    # Generate 6 upcoming business-hour slots (11:00 AM and 4:00 PM IST over next 3 days)
    slot_hours_ist = (11, 16)
    upcoming_slots: list[AvailableSlot] = []
    day_offset = 1

    while len(upcoming_slots) < 6:
        candidate_day = (now_ist + timedelta(days=day_offset)).replace(
            minute=0, second=0, microsecond=0
        )
        # Skip Sundays (weekday() == 6) to keep slots realistic for B2B sales calls
        if candidate_day.weekday() != 6:
            for hour in slot_hours_ist:
                slot_dt_utc = candidate_day.replace(hour=hour).astimezone(timezone.utc)
                upcoming_slots.append(
                    AvailableSlot(slot_datetime=slot_dt_utc, is_booked=False)
                )
        day_offset += 1

    db.add_all(upcoming_slots)
    db.commit()


def init_db() -> None:
    """Create database tables and run initial data seeding."""
    # Ensure all ORM models are registered on Base.metadata before create_all
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        seed_available_slots(db)
    finally:
        db.close()
