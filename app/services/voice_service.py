"""Voice processing service for Groq Whisper STT and Deepgram Aura TTS."""

import os
import threading
from typing import Optional

import httpx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"
PREFERRED_WHISPER_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"

_whisper_lock = threading.Lock()
_cached_whisper_model: Optional[str] = None


def _get_groq_client() -> Groq:
    """Create a Groq client using GROQ_API_KEY from the environment."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set a valid key in .env to transcribe audio."
        )
    return Groq(api_key=api_key)


def _select_active_whisper_model(model_ids: list[str]) -> str:
    """Pick the best active Whisper transcription model from a list of Groq model IDs."""
    whisper_ids = [mid for mid in model_ids if mid and "whisper" in mid.lower()]
    for preferred in PREFERRED_WHISPER_MODELS:
        for mid in whisper_ids:
            if mid.lower() == preferred:
                return mid
    for preferred in PREFERRED_WHISPER_MODELS:
        for mid in whisper_ids:
            if preferred in mid.lower():
                return mid
    if whisper_ids:
        return whisper_ids[0]
    return DEFAULT_WHISPER_MODEL


def resolve_whisper_model(client: Groq) -> str:
    """Resolve and cache an active Groq Whisper model from client.models.list()."""
    global _cached_whisper_model

    env_model = os.getenv("GROQ_WHISPER_MODEL", "").strip()
    if env_model:
        return env_model

    if _cached_whisper_model:
        return _cached_whisper_model

    with _whisper_lock:
        if _cached_whisper_model:
            return _cached_whisper_model

        try:
            models_page = client.models.list()
            active_ids = [
                m.id
                for m in (getattr(models_page, "data", None) or [])
                if getattr(m, "id", None) and getattr(m, "active", True) is not False
            ]
            _cached_whisper_model = _select_active_whisper_model(active_ids)
        except Exception:
            _cached_whisper_model = DEFAULT_WHISPER_MODEL

        return _cached_whisper_model


def reset_cached_whisper_model() -> None:
    """Clear the cached Whisper model selection (useful for tests)."""
    global _cached_whisper_model
    with _whisper_lock:
        _cached_whisper_model = None


def warm_whisper_model() -> None:
    """Pre-resolve the Whisper model on startup if GROQ_API_KEY is configured."""
    try:
        resolve_whisper_model(_get_groq_client())
    except Exception:
        pass


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe raw audio bytes via Groq Whisper and return clean transcript text."""
    if not audio_bytes:
        return ""

    client = _get_groq_client()
    whisper_model = resolve_whisper_model(client)
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=whisper_model,
        response_format="text",
    )

    if isinstance(transcription, str):
        return transcription.strip()

    text = getattr(transcription, "text", "")
    return str(text).strip()


def synthesize_speech(text: str) -> bytes:
    """Synthesize speech via Deepgram Aura TTS REST API and return MP3 bytes.

    Returns empty bytes (b"") if DEEPGRAM_API_KEY is unset or Deepgram fails,
    allowing the frontend to fall back gracefully to browser SpeechSynthesis.
    """
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return b""

    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key or api_key == "your_deepgram_api_key_here":
        return b""

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": cleaned_text}

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(DEEPGRAM_TTS_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.content
    except Exception:
        return b""
