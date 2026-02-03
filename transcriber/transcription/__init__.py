"""Transcription module for video transcriber."""

from .protocols import Transcriber
from .whisper_api import WhisperAPITranscriber

__all__ = ["Transcriber", "WhisperAPITranscriber"]
