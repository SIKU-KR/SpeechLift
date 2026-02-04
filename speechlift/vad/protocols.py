"""Protocols for Voice Activity Detection components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class SpeechSegment:
    """A segment of detected speech."""

    start_sample: int
    end_sample: int

    @property
    def start_seconds(self) -> float:
        """Start time in seconds (assuming 16kHz sample rate)."""
        return self.start_sample / 16000

    @property
    def end_seconds(self) -> float:
        """End time in seconds (assuming 16kHz sample rate)."""
        return self.end_sample / 16000

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.end_seconds - self.start_seconds


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Protocol for detecting voice activity in audio."""

    def detect_speech(self, audio_path: Path) -> list[SpeechSegment]:
        """
        Detect speech segments in an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of detected speech segments.
        """
        ...
