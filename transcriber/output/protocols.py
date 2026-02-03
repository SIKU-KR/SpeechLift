"""Protocols for output components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class OutputWriter(Protocol):
    """Protocol for writing transcription output."""

    def write(
        self,
        video_path: Path,
        transcription: str,
        duration_seconds: float,
        cost_per_minute: float,
    ) -> Path:
        """
        Write transcription output to a file.

        Args:
            video_path: Path to the original video file.
            transcription: The transcribed text.
            duration_seconds: Duration of the transcribed audio.
            cost_per_minute: Cost per minute for transcription.

        Returns:
            Path to the output file.
        """
        ...
