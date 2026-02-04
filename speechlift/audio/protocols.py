"""Protocols for audio processing components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class AudioExtractor(Protocol):
    """Protocol for extracting audio from video files."""

    def extract(self, video_path: Path, output_dir: Path) -> Path:
        """
        Extract audio from a video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save the extracted audio.

        Returns:
            Path to the extracted audio file.

        Raises:
            RuntimeError: If extraction fails.
        """
        ...

    def is_available(self) -> bool:
        """Check if the audio extractor is available for use."""
        ...
