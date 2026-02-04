"""Protocols for transcription components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Callable, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for transcribing audio to text."""

    async def transcribe_chunk(self, audio_path: Path) -> str:
        """
        Transcribe a single audio chunk.

        Args:
            audio_path: Path to the audio chunk file.

        Returns:
            Transcribed text.
        """
        ...

    async def transcribe_all(
        self,
        chunk_paths: list[Path],
        progress_callback: Callable[[], None] | None = None,
    ) -> str:
        """
        Transcribe all audio chunks in parallel.

        Args:
            chunk_paths: List of paths to audio chunk files.
            progress_callback: Optional callback to report progress.

        Returns:
            Combined transcription text.
        """
        ...
