"""OpenAI Whisper API transcription implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from speechlift.config.settings import TranscriptionSettings
    from speechlift.ui.protocols import ProgressReporter, UserInterface


class WhisperAPITranscriber:
    """Transcribes audio using OpenAI's Whisper API."""

    def __init__(
        self,
        settings: TranscriptionSettings,
        progress_reporter: ProgressReporter,
        ui: UserInterface,
    ) -> None:
        self._settings = settings
        self._progress = progress_reporter
        self._ui = ui
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client (lazy initialization)."""
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def transcribe_chunk(self, audio_path: Path) -> str:
        """
        Transcribe a single audio chunk.

        Args:
            audio_path: Path to the audio chunk file.

        Returns:
            Transcribed text.
        """
        client = self._get_client()
        with open(audio_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return response.text

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
        if not chunk_paths:
            return ""

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)

        tasks = [
            self._transcribe_with_retry(i, path, semaphore, progress_callback)
            for i, path in enumerate(chunk_paths)
        ]

        results = await asyncio.gather(*tasks)

        # Sort by index and join texts
        sorted_results = sorted(results, key=lambda x: x[0])
        transcription = " ".join(text.strip() for _, text in sorted_results if text.strip())

        return transcription

    async def _transcribe_with_retry(
        self,
        index: int,
        path: Path,
        semaphore: asyncio.Semaphore,
        progress_callback: Callable[[], None] | None,
    ) -> tuple[int, str]:
        """Transcribe a single chunk with retry logic."""
        async with semaphore:
            last_error: Exception | None = None

            for attempt in range(self._settings.max_retries):
                try:
                    text = await self.transcribe_chunk(path)
                    if progress_callback:
                        progress_callback()
                    return (index, text)

                except Exception as e:
                    last_error = e
                    if attempt < self._settings.max_retries - 1:
                        # Exponential backoff
                        wait_time = (2**attempt) + (0.1 * attempt)
                        await asyncio.sleep(wait_time)

            # All retries failed
            self._ui.display_warning(
                f"Failed to transcribe chunk {index} after {self._settings.max_retries} attempts: {last_error}"
            )
            if progress_callback:
                progress_callback()
            return (index, "")
