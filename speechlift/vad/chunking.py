"""Chunk merging logic for audio segments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydub import AudioSegment

from speechlift.vad.protocols import SpeechSegment

if TYPE_CHECKING:
    from speechlift.config.settings import TranscriptionSettings
    from speechlift.ui.protocols import UserInterface


class ChunkMerger:
    """Merges speech segments into chunks suitable for transcription."""

    SAMPLE_RATE = 16000

    def __init__(self, settings: TranscriptionSettings, ui: UserInterface) -> None:
        self._settings = settings
        self._ui = ui

    def merge(
        self,
        speech_segments: list[SpeechSegment],
        total_samples: int,
    ) -> list[tuple[float, float]]:
        """
        Merge speech segments into chunks of appropriate duration.

        Args:
            speech_segments: List of detected speech segments.
            total_samples: Total number of samples in the audio.

        Returns:
            List of (start_time, end_time) tuples in seconds.
        """
        if not speech_segments:
            return []

        min_dur = self._settings.min_chunk_duration
        max_dur = self._settings.max_chunk_duration

        chunks: list[tuple[float, float]] = []
        current_start = speech_segments[0].start_seconds
        current_end = speech_segments[0].end_seconds

        for segment in speech_segments[1:]:
            ts_start = segment.start_seconds
            ts_end = segment.end_seconds

            # Check if adding this segment keeps us under max duration
            potential_end = ts_end
            potential_duration = potential_end - current_start

            if potential_duration <= max_dur:
                # Extend current chunk
                current_end = ts_end
            else:
                # Save current chunk if it meets minimum duration
                if current_end - current_start >= min_dur:
                    chunks.append((current_start, current_end))
                elif chunks:
                    # Merge short chunk with previous
                    prev_start, _ = chunks.pop()
                    chunks.append((prev_start, current_end))
                else:
                    # First chunk is short, keep it anyway
                    chunks.append((current_start, current_end))

                # Start new chunk
                current_start = ts_start
                current_end = ts_end

        # Handle the last chunk
        if current_end - current_start >= min_dur:
            chunks.append((current_start, current_end))
        elif chunks:
            prev_start, _ = chunks.pop()
            chunks.append((prev_start, current_end))
        else:
            chunks.append((current_start, current_end))

        # Split any chunks that are too long
        final_chunks = self._split_long_chunks(chunks, max_dur)

        return final_chunks

    def _split_long_chunks(
        self,
        chunks: list[tuple[float, float]],
        max_duration: float,
    ) -> list[tuple[float, float]]:
        """Split chunks that exceed the maximum duration."""
        final_chunks: list[tuple[float, float]] = []

        for start, end in chunks:
            duration = end - start
            if duration > max_duration:
                # Split into ~equal parts
                num_parts = int(duration / max_duration) + 1
                part_duration = duration / num_parts
                for i in range(num_parts):
                    part_start = start + i * part_duration
                    part_end = start + (i + 1) * part_duration
                    final_chunks.append((part_start, min(part_end, end)))
            else:
                final_chunks.append((start, end))

        return final_chunks

    def create_chunk_files(
        self,
        audio_path: Path,
        chunk_ranges: list[tuple[float, float]],
        output_dir: Path,
    ) -> list[Path]:
        """
        Create audio chunk files from the given ranges.

        Args:
            audio_path: Path to the source audio file.
            chunk_ranges: List of (start, end) time ranges in seconds.
            output_dir: Directory to save chunk files.

        Returns:
            List of paths to the created chunk files.
        """
        if not chunk_ranges:
            self._ui.display_warning("No valid chunks to create.")
            return []

        # Load full audio with pydub for slicing
        audio = AudioSegment.from_wav(str(audio_path))

        # Create chunk files
        chunk_paths: list[Path] = []
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        for i, (start, end) in enumerate(chunk_ranges):
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            chunk = audio[start_ms:end_ms]

            chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
            chunk.export(str(chunk_path), format="wav")
            chunk_paths.append(chunk_path)

        self._ui.display_chunk_info(
            len(chunk_paths),
            self._settings.min_chunk_duration,
            self._settings.max_chunk_duration,
        )

        return chunk_paths

    @staticmethod
    def calculate_speech_duration(chunk_ranges: list[tuple[float, float]]) -> float:
        """Calculate the total speech duration from chunk ranges."""
        return sum(end - start for start, end in chunk_ranges)
