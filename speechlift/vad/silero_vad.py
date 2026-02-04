"""Silero VAD implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from silero_vad import get_speech_timestamps, load_silero_vad

from speechlift.audio.utils import read_audio
from speechlift.vad.protocols import SpeechSegment

if TYPE_CHECKING:
    from speechlift.ui.protocols import ProgressReporter


class SileroVAD:
    """Voice Activity Detection using Silero VAD model."""

    SAMPLE_RATE = 16000
    THRESHOLD = 0.5
    MIN_SPEECH_DURATION_MS = 250
    MIN_SILENCE_DURATION_MS = 100

    def __init__(self, progress_reporter: ProgressReporter) -> None:
        self._progress = progress_reporter
        self._model = None

    def detect_speech(self, audio_path: Path) -> list[SpeechSegment]:
        """
        Detect speech segments in an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of detected speech segments.
        """
        with self._progress.spinner("Analyzing voice activity..."):
            if self._model is None:
                self._model = load_silero_vad()

            wav = read_audio(str(audio_path))

        speech_timestamps = get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=self.THRESHOLD,
            min_speech_duration_ms=self.MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=self.MIN_SILENCE_DURATION_MS,
        )

        return [
            SpeechSegment(start_sample=ts["start"], end_sample=ts["end"])
            for ts in speech_timestamps
        ]

    def get_total_duration(self, audio_path: Path) -> float:
        """
        Get the total duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        wav = read_audio(str(audio_path))
        return len(wav) / self.SAMPLE_RATE
