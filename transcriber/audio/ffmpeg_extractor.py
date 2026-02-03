"""FFmpeg-based audio extractor implementation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriber.ui.protocols import ProgressReporter, UserInterface


class FFmpegAudioExtractor:
    """Extracts audio from video files using FFmpeg."""

    def __init__(
        self,
        progress_reporter: ProgressReporter,
        ui: UserInterface,
    ) -> None:
        self._progress = progress_reporter
        self._ui = ui

    def extract(self, video_path: Path, output_dir: Path) -> Path:
        """
        Extract audio from a video file as 16kHz mono WAV.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save the extracted audio.

        Returns:
            Path to the extracted audio file.

        Raises:
            RuntimeError: If extraction fails.
        """
        output_path = output_dir / "audio.wav"

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            str(output_path),
        ]

        with self._progress.spinner(f"Extracting audio from {video_path.name}..."):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self._ui.display_error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("Failed to extract audio")

        self._ui.display_success("Audio extracted successfully")
        return output_path

    def is_available(self) -> bool:
        """Check if FFmpeg is available in PATH."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def display_install_instructions(self) -> None:
        """Display installation instructions for FFmpeg."""
        self._ui.display_error("ffmpeg not found in PATH")
        self._ui.display_info("Install with:")
        self._ui.display_info("  macOS:   brew install ffmpeg")
        self._ui.display_info("  Ubuntu:  sudo apt install ffmpeg")
        self._ui.display_info("  Windows: choco install ffmpeg")
