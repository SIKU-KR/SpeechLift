"""Text file output writer implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriber.ui.protocols import UserInterface


class TextFileWriter:
    """Writes transcription output to text files."""

    def __init__(self, ui: UserInterface) -> None:
        self._ui = ui

    def write(
        self,
        video_path: Path,
        transcription: str,
        duration_seconds: float,
        cost_per_minute: float,
    ) -> Path:
        """
        Write transcription output to a text file.

        Args:
            video_path: Path to the original video file.
            transcription: The transcribed text.
            duration_seconds: Duration of the transcribed audio.
            cost_per_minute: Cost per minute for transcription.

        Returns:
            Path to the output file.
        """
        output_path = video_path.with_name(f"{video_path.stem}_transcription.txt")

        minutes = duration_seconds / 60
        cost = minutes * cost_per_minute

        content = f"""# Transcription of {video_path.name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Audio duration: {minutes:.1f} minutes
# Estimated cost: ${cost:.4f}

{transcription}
"""

        output_path.write_text(content, encoding="utf-8")
        self._ui.display_transcription_complete(output_path)

        return output_path
