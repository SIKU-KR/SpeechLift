"""Plain text fallback implementation of UI components."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class PlainProgressContext:
    """Progress context for plain text progress bars."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0

    def advance(self, amount: int = 1) -> None:
        """Advance the progress by the given amount."""
        self._completed += amount
        percentage = int(self._completed / self._total * 100)
        bar_width = 30
        filled = int(bar_width * self._completed / self._total)
        if filled < bar_width:
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
        else:
            bar = "=" * bar_width
        print(f"\rProgress: [{bar}] {percentage}% ({self._completed}/{self._total})", end="", flush=True)


class PlainProgressReporter:
    """Plain text fallback implementation of ProgressReporter."""

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        """Display a message during an operation."""
        print(message)
        yield

    @contextmanager
    def progress_bar(self, total: int, description: str) -> Iterator[PlainProgressContext]:
        """Display a progress bar for tracking completion."""
        print(description)
        ctx = PlainProgressContext(total)
        # Print initial progress
        print(f"\rProgress: [{'>' + ' ' * 29}] 0% (0/{total})", end="", flush=True)
        try:
            yield ctx
        finally:
            print()  # New line after progress bar completes


class PlainUserInterface:
    """Plain text fallback implementation of UserInterface."""

    def display_header(self) -> None:
        """Display the application header."""
        print("=" * 50)
        print("Video-to-Text Transcription (OpenAI Whisper)")
        print("=" * 50)

    def display_error(self, message: str) -> None:
        """Display an error message."""
        print(f"Error: {message}")

    def display_success(self, message: str) -> None:
        """Display a success message."""
        print(message)

    def display_warning(self, message: str) -> None:
        """Display a warning message."""
        print(f"Warning: {message}")

    def display_info(self, message: str) -> None:
        """Display an informational message."""
        print(message)

    def display_video_menu(self, videos: list[Path], extensions: frozenset[str]) -> Path | None:
        """Display a menu of videos and get user selection."""
        if not videos:
            print("No video files found in current directory.")
            print(f"Supported formats: {', '.join(sorted(extensions))}")
            return None

        print("\nVideo files in current directory:")
        for i, video in enumerate(videos, 1):
            size = self._format_file_size(video.stat().st_size)
            print(f"  {i}. {video.name} ({size})")

        print()
        while True:
            try:
                choice = input("Enter the number (or 'q' to quit): ").strip()
                if choice.lower() == "q":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(videos):
                    return videos[idx]
                print(f"Please enter a number between 1 and {len(videos)}")
            except ValueError:
                print("Please enter a valid number")

    def confirm_cost(self, duration_seconds: float, cost_per_minute: float) -> bool:
        """Display cost estimate and get user confirmation."""
        minutes = duration_seconds / 60
        cost = minutes * cost_per_minute

        print(f"\nAudio duration: {minutes:.1f} minutes")
        print(f"Estimated cost: ${cost:.4f}")

        response = input("Proceed with transcription? [y/N]: ").strip()
        return response.lower() in ("y", "yes")

    def prompt_api_key(self) -> str:
        """Prompt user for their OpenAI API key."""
        print("=" * 50)
        print("OpenAI API Key Setup")
        print("=" * 50)
        print("Get your API key from: https://platform.openai.com/api-keys")
        print()
        return input("Enter your OpenAI API Key: ").strip()

    def prompt_save_method(self) -> str:
        """Prompt user for how to save the API key."""
        print("\nHow would you like to save the API key?")
        print("  1. Save to config file only (recommended)")
        print("  2. Add to shell profile (~/.zshrc or ~/.bashrc)")
        print("  3. Don't save (use for this session only)")
        return input("Enter choice [1]: ").strip() or "1"

    def display_chunk_info(self, num_chunks: int, min_duration: float, max_duration: float) -> None:
        """Display information about created chunks."""
        print(f"Created {num_chunks} chunks ({min_duration}-{max_duration}s each)")

    def display_transcription_complete(self, output_path: Path) -> None:
        """Display completion message with output path."""
        print(f"\nTranscription saved to: {output_path}")

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
