"""Protocols for UI components."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, Iterator, Callable, runtime_checkable


class ProgressContext(Protocol):
    """Context for tracking progress within a progress bar."""

    def advance(self, amount: int = 1) -> None:
        """Advance the progress by the given amount."""
        ...


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol for reporting progress to the user."""

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        """Display a spinner with a message during an operation."""
        ...

    @contextmanager
    def progress_bar(self, total: int, description: str) -> Iterator[ProgressContext]:
        """Display a progress bar for tracking completion."""
        ...


@runtime_checkable
class UserInterface(Protocol):
    """Protocol for user interaction."""

    def display_header(self) -> None:
        """Display the application header."""
        ...

    def display_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def display_success(self, message: str) -> None:
        """Display a success message."""
        ...

    def display_warning(self, message: str) -> None:
        """Display a warning message."""
        ...

    def display_info(self, message: str) -> None:
        """Display an informational message."""
        ...

    def display_video_menu(self, videos: list[Path], extensions: frozenset[str]) -> Path | None:
        """
        Display a menu of videos and get user selection.

        Args:
            videos: List of video file paths to display.
            extensions: Supported video file extensions.

        Returns:
            Selected video path, or None if user cancelled.
        """
        ...

    def confirm_cost(self, duration_seconds: float, cost_per_minute: float) -> bool:
        """
        Display cost estimate and get user confirmation.

        Args:
            duration_seconds: Duration of audio in seconds.
            cost_per_minute: Cost per minute of transcription.

        Returns:
            True if user confirmed, False otherwise.
        """
        ...

    def prompt_api_key(self) -> str:
        """Prompt user for their OpenAI API key."""
        ...

    def prompt_save_method(self) -> str:
        """
        Prompt user for how to save the API key.

        Returns:
            "1" for config file, "2" for shell profile, "3" for session only.
        """
        ...

    def display_chunk_info(self, num_chunks: int, min_duration: float, max_duration: float) -> None:
        """Display information about created chunks."""
        ...

    def display_transcription_complete(self, output_path: Path) -> None:
        """Display completion message with output path."""
        ...
