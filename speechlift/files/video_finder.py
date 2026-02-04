"""Video file discovery."""

from __future__ import annotations

from pathlib import Path


class VideoFinder:
    """Finds video files in a directory."""

    def __init__(self, extensions: frozenset[str]) -> None:
        self._extensions = extensions

    def find(self, directory: Path) -> list[Path]:
        """
        Find all video files in the given directory.

        Args:
            directory: Directory to search for video files.

        Returns:
            Sorted list of video file paths.
        """
        videos: list[Path] = []
        for file in directory.iterdir():
            if file.is_file() and file.suffix.lower() in self._extensions:
                videos.append(file)
        return sorted(videos, key=lambda p: p.name.lower())

    @property
    def extensions(self) -> frozenset[str]:
        """Get the set of supported video extensions."""
        return self._extensions
