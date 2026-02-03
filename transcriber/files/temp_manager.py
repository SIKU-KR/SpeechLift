"""Temporary directory lifecycle management."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import TracebackType


class TempDirectoryManager:
    """Manages a temporary directory for processing files."""

    def __init__(self, prefix: str = "video_transcribe_") -> None:
        self._prefix = prefix
        self._temp_dir: Path | None = None

    def __enter__(self) -> Path:
        """Create and return a temporary directory."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix=self._prefix))
        return self._temp_dir

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the temporary directory."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up the temporary directory if it exists."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    @property
    def path(self) -> Path | None:
        """Get the path to the temporary directory, if created."""
        return self._temp_dir
