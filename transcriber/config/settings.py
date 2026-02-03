"""Configuration settings and dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TranscriptionSettings:
    """Settings for the transcription process."""

    min_chunk_duration: float = 8.0
    max_chunk_duration: float = 15.0
    max_concurrent_requests: int = 10
    max_retries: int = 3
    cost_per_minute: float = 0.006


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration."""

    config_file: Path = field(default_factory=lambda: Path.home() / ".video_transcriber_config.json")
    video_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"})
    )
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)


def get_default_config() -> AppConfig:
    """Get the default application configuration."""
    return AppConfig()
