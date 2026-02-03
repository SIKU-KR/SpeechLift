"""Audio processing module for video transcriber."""

from .protocols import AudioExtractor
from .ffmpeg_extractor import FFmpegAudioExtractor
from .utils import read_audio

__all__ = ["AudioExtractor", "FFmpegAudioExtractor", "read_audio"]
