"""Voice Activity Detection module for video transcriber."""

from .protocols import VoiceActivityDetector, SpeechSegment
from .silero_vad import SileroVAD
from .chunking import ChunkMerger

__all__ = ["VoiceActivityDetector", "SpeechSegment", "SileroVAD", "ChunkMerger"]
