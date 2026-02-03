"""
Video-to-Text Transcription Package.

A CLI tool that transcribes video files to text using OpenAI's Whisper API.
"""

from __future__ import annotations

from transcriber.orchestrator import TranscriptionOrchestrator
from transcriber.config.settings import AppConfig, TranscriptionSettings, get_default_config

__all__ = [
    "create_app",
    "TranscriptionOrchestrator",
    "AppConfig",
    "TranscriptionSettings",
]


def create_app(config: AppConfig | None = None) -> TranscriptionOrchestrator:
    """
    Create and wire together the application components.

    Args:
        config: Optional configuration. If not provided, uses defaults.

    Returns:
        A configured TranscriptionOrchestrator ready to run.
    """
    from transcriber.ui.protocols import UserInterface, ProgressReporter

    if config is None:
        config = get_default_config()

    # Try to use Rich UI, fall back to plain UI
    ui: UserInterface
    progress: ProgressReporter
    try:
        from rich.console import Console

        from transcriber.ui.rich_ui import RichUserInterface, RichProgressReporter

        console = Console()
        ui = RichUserInterface(console)
        progress = RichProgressReporter(console)
    except ImportError:
        from transcriber.ui.plain_ui import PlainUserInterface, PlainProgressReporter

        ui = PlainUserInterface()
        progress = PlainProgressReporter()

    # Create components
    from transcriber.config.api_key import ApiKeyManager
    from transcriber.files.video_finder import VideoFinder
    from transcriber.files.temp_manager import TempDirectoryManager
    from transcriber.audio.ffmpeg_extractor import FFmpegAudioExtractor
    from transcriber.vad.silero_vad import SileroVAD
    from transcriber.vad.chunking import ChunkMerger
    from transcriber.transcription.whisper_api import WhisperAPITranscriber
    from transcriber.output.text_writer import TextFileWriter

    api_key_manager = ApiKeyManager(config.config_file)
    video_finder = VideoFinder(config.video_extensions)
    audio_extractor = FFmpegAudioExtractor(progress, ui)
    vad = SileroVAD(progress)
    chunk_merger = ChunkMerger(config.transcription, ui)
    transcriber = WhisperAPITranscriber(config.transcription, progress, ui)
    output_writer = TextFileWriter(ui)

    return TranscriptionOrchestrator(
        config=config,
        ui=ui,
        progress=progress,
        api_key_manager=api_key_manager,
        video_finder=video_finder,
        audio_extractor=audio_extractor,
        vad=vad,
        chunk_merger=chunk_merger,
        transcriber=transcriber,
        output_writer=output_writer,
        temp_manager_factory=TempDirectoryManager,
    )
