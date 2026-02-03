"""Main workflow orchestration."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriber.audio.ffmpeg_extractor import FFmpegAudioExtractor
    from transcriber.config.api_key import ApiKeyManager
    from transcriber.config.settings import AppConfig
    from transcriber.files.temp_manager import TempDirectoryManager
    from transcriber.files.video_finder import VideoFinder
    from transcriber.output.text_writer import TextFileWriter
    from transcriber.transcription.whisper_api import WhisperAPITranscriber
    from transcriber.ui.protocols import ProgressReporter, UserInterface
    from transcriber.vad.chunking import ChunkMerger
    from transcriber.vad.silero_vad import SileroVAD


class TranscriptionOrchestrator:
    """Coordinates the video transcription workflow."""

    def __init__(
        self,
        config: AppConfig,
        ui: UserInterface,
        progress: ProgressReporter,
        api_key_manager: ApiKeyManager,
        video_finder: VideoFinder,
        audio_extractor: FFmpegAudioExtractor,
        vad: SileroVAD,
        chunk_merger: ChunkMerger,
        transcriber: WhisperAPITranscriber,
        output_writer: TextFileWriter,
        temp_manager_factory: type[TempDirectoryManager],
    ) -> None:
        self._config = config
        self._ui = ui
        self._progress = progress
        self._api_key_manager = api_key_manager
        self._video_finder = video_finder
        self._audio_extractor = audio_extractor
        self._vad = vad
        self._chunk_merger = chunk_merger
        self._transcriber = transcriber
        self._output_writer = output_writer
        self._temp_manager_factory = temp_manager_factory
        self._temp_manager: TempDirectoryManager | None = None

    def run(self, directory: Path) -> None:
        """
        Run the transcription workflow.

        Args:
            directory: Directory to search for video files.
        """
        # Set up signal handlers
        self._setup_signal_handlers()

        # Display header
        self._ui.display_header()

        # Check prerequisites
        self._check_prerequisites()

        # Find videos
        videos = self._video_finder.find(directory)

        # Display menu and get selection
        selected_video = self._ui.display_video_menu(
            videos, self._config.video_extensions
        )
        if not selected_video:
            self._ui.display_info("Exiting.")
            return

        # Process the selected video
        self._process_video(selected_video)

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful cleanup."""

        def signal_handler(signum: int, frame: object) -> None:
            self._ui.display_warning("\nInterrupted. Cleaning up...")
            self._cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _check_prerequisites(self) -> None:
        """Check all prerequisites and set up API key."""
        # Get/setup API key
        api_key = self._api_key_manager.get_api_key(self._ui)
        os.environ["OPENAI_API_KEY"] = api_key

        # Check FFmpeg
        if not self._audio_extractor.is_available():
            self._audio_extractor.display_install_instructions()
            sys.exit(1)

    def _process_video(self, video_path: Path) -> None:
        """Process a single video file."""
        self._temp_manager = self._temp_manager_factory()

        try:
            with self._temp_manager as temp_dir:
                # Extract audio
                audio_path = self._audio_extractor.extract(video_path, temp_dir)

                # Detect speech segments
                speech_segments = self._vad.detect_speech(audio_path)

                if not speech_segments:
                    self._ui.display_warning("No speech detected in the audio.")
                    return

                # Get total samples for merging
                from transcriber.audio.utils import read_audio

                wav = read_audio(str(audio_path))
                total_samples = len(wav)

                # Merge into chunks
                chunk_ranges = self._chunk_merger.merge(speech_segments, total_samples)

                if not chunk_ranges:
                    self._ui.display_warning("No valid chunks created.")
                    return

                # Create chunk files
                chunk_paths = self._chunk_merger.create_chunk_files(
                    audio_path, chunk_ranges, temp_dir
                )

                if not chunk_paths:
                    self._ui.display_warning("No speech found in the video. Exiting.")
                    return

                # Calculate speech duration
                speech_duration = self._chunk_merger.calculate_speech_duration(chunk_ranges)

                # Confirm cost
                if not self._ui.confirm_cost(
                    speech_duration, self._config.transcription.cost_per_minute
                ):
                    self._ui.display_info("Cancelled.")
                    return

                # Transcribe
                self._ui.display_info("")
                transcription = self._run_transcription(chunk_paths)

                if not transcription:
                    self._ui.display_warning("No transcription generated.")
                    return

                # Save result
                self._output_writer.write(
                    video_path,
                    transcription,
                    speech_duration,
                    self._config.transcription.cost_per_minute,
                )

        except Exception as e:
            self._ui.display_error(str(e))
            raise

        finally:
            self._cleanup()

    def _run_transcription(self, chunk_paths: list[Path]) -> str:
        """Run the transcription process with progress reporting."""
        total = len(chunk_paths)

        with self._progress.progress_bar(total, "[cyan]Transcribing...") as ctx:

            def progress_callback() -> None:
                ctx.advance(1)

            return asyncio.run(
                self._transcriber.transcribe_all(chunk_paths, progress_callback)
            )

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_manager:
            self._temp_manager.cleanup()
