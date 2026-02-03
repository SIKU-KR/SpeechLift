#!/usr/bin/env python3
"""
Video-to-Text Transcription using OpenAI Whisper API.

Extracts audio from video files and transcribes using Whisper with VAD-based chunking.
"""

from pathlib import Path


def main() -> None:
    """Main entry point."""
    from transcriber import create_app

    app = create_app()
    app.run(Path.cwd())


if __name__ == "__main__":
    main()
