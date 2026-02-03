# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CLI tool that transcribes video files to text using OpenAI's Whisper API. It uses Silero VAD (Voice Activity Detection) to intelligently split audio at speech boundaries, then transcribes chunks in parallel.

## Commands

```bash
# Setup (creates venv, installs dependencies)
source setup.sh

# Run the transcriber
python main.py

# Activate existing venv for subsequent sessions
source venv/bin/activate

# Linting and type checking
ruff check transcriber/ main.py
mypy transcriber/ main.py --ignore-missing-imports
```

## Architecture

The codebase follows SOLID principles with a modular package structure. `main.py` is a minimal entry point that delegates to `transcriber.create_app()`.

### Processing Pipeline

1. **Audio Extraction**: FFmpeg extracts 16kHz mono WAV from video
2. **VAD Analysis**: Silero VAD detects speech segments
3. **Chunking**: Speech segments merged into 8-15 second chunks at natural boundaries
4. **Parallel Transcription**: Up to 10 concurrent Whisper API requests with retry logic
5. **Assembly**: Transcriptions ordered and saved to `<video_name>_transcription.txt`

### Package Structure

```
transcriber/
├── __init__.py          # Factory function create_app() wires all dependencies
├── orchestrator.py      # TranscriptionOrchestrator coordinates the workflow
├── config/              # Settings dataclasses and API key management
├── ui/                  # UserInterface and ProgressReporter protocols with Rich/plain implementations
├── audio/               # AudioExtractor protocol and FFmpeg implementation
├── vad/                 # VoiceActivityDetector protocol, Silero VAD, chunk merging
├── transcription/       # Transcriber protocol and Whisper API implementation
├── output/              # OutputWriter protocol and text file writer
└── files/               # VideoFinder and TempDirectoryManager
```

### Key Design Patterns

- **Dependency Injection**: `TranscriptionOrchestrator` receives all dependencies via constructor
- **Protocol-based interfaces**: Each layer defines protocols (e.g., `UserInterface`, `AudioExtractor`) for loose coupling
- **Factory pattern**: `create_app()` in `__init__.py` handles all wiring
- **Graceful degradation**: Rich UI falls back to plain text if Rich library unavailable

### Configuration

Settings are immutable dataclasses in `transcriber/config/settings.py`:
- `TranscriptionSettings`: chunk durations, concurrency, retries, cost per minute
- `AppConfig`: config file path, video extensions, transcription settings

## Dependencies

- **FFmpeg**: Required system dependency for audio extraction
- **OpenAI API key**: Stored in `~/.video_transcriber_config.json` or `OPENAI_API_KEY` env var
- **Python packages**: openai, silero-vad, torch, pydub, soundfile, rich (optional for CLI UI)

## File Output

Transcriptions saved as `<video_name>_transcription.txt` in the same directory as the source video, with metadata header including timestamp and cost estimate.
