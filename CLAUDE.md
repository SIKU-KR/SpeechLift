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
```

## Architecture

The application is a single-file Python script (`main.py`) with the following processing pipeline:

1. **Audio Extraction**: FFmpeg extracts 16kHz mono WAV from video
2. **VAD Analysis**: Silero VAD detects speech segments
3. **Chunking**: Speech segments are merged into 8-15 second chunks at natural boundaries
4. **Parallel Transcription**: Chunks are sent to Whisper API concurrently (up to 10 parallel requests with retry logic)
5. **Assembly**: Transcriptions are ordered and saved to `<video_name>_transcription.txt`

Key constants in `main.py`:
- `MIN_CHUNK_DURATION` / `MAX_CHUNK_DURATION`: 8-15 seconds per chunk
- `MAX_CONCURRENT_REQUESTS`: 10 parallel API calls
- `MAX_RETRIES`: 3 attempts per chunk with exponential backoff
- `COST_PER_MINUTE`: $0.006 (Whisper API pricing)

## Dependencies

- **FFmpeg**: Required system dependency for audio extraction
- **OpenAI API key**: Stored in `~/.video_transcriber_config.json` or `OPENAI_API_KEY` env var
- **Python packages**: openai, silero-vad, torch, pydub, soundfile, rich (for CLI UI)

## File Output

Transcriptions are saved as `<video_name>_transcription.txt` in the same directory as the source video, with metadata header including timestamp and cost estimate.
