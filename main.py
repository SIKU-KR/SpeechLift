#!/usr/bin/env python3
"""
Video-to-Text Transcription using OpenAI Whisper API
Extracts audio from video files and transcribes using Whisper with VAD-based chunking.
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Rich library for beautiful CLI UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Initialize console
console = Console() if RICH_AVAILABLE else None

# Third-party imports (checked after prerequisites)
try:
    import torch
    import soundfile as sf
    from openai import AsyncOpenAI
    from pydub import AudioSegment
    from silero_vad import get_speech_timestamps, load_silero_vad
except ImportError as e:
    if RICH_AVAILABLE:
        console.print(f"[red]Missing dependency:[/red] {e}")
        console.print("Please run: [cyan]pip install -r requirements.txt[/cyan]")
    else:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
    sys.exit(1)


def read_audio(path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """Read audio file and return as torch tensor."""
    wav, sr = sf.read(path)

    # Convert to mono if stereo
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    # Resample if needed (simple approach - audio should already be 16kHz from ffmpeg)
    if sr != sampling_rate:
        # Use simple linear interpolation for resampling
        import numpy as np
        duration = len(wav) / sr
        new_length = int(duration * sampling_rate)
        wav = np.interp(
            np.linspace(0, len(wav) - 1, new_length),
            np.arange(len(wav)),
            wav
        )

    return torch.tensor(wav, dtype=torch.float32)


# Constants
CONFIG_FILE = Path.home() / '.video_transcriber_config.json'
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
COST_PER_MINUTE = 0.006  # $0.006 per minute for Whisper API
MIN_CHUNK_DURATION = 8.0  # seconds
MAX_CHUNK_DURATION = 15.0  # seconds
MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3

# Global for cleanup
temp_dir = None


def cleanup_temp_files():
    """Clean up temporary files on exit."""
    global temp_dir
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    if RICH_AVAILABLE:
        console.print("\n\n[yellow]Interrupted. Cleaning up...[/yellow]")
    else:
        print("\n\nInterrupted. Cleaning up...")
    cleanup_temp_files()
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def check_ffmpeg_installed() -> bool:
    """Check if FFmpeg is available in PATH."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_api_key() -> str:
    """Get OpenAI API key from environment, config file, or user input."""
    # 1. Check environment variable
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']

    # 2. Check config file
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
            if config.get('api_key'):
                return config['api_key']
        except (json.JSONDecodeError, IOError):
            pass

    # 3. First run: prompt user for API key
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(
            "[bold]OpenAI API Key Setup[/bold]\n\n"
            "Get your API key from: [cyan]https://platform.openai.com/api-keys[/cyan]",
            border_style="blue"
        ))
        console.print()
        api_key = Prompt.ask("[bold]Enter your OpenAI API Key[/bold]").strip()
    else:
        print("=" * 50)
        print("OpenAI API Key Setup")
        print("=" * 50)
        print("Get your API key from: https://platform.openai.com/api-keys")
        print()
        api_key = input("Enter your OpenAI API Key: ").strip()

    if not api_key.startswith('sk-'):
        if RICH_AVAILABLE:
            console.print("[red]Invalid API key format (should start with 'sk-')[/red]")
        else:
            print("Invalid API key format (should start with 'sk-')")
        sys.exit(1)

    # 4. Choose save method
    if RICH_AVAILABLE:
        console.print("\n[bold]How would you like to save the API key?[/bold]")
        console.print("  [cyan]1.[/cyan] Save to config file only (recommended)")
        console.print("  [cyan]2.[/cyan] Add to shell profile (~/.zshrc or ~/.bashrc)")
        console.print("  [cyan]3.[/cyan] Don't save (use for this session only)")
        console.print()
        choice = Prompt.ask("Enter choice", default="1").strip()
    else:
        print("\nHow would you like to save the API key?")
        print("  1. Save to config file only (recommended)")
        print("  2. Add to shell profile (~/.zshrc or ~/.bashrc)")
        print("  3. Don't save (use for this session only)")
        choice = input("Enter choice [1]: ").strip() or "1"

    if choice == "1":
        # Save to config file
        CONFIG_FILE.write_text(json.dumps({'api_key': api_key}, indent=2))
        CONFIG_FILE.chmod(0o600)  # Restrict permissions
        if RICH_AVAILABLE:
            console.print(f"[green]API key saved to {CONFIG_FILE}[/green]")
        else:
            print(f"API key saved to {CONFIG_FILE}")

    elif choice == "2":
        # Add to shell profile
        shell = os.environ.get('SHELL', '/bin/zsh')
        if 'zsh' in shell:
            profile_path = Path.home() / '.zshrc'
        else:
            profile_path = Path.home() / '.bashrc'

        export_line = f'\nexport OPENAI_API_KEY="{api_key}"\n'

        with open(profile_path, 'a') as f:
            f.write(export_line)

        if RICH_AVAILABLE:
            console.print(f"[green]Added to {profile_path}[/green]")
            console.print(f"Run [cyan]source {profile_path}[/cyan] or restart terminal to apply")
        else:
            print(f"Added to {profile_path}")
            print(f"Run 'source {profile_path}' or restart terminal to apply")

    else:
        if RICH_AVAILABLE:
            console.print("[yellow]API key will only be used for this session[/yellow]")
        else:
            print("API key will only be used for this session")

    return api_key


def check_prerequisites():
    """Check all prerequisites and set up API key."""
    # Get/setup API key
    api_key = get_api_key()
    os.environ['OPENAI_API_KEY'] = api_key

    # Check FFmpeg
    if not check_ffmpeg_installed():
        if RICH_AVAILABLE:
            console.print("\n[red]Error: ffmpeg not found in PATH[/red]")
            console.print("Install with:")
            console.print("  [cyan]macOS:[/cyan]   brew install ffmpeg")
            console.print("  [cyan]Ubuntu:[/cyan]  sudo apt install ffmpeg")
            console.print("  [cyan]Windows:[/cyan] choco install ffmpeg")
        else:
            print("\nError: ffmpeg not found in PATH")
            print("Install with:")
            print("  macOS:   brew install ffmpeg")
            print("  Ubuntu:  sudo apt install ffmpeg")
            print("  Windows: choco install ffmpeg")
        sys.exit(1)


def find_video_files(directory: Path) -> list[Path]:
    """Find all video files in the given directory."""
    videos = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(file)
    return sorted(videos, key=lambda p: p.name.lower())


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def display_menu(videos: list[Path]) -> Path | None:
    """Display video selection menu and get user choice."""
    if not videos:
        if RICH_AVAILABLE:
            console.print("\n[yellow]No video files found in current directory.[/yellow]")
            console.print(f"Supported formats: [cyan]{', '.join(sorted(VIDEO_EXTENSIONS))}[/cyan]")
        else:
            print("No video files found in current directory.")
            print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        return None

    if RICH_AVAILABLE:
        console.print()
        table = Table(
            title="Available Videos",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan"
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("File Name", style="white")
        table.add_column("Size", justify="right", style="green")

        for i, video in enumerate(videos, 1):
            size = format_file_size(video.stat().st_size)
            table.add_row(str(i), video.name, size)

        console.print(table)
        console.print()

        while True:
            try:
                choice = Prompt.ask("Enter number to select [dim](or 'q' to quit)[/dim]").strip()
                if choice.lower() == 'q':
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(videos):
                    return videos[idx]
                console.print(f"[yellow]Please enter a number between 1 and {len(videos)}[/yellow]")
            except ValueError:
                console.print("[yellow]Please enter a valid number[/yellow]")
    else:
        print("\nVideo files in current directory:")
        for i, video in enumerate(videos, 1):
            size = format_file_size(video.stat().st_size)
            print(f"  {i}. {video.name} ({size})")

        print()
        while True:
            try:
                choice = input("Enter the number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(videos):
                    return videos[idx]
                print(f"Please enter a number between 1 and {len(videos)}")
            except ValueError:
                print("Please enter a valid number")


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract audio from video as 16kHz mono WAV."""
    output_path = output_dir / "audio.wav"

    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        str(output_path)
    ]

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            progress.add_task(description=f"Extracting audio from [cyan]{video_path.name}[/cyan]...", total=None)
            result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        print(f"\nExtracting audio from {video_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if RICH_AVAILABLE:
            console.print(f"[red]FFmpeg error:[/red] {result.stderr}")
        else:
            print(f"FFmpeg error: {result.stderr}")
        raise RuntimeError("Failed to extract audio")

    if RICH_AVAILABLE:
        console.print("[green]Audio extracted successfully[/green]")

    return output_path


def merge_into_chunks(speech_timestamps: list[dict], total_samples: int, sample_rate: int = 16000) -> list[tuple[float, float]]:
    """
    Merge speech timestamps into chunks of 8-15 seconds.
    Returns list of (start_time, end_time) tuples in seconds.
    """
    if not speech_timestamps:
        return []

    chunks = []
    current_start = speech_timestamps[0]['start'] / sample_rate
    current_end = speech_timestamps[0]['end'] / sample_rate

    for ts in speech_timestamps[1:]:
        ts_start = ts['start'] / sample_rate
        ts_end = ts['end'] / sample_rate

        # Check if adding this segment keeps us under MAX_CHUNK_DURATION
        potential_end = ts_end
        potential_duration = potential_end - current_start

        if potential_duration <= MAX_CHUNK_DURATION:
            # Extend current chunk
            current_end = ts_end
        else:
            # Save current chunk if it meets minimum duration
            if current_end - current_start >= MIN_CHUNK_DURATION:
                chunks.append((current_start, current_end))
            elif chunks:
                # Merge short chunk with previous
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, current_end))
            else:
                # First chunk is short, keep it anyway
                chunks.append((current_start, current_end))

            # Start new chunk
            current_start = ts_start
            current_end = ts_end

    # Handle the last chunk
    if current_end - current_start >= MIN_CHUNK_DURATION:
        chunks.append((current_start, current_end))
    elif chunks:
        prev_start, _ = chunks.pop()
        chunks.append((prev_start, current_end))
    else:
        chunks.append((current_start, current_end))

    # Split any chunks that are too long
    final_chunks = []
    for start, end in chunks:
        duration = end - start
        if duration > MAX_CHUNK_DURATION:
            # Split into ~equal parts of MAX_CHUNK_DURATION
            num_parts = int(duration / MAX_CHUNK_DURATION) + 1
            part_duration = duration / num_parts
            for i in range(num_parts):
                part_start = start + i * part_duration
                part_end = start + (i + 1) * part_duration
                final_chunks.append((part_start, min(part_end, end)))
        else:
            final_chunks.append((start, end))

    return final_chunks


def create_vad_chunks(audio_path: Path, output_dir: Path) -> tuple[list[Path], float]:
    """
    Use Silero VAD to detect speech and create audio chunks.
    Returns (list of chunk paths, total duration in seconds).
    """
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            progress.add_task(description="Analyzing voice activity...", total=None)
            model = load_silero_vad()
            wav = read_audio(str(audio_path))
    else:
        print("Analyzing voice activity...")
        model = load_silero_vad()
        wav = read_audio(str(audio_path))

    total_samples = len(wav)
    sample_rate = 16000
    total_duration = total_samples / sample_rate

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )

    if not speech_timestamps:
        if RICH_AVAILABLE:
            console.print("[yellow]No speech detected in the audio.[/yellow]")
        else:
            print("No speech detected in the audio.")
        return [], total_duration

    # Merge into 8-15 second chunks
    chunk_ranges = merge_into_chunks(speech_timestamps, total_samples, sample_rate)

    if not chunk_ranges:
        if RICH_AVAILABLE:
            console.print("[yellow]No valid chunks created.[/yellow]")
        else:
            print("No valid chunks created.")
        return [], total_duration

    # Load full audio with pydub for slicing
    audio = AudioSegment.from_wav(str(audio_path))

    # Create chunk files
    chunk_paths = []
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    for i, (start, end) in enumerate(chunk_ranges):
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        chunk = audio[start_ms:end_ms]

        chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunk_paths.append(chunk_path)

    # Calculate actual speech duration
    speech_duration = sum(end - start for start, end in chunk_ranges)

    if RICH_AVAILABLE:
        console.print(f"[green]Created {len(chunk_paths)} chunks[/green] ({MIN_CHUNK_DURATION}-{MAX_CHUNK_DURATION}s each)")
    else:
        print(f"Created {len(chunk_paths)} chunks ({MIN_CHUNK_DURATION}-{MAX_CHUNK_DURATION}s each)")

    return chunk_paths, speech_duration


def confirm_cost(duration_seconds: float) -> bool:
    """Display estimated cost and get user confirmation."""
    minutes = duration_seconds / 60
    cost = minutes * COST_PER_MINUTE

    if RICH_AVAILABLE:
        console.print()
        info_text = Text()
        info_text.append("Audio duration: ", style="bold")
        info_text.append(f"{minutes:.1f} minutes\n", style="cyan")
        info_text.append("Estimated cost: ", style="bold")
        info_text.append(f"${cost:.4f}", style="green")

        console.print(Panel(info_text, title="Cost Estimate", border_style="blue"))
        console.print()

        return Confirm.ask("Proceed with transcription?", default=False)
    else:
        print(f"\nAudio duration: {minutes:.1f} minutes")
        print(f"Estimated cost: ${cost:.4f}")

        response = input("Proceed with transcription? [y/N]: ").strip()
        return response.lower() in ('y', 'yes')


async def transcribe_chunk_with_retry(
    client: AsyncOpenAI,
    index: int,
    path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback
) -> tuple[int, str]:
    """Transcribe a single chunk with retry logic."""
    async with semaphore:
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                with open(path, 'rb') as audio_file:
                    response = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                        # language omitted for auto-detection
                    )
                progress_callback()
                return (index, response.text)

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        # All retries failed
        if RICH_AVAILABLE:
            console.print(f"\n[yellow]Warning: Failed to transcribe chunk {index} after {MAX_RETRIES} attempts: {last_error}[/yellow]")
        else:
            print(f"\nWarning: Failed to transcribe chunk {index} after {MAX_RETRIES} attempts: {last_error}")
        progress_callback()
        return (index, "")


async def transcribe_all_chunks(chunk_paths: list[Path]) -> str:
    """Transcribe all chunks using parallel API calls."""
    if not chunk_paths:
        return ""

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    total = len(chunk_paths)

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Transcribing...", total=total)

            def progress_callback():
                progress.update(task, advance=1)

            tasks = [
                transcribe_chunk_with_retry(client, i, path, semaphore, progress_callback)
                for i, path in enumerate(chunk_paths)
            ]

            results = await asyncio.gather(*tasks)
    else:
        completed = [0]

        def progress_callback():
            completed[0] += 1
            percentage = int(completed[0] / total * 100)
            bar_width = 30
            filled = int(bar_width * completed[0] / total)
            bar = '=' * filled + '>' + ' ' * (bar_width - filled - 1) if filled < bar_width else '=' * bar_width
            print(f"\rTranscribing: [{bar}] {percentage}% ({completed[0]}/{total})", end='', flush=True)

        progress_callback_init = lambda: None
        print(f"\rTranscribing: [{'>' + ' ' * 29}] 0% (0/{total})", end='', flush=True)

        tasks = [
            transcribe_chunk_with_retry(client, i, path, semaphore, progress_callback)
            for i, path in enumerate(chunk_paths)
        ]

        results = await asyncio.gather(*tasks)
        print()  # New line after progress bar

    # Sort by index and join texts
    sorted_results = sorted(results, key=lambda x: x[0])
    transcription = ' '.join(text.strip() for _, text in sorted_results if text.strip())

    return transcription


def save_transcription(video_path: Path, transcription: str, duration_seconds: float):
    """Save transcription to a text file."""
    output_path = video_path.with_name(f"{video_path.stem}_transcription.txt")

    minutes = duration_seconds / 60
    cost = minutes * COST_PER_MINUTE

    content = f"""# Transcription of {video_path.name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Audio duration: {minutes:.1f} minutes
# Estimated cost: ${cost:.4f}

{transcription}
"""

    output_path.write_text(content, encoding='utf-8')

    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(
            f"[green]Transcription saved to:[/green]\n[cyan]{output_path}[/cyan]",
            title="Complete",
            border_style="green"
        ))
    else:
        print(f"\nTranscription saved to: {output_path}")

    return output_path


def main():
    """Main entry point."""
    global temp_dir

    # Display header
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(
            "[bold]Video-to-Text Transcriber[/bold]\n"
            "[dim]OpenAI Whisper API with VAD[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        ))
    else:
        print("=" * 50)
        print("Video-to-Text Transcription (OpenAI Whisper)")
        print("=" * 50)

    # Check prerequisites
    check_prerequisites()

    # Find videos in current directory
    current_dir = Path.cwd()
    videos = find_video_files(current_dir)

    # Display menu and get selection
    selected_video = display_menu(videos)
    if not selected_video:
        if RICH_AVAILABLE:
            console.print("[dim]Exiting.[/dim]")
        else:
            print("Exiting.")
        return

    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="video_transcribe_")
        temp_path = Path(temp_dir)

        # Extract audio
        audio_path = extract_audio(selected_video, temp_path)

        # Create VAD-based chunks
        chunk_paths, speech_duration = create_vad_chunks(audio_path, temp_path)

        if not chunk_paths:
            if RICH_AVAILABLE:
                console.print("[yellow]No speech found in the video. Exiting.[/yellow]")
            else:
                print("No speech found in the video. Exiting.")
            return

        # Confirm cost
        if not confirm_cost(speech_duration):
            if RICH_AVAILABLE:
                console.print("[dim]Cancelled.[/dim]")
            else:
                print("Cancelled.")
            return

        # Transcribe
        if RICH_AVAILABLE:
            console.print()
        else:
            print()
        transcription = asyncio.run(transcribe_all_chunks(chunk_paths))

        if not transcription:
            if RICH_AVAILABLE:
                console.print("[yellow]No transcription generated.[/yellow]")
            else:
                print("No transcription generated.")
            return

        # Save result
        save_transcription(selected_video, transcription, speech_duration)

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]Error:[/red] {e}")
        else:
            print(f"\nError: {e}")
        raise

    finally:
        # Clean up
        cleanup_temp_files()


if __name__ == "__main__":
    main()
