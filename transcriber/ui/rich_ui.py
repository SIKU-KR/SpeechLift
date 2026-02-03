"""Rich library implementation of UI components."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import box


class RichProgressContext:
    """Progress context for Rich progress bars."""

    def __init__(self, progress: Progress, task_id: int) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, amount: int = 1) -> None:
        """Advance the progress by the given amount."""
        self._progress.update(self._task_id, advance=amount)


class RichProgressReporter:
    """Rich library implementation of ProgressReporter."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        """Display a spinner with a message during an operation."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self._console,
        ) as progress:
            progress.add_task(description=message, total=None)
            yield

    @contextmanager
    def progress_bar(self, total: int, description: str) -> Iterator[RichProgressContext]:
        """Display a progress bar for tracking completion."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
        ) as progress:
            task_id = progress.add_task(description, total=total)
            yield RichProgressContext(progress, task_id)


class RichUserInterface:
    """Rich library implementation of UserInterface."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def display_header(self) -> None:
        """Display the application header."""
        self._console.print()
        self._console.print(
            Panel(
                "[bold]Video-to-Text Transcriber[/bold]\n"
                "[dim]OpenAI Whisper API with VAD[/dim]",
                border_style="cyan",
                box=box.DOUBLE,
            )
        )

    def display_error(self, message: str) -> None:
        """Display an error message."""
        self._console.print(f"[red]Error:[/red] {message}")

    def display_success(self, message: str) -> None:
        """Display a success message."""
        self._console.print(f"[green]{message}[/green]")

    def display_warning(self, message: str) -> None:
        """Display a warning message."""
        self._console.print(f"[yellow]{message}[/yellow]")

    def display_info(self, message: str) -> None:
        """Display an informational message."""
        self._console.print(message)

    def display_video_menu(self, videos: list[Path], extensions: frozenset[str]) -> Path | None:
        """Display a menu of videos and get user selection."""
        if not videos:
            self._console.print("\n[yellow]No video files found in current directory.[/yellow]")
            self._console.print(f"Supported formats: [cyan]{', '.join(sorted(extensions))}[/cyan]")
            return None

        self._console.print()
        table = Table(
            title="Available Videos",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("File Name", style="white")
        table.add_column("Size", justify="right", style="green")

        for i, video in enumerate(videos, 1):
            size = self._format_file_size(video.stat().st_size)
            table.add_row(str(i), video.name, size)

        self._console.print(table)
        self._console.print()

        while True:
            try:
                choice = Prompt.ask("Enter number to select [dim](or 'q' to quit)[/dim]").strip()
                if choice.lower() == "q":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(videos):
                    return videos[idx]
                self._console.print(f"[yellow]Please enter a number between 1 and {len(videos)}[/yellow]")
            except ValueError:
                self._console.print("[yellow]Please enter a valid number[/yellow]")

    def confirm_cost(self, duration_seconds: float, cost_per_minute: float) -> bool:
        """Display cost estimate and get user confirmation."""
        minutes = duration_seconds / 60
        cost = minutes * cost_per_minute

        self._console.print()
        info_text = Text()
        info_text.append("Audio duration: ", style="bold")
        info_text.append(f"{minutes:.1f} minutes\n", style="cyan")
        info_text.append("Estimated cost: ", style="bold")
        info_text.append(f"${cost:.4f}", style="green")

        self._console.print(Panel(info_text, title="Cost Estimate", border_style="blue"))
        self._console.print()

        return Confirm.ask("Proceed with transcription?", default=False)

    def prompt_api_key(self) -> str:
        """Prompt user for their OpenAI API key."""
        self._console.print()
        self._console.print(
            Panel(
                "[bold]OpenAI API Key Setup[/bold]\n\n"
                "Get your API key from: [cyan]https://platform.openai.com/api-keys[/cyan]",
                border_style="blue",
            )
        )
        self._console.print()
        return Prompt.ask("[bold]Enter your OpenAI API Key[/bold]").strip()

    def prompt_save_method(self) -> str:
        """Prompt user for how to save the API key."""
        self._console.print("\n[bold]How would you like to save the API key?[/bold]")
        self._console.print("  [cyan]1.[/cyan] Save to config file only (recommended)")
        self._console.print("  [cyan]2.[/cyan] Add to shell profile (~/.zshrc or ~/.bashrc)")
        self._console.print("  [cyan]3.[/cyan] Don't save (use for this session only)")
        self._console.print()
        return Prompt.ask("Enter choice", default="1").strip()

    def display_chunk_info(self, num_chunks: int, min_duration: float, max_duration: float) -> None:
        """Display information about created chunks."""
        self._console.print(
            f"[green]Created {num_chunks} chunks[/green] ({min_duration}-{max_duration}s each)"
        )

    def display_transcription_complete(self, output_path: Path) -> None:
        """Display completion message with output path."""
        self._console.print()
        self._console.print(
            Panel(
                f"[green]Transcription saved to:[/green]\n[cyan]{output_path}[/cyan]",
                title="Complete",
                border_style="green",
            )
        )

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
