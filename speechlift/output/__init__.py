"""Output module for video transcriber."""

from .protocols import OutputWriter
from .text_writer import TextFileWriter

__all__ = ["OutputWriter", "TextFileWriter"]
