"""UI module for video transcriber."""

from .protocols import UserInterface, ProgressReporter, ProgressContext
from .plain_ui import PlainUserInterface, PlainProgressReporter

__all__ = [
    "UserInterface",
    "ProgressReporter",
    "ProgressContext",
    "PlainUserInterface",
    "PlainProgressReporter",
]

# Conditionally export Rich implementations
try:
    from .rich_ui import RichUserInterface as RichUserInterface
    from .rich_ui import RichProgressReporter as RichProgressReporter

    __all__.extend(["RichUserInterface", "RichProgressReporter"])
except ImportError:
    pass
