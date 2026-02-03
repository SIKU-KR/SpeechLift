"""Audio utility functions."""

from __future__ import annotations

import torch
import soundfile as sf


def read_audio(path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Read audio file and return as torch tensor.

    Args:
        path: Path to the audio file.
        sampling_rate: Target sampling rate in Hz.

    Returns:
        Audio data as a torch tensor.
    """
    wav, sr = sf.read(path)

    # Convert to mono if stereo
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    # Resample if needed (simple approach - audio should already be 16kHz from ffmpeg)
    if sr != sampling_rate:
        import numpy as np

        duration = len(wav) / sr
        new_length = int(duration * sampling_rate)
        wav = np.interp(
            np.linspace(0, len(wav) - 1, new_length),
            np.arange(len(wav)),
            wav,
        )

    return torch.tensor(wav, dtype=torch.float32)
