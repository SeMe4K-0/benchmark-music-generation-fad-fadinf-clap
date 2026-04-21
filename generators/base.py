from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import soundfile as sf


class BaseGenerator(abc.ABC):
    """Base class for all music generators."""

    name: str = "base"

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory / GPU."""

    @abc.abstractmethod
    def generate(self, prompt: str, duration: float = 10.0) -> tuple[np.ndarray, int]:
        """Generate audio from a text prompt.

        Returns
        -------
        audio : np.ndarray
            1-D float32 waveform normalised to [-1, 1].
        sr : int
            Sample rate of the returned waveform.
        """

    def generate_and_save(self, prompt: str, output_path: Path, duration: float = 10.0) -> Path:
        """Generate audio and save to *output_path* (WAV)."""
        audio, sr = self.generate(prompt, duration=duration)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr)
        return output_path

    def unload_model(self) -> None:
        """Free GPU memory (optional override)."""
