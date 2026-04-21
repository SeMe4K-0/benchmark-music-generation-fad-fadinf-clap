from __future__ import annotations

import numpy as np
import torch

from .base import BaseGenerator


class MustangoGenerator(BaseGenerator):
    """MusicLDM — latent diffusion for text-to-music (replaces Mustango)."""

    name = "musicldm"

    def __init__(self, model_id: str = "ucsd-reach/musicldm", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def load_model(self) -> None:
        from diffusers import MusicLDMPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = MusicLDMPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompt: str, duration: float = 10.0) -> tuple[np.ndarray, int]:
        if self.pipe is None:
            self.load_model()

        sr = 16000

        result = self.pipe(
            prompt,
            num_inference_steps=100,
            audio_length_in_s=duration,
            num_waveforms_per_prompt=1,
            negative_prompt="low quality, average quality",
        )

        audio = result.audios[0].astype(np.float32)
        return audio, sr

    def unload_model(self) -> None:
        del self.pipe
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
