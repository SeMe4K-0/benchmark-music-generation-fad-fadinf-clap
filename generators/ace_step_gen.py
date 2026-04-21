from __future__ import annotations

import numpy as np
import torch

from .base import BaseGenerator


class ACEStepGenerator(BaseGenerator):
    """AudioLDM Large — larger latent diffusion model for text-to-audio."""

    name = "audioldm_l"

    def __init__(self, model_id: str = "cvssp/audioldm-l-full", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def load_model(self) -> None:
        from diffusers import AudioLDMPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = AudioLDMPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompt: str, duration: float = 10.0) -> tuple[np.ndarray, int]:
        if self.pipe is None:
            self.load_model()

        sr = 16000

        result = self.pipe(
            prompt,
            num_inference_steps=50,
            audio_length_in_s=duration,
            num_waveforms_per_prompt=1,
        )

        audio = result.audios[0].astype(np.float32)
        return audio, sr

    def unload_model(self) -> None:
        del self.pipe
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
