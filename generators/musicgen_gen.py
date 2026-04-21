from __future__ import annotations

import numpy as np
import torch

from .base import BaseGenerator


class MusicGenGenerator(BaseGenerator):
    """Meta MusicGen — autoregressive transformer over EnCodec tokens."""

    name = "musicgen"

    def __init__(self, model_id: str = "facebook/musicgen-small", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(self, prompt: str, duration: float = 10.0) -> tuple[np.ndarray, int]:
        if self.model is None:
            self.load_model()

        sr = self.model.config.audio_encoder.sampling_rate  # 32000
        frame_rate = self.model.config.audio_encoder.frame_rate  # 50
        max_new_tokens = int(duration * frame_rate)

        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
        return audio, sr

    def unload_model(self) -> None:
        del self.model, self.processor
        self.model = self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
