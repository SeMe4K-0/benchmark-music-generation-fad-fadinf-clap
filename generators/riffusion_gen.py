from __future__ import annotations

import numpy as np
import torch

from .base import BaseGenerator


class RiffusionGenerator(BaseGenerator):
    """Riffusion — Stable Diffusion fine-tuned on spectrograms."""

    name = "riffusion"

    def __init__(self, model_id: str = "riffusion/riffusion-model-v1", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def load_model(self) -> None:
        from diffusers import StableDiffusionPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)

    def _spectrogram_to_audio(self, image: "PIL.Image.Image", sr: int, duration: float) -> np.ndarray:
        """Convert a Riffusion spectrogram image to a waveform via Griffin-Lim."""
        import librosa

        img_array = np.array(image.convert("L")).astype(np.float32)
        img_array = img_array / 255.0

        n_fft = 2048
        n_mels = img_array.shape[0]
        hop_length = int(sr * duration / img_array.shape[1])

        img_array = img_array[::-1, :]
        db = img_array * 80.0 - 80.0
        power = librosa.db_to_power(db)

        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_basis_pinv = np.linalg.pinv(mel_basis)
        spec = mel_basis_pinv @ power
        spec = np.maximum(spec, 0)

        audio = librosa.griffinlim(spec, hop_length=hop_length, n_iter=64)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)

    def generate(self, prompt: str, duration: float = 10.0) -> tuple[np.ndarray, int]:
        if self.pipe is None:
            self.load_model()

        sr = 44100
        spectrogram_prompt = f"spectogram of {prompt}"

        image = self.pipe(
            spectrogram_prompt,
            height=512,
            width=512,
            num_inference_steps=50,
        ).images[0]

        audio = self._spectrogram_to_audio(image, sr=sr, duration=duration)

        target_len = int(sr * duration)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        return audio, sr

    def unload_model(self) -> None:
        del self.pipe
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
