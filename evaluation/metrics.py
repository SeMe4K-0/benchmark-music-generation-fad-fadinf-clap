"""Additional audio quality metrics beyond FAD."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def compute_clap_score(
    generated_root: Path,
    prompts: list[str],
) -> pd.DataFrame:
    """CLAP cosine-similarity between each prompt and its generated audio.

    Uses ``transformers`` ClapModel (laion/larger_clap_music).

    Returns DataFrame: generator, index, prompt, clap_score
    """
    from transformers import ClapModel, ClapProcessor

    model_id = "laion/larger_clap_music"
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    rows: list[dict] = []
    model_dirs = sorted(
        p for p in generated_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    for model_dir in model_dirs:
        gen_name = model_dir.name
        wav_files = sorted(model_dir.glob("*.wav"))

        for wav in wav_files:
            idx = int(wav.stem)
            if idx >= len(prompts):
                continue
            prompt = prompts[idx]

            import librosa
            audio, _sr = librosa.load(str(wav), sr=48000, mono=True)

            text_inputs = processor(
                text=[prompt], return_tensors="pt", padding=True,
            ).to(device)
            audio_inputs = processor(
                audios=[audio], sampling_rate=48000, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                text_emb = model.get_text_features(**text_inputs)
                audio_emb = model.get_audio_features(**audio_inputs)

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
            score = (text_emb @ audio_emb.T).item()

            rows.append({
                "generator": gen_name,
                "index": idx,
                "prompt": prompt,
                "clap_score": score,
            })

    return pd.DataFrame(rows)
