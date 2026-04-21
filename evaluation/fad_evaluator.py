"""FAD evaluation with native embedding extraction and scipy-based Frechet distance.

All embeddings are extracted via ``transformers`` / ``torchaudio`` — no
third-party FAD libraries are required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg, stats

import config

log = logging.getLogger(__name__)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Frechet distance (the core formula)
# ======================================================================

def _frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
) -> float:
    """FAD = ||mu1-mu2||^2 + Tr(S1 + S2 - 2*sqrtm(S1 @ S2))."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ======================================================================
# Audio loading
# ======================================================================

def _load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _is_readable_wav(path: Path) -> bool:
    """Fast check if audio file can be decoded."""
    try:
        _ = _load_audio(path, sr=16000)
        return True
    except Exception:
        return False


# ======================================================================
# Embedding extractors (all via transformers / torchaudio)
# ======================================================================

class _BaseEmbedder:
    """Lazy-loading base for embedding models."""
    def extract(self, wav_files: list[Path]) -> np.ndarray:
        raise NotImplementedError


class _VGGishEmbedder(_BaseEmbedder):
    """VGGish via torchaudio (ships with torchaudio, no extra downloads)."""

    def __init__(self):
        self._model = None
        self._pipeline = None

    def _load(self):
        from torchaudio.pipelines import VGGISH as bundle
        self._pipeline = bundle
        self._model = bundle.get_model().to(_DEVICE).eval()

    def extract(self, wav_files: list[Path]) -> np.ndarray:
        import torchaudio
        if self._model is None:
            self._load()

        embs: list[np.ndarray] = []
        for wav in wav_files:
            waveform, sr = torchaudio.load(str(wav))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self._pipeline.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self._pipeline.sample_rate,
                )
            with torch.no_grad():
                emb = self._model(waveform.to(_DEVICE))  # (1, T, 128)
            embs.append(emb.mean(dim=1).cpu().numpy().flatten())
        return np.stack(embs)


class _CLAPEmbedder(_BaseEmbedder):
    """CLAP audio embeddings via transformers (laion/larger_clap_music)."""

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        from transformers import ClapModel, ClapProcessor
        model_id = "laion/larger_clap_music"
        self._processor = ClapProcessor.from_pretrained(model_id)
        self._model = ClapModel.from_pretrained(model_id).to(_DEVICE).eval()

    def extract(self, wav_files: list[Path]) -> np.ndarray:
        if self._model is None:
            self._load()
        embs: list[np.ndarray] = []
        for wav in wav_files:
            audio = _load_audio(wav, sr=48000)
            inputs = self._processor(
                audios=[audio], sampling_rate=48000, return_tensors="pt",
            ).to(_DEVICE)
            with torch.no_grad():
                emb = self._model.get_audio_features(**inputs)
            embs.append(emb.cpu().numpy().flatten())
        return np.stack(embs)


class _EnCodecEmbedder(_BaseEmbedder):
    """EnCodec encoder embeddings via transformers."""

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        from transformers import AutoProcessor, EncodecModel
        model_id = "facebook/encodec_24khz"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = EncodecModel.from_pretrained(model_id).to(_DEVICE).eval()

    def extract(self, wav_files: list[Path]) -> np.ndarray:
        if self._model is None:
            self._load()
        embs: list[np.ndarray] = []
        for wav in wav_files:
            audio = _load_audio(wav, sr=24000)
            inputs = self._processor(
                raw_audio=audio, sampling_rate=24000, return_tensors="pt",
            ).to(_DEVICE)
            with torch.no_grad():
                enc_out = self._model.encode(
                    inputs["input_values"],
                    inputs.get("padding_mask", None),
                )
            codes = enc_out.audio_codes  # (batch, n_q, T)
            emb = codes.float().mean(dim=-1).cpu().numpy().flatten()
            embs.append(emb)
        return np.stack(embs)


class _MERTEmbedder(_BaseEmbedder):
    """MERT layer-4 embeddings via transformers."""

    def __init__(self, layer: int = 4):
        self._layer = layer
        self._model = None
        self._processor = None

    def _load(self):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        model_id = "m-a-p/MERT-v1-95M"
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
        ).to(_DEVICE).eval()

    def extract(self, wav_files: list[Path]) -> np.ndarray:
        if self._model is None:
            self._load()
        embs: list[np.ndarray] = []
        for wav in wav_files:
            audio = _load_audio(wav, sr=24000)
            inputs = self._processor(
                audio, sampling_rate=24000, return_tensors="pt",
            ).to(_DEVICE)
            with torch.no_grad():
                out = self._model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[self._layer]  # (1, T, D)
            emb = hidden.mean(dim=1).cpu().numpy().flatten()
            embs.append(emb)
        return np.stack(embs)


# Singleton instances — lazily loaded on first use
_EMBEDDERS: dict[str, _BaseEmbedder] = {}


def _get_embedder(name: str) -> _BaseEmbedder:
    if name not in _EMBEDDERS:
        if name == "vggish":
            _EMBEDDERS[name] = _VGGishEmbedder()
        elif name == "clap-laion-music":
            _EMBEDDERS[name] = _CLAPEmbedder()
        elif name == "encodec":
            _EMBEDDERS[name] = _EnCodecEmbedder()
        elif name == "MERT-v1-95M-layer4":
            _EMBEDDERS[name] = _MERTEmbedder(layer=4)
        else:
            log.warning("Unknown embedding '%s' — falling back to vggish", name)
            _EMBEDDERS[name] = _VGGishEmbedder()
    return _EMBEDDERS[name]


# ======================================================================
# Embedding cache (compute once per directory+embedding)
# ======================================================================

_EMB_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _get_embeddings_cached(directory: Path, emb_name: str) -> np.ndarray:
    key = (str(directory), emb_name)
    if key not in _EMB_CACHE:
        wav_files = sorted(directory.rglob("*.wav"))
        readable_wavs = [p for p in wav_files if _is_readable_wav(p)]
        skipped = len(wav_files) - len(readable_wavs)
        if skipped > 0:
            log.warning(
                "Skipping %d unreadable WAV files in %s",
                skipped,
                directory,
            )
        wav_files = readable_wavs
        if not wav_files:
            raise ValueError(f"No WAV files in {directory}")
        log.info("  Extracting %s embeddings for %d files in %s …",
                 emb_name, len(wav_files), directory.name)
        embedder = _get_embedder(emb_name)
        _EMB_CACHE[key] = embedder.extract(wav_files)
    return _EMB_CACHE[key]


# ======================================================================
# FADEvaluator
# ======================================================================

class FADEvaluator:
    """Compute FAD, FAD-inf, and per-song FAD across multiple embeddings."""

    def __init__(
        self,
        embedding_models: list[str] | None = None,
        reference_dirs: dict[str, Path] | None = None,
    ):
        self.embedding_models = embedding_models or config.EMBEDDING_MODELS
        self.reference_dirs = reference_dirs or config.REFERENCE_SETS

    def evaluate_all(self, generated_root: Path) -> pd.DataFrame:
        """FAD and FAD-inf for every (model, embedding, reference) triple."""
        model_dirs = sorted(
            p for p in generated_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

        rows: list[dict] = []
        for model_dir in model_dirs:
            gen_name = model_dir.name
            for emb_name in self.embedding_models:
                for ref_name, ref_dir in self.reference_dirs.items():
                    if not ref_dir.exists() or not any(ref_dir.rglob("*.wav")):
                        log.warning("Reference '%s' empty — skipping.", ref_name)
                        continue
                    log.info("FAD  %s | emb=%s | ref=%s",
                             gen_name, emb_name, ref_name)

                    try:
                        fad = self._compute_fad(model_dir, ref_dir, emb_name)
                        fad_inf = self._compute_fad_inf(model_dir, ref_dir, emb_name)
                    except Exception as exc:
                        log.error(
                            "Skipping combo due to error: gen=%s emb=%s ref=%s err=%s",
                            gen_name,
                            emb_name,
                            ref_name,
                            exc,
                        )
                        continue

                    rows.append({
                        "generator": gen_name,
                        "embedding": emb_name,
                        "reference": ref_name,
                        "fad": fad,
                        "fad_inf": fad_inf,
                    })

        return pd.DataFrame(rows)

    def evaluate_per_song(self, generated_root: Path) -> pd.DataFrame:
        """Per-song FAD for outlier detection."""
        model_dirs = sorted(
            p for p in generated_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

        rows: list[dict] = []
        for model_dir in model_dirs:
            gen_name = model_dir.name
            wav_files = sorted(model_dir.glob("*.wav"))
            if not wav_files:
                continue

            for emb_name in self.embedding_models:
                for ref_name, ref_dir in self.reference_dirs.items():
                    if not ref_dir.exists() or not any(ref_dir.rglob("*.wav")):
                        continue
                    log.info("Per-song FAD  %s | emb=%s | ref=%s",
                             gen_name, emb_name, ref_name)

                    try:
                        ref_emb = _get_embeddings_cached(ref_dir, emb_name)
                        mu_ref = ref_emb.mean(axis=0)
                        sigma_ref = np.cov(ref_emb, rowvar=False)
                        eval_emb = _get_embeddings_cached(model_dir, emb_name)
                    except Exception as exc:
                        log.error(
                            "Skipping per-song due to error: gen=%s emb=%s ref=%s err=%s",
                            gen_name,
                            emb_name,
                            ref_name,
                            exc,
                        )
                        continue

                    for i, wav in enumerate(wav_files):
                        if i >= eval_emb.shape[0]:
                            break
                        e = eval_emb[i]
                        score = float(np.sum((mu_ref - e) ** 2) + np.trace(sigma_ref))
                        rows.append({
                            "generator": gen_name,
                            "embedding": emb_name,
                            "reference": ref_name,
                            "file": wav.name,
                            "per_song_fad": score,
                        })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Core FAD
    # ------------------------------------------------------------------

    def _compute_fad(self, eval_dir: Path, ref_dir: Path, emb_name: str) -> float:
        ref_emb = _get_embeddings_cached(ref_dir, emb_name)
        eval_emb = _get_embeddings_cached(eval_dir, emb_name)

        mu_r, sig_r = ref_emb.mean(axis=0), np.cov(ref_emb, rowvar=False)
        mu_e, sig_e = eval_emb.mean(axis=0), np.cov(eval_emb, rowvar=False)

        return _frechet_distance(mu_r, sig_r, mu_e, sig_e)

    # ------------------------------------------------------------------
    # FAD-inf: extrapolate to N -> inf
    # ------------------------------------------------------------------

    def _compute_fad_inf(
        self, eval_dir: Path, ref_dir: Path, emb_name: str,
    ) -> float:
        subsample_sizes = config.FAD_INF_SUBSAMPLE_SIZES
        n_runs = config.FAD_INF_NUM_RUNS

        ref_emb = _get_embeddings_cached(ref_dir, emb_name)
        eval_emb = _get_embeddings_cached(eval_dir, emb_name)
        n_total = eval_emb.shape[0]

        mu_ref = ref_emb.mean(axis=0)
        sigma_ref = np.cov(ref_emb, rowvar=False)

        valid_sizes = [s for s in subsample_sizes if s <= n_total]
        if len(valid_sizes) < 3:
            log.warning("Not enough samples (%d) for FAD-inf", n_total)
            return float("nan")

        inv_sizes: list[float] = []
        fad_means: list[float] = []

        for size in valid_sizes:
            scores: list[float] = []
            for _ in range(n_runs):
                idxs = np.random.choice(n_total, size=size, replace=True)
                subset = eval_emb[idxs]
                mu_s = subset.mean(axis=0)
                sigma_s = np.cov(subset, rowvar=False)
                scores.append(_frechet_distance(mu_ref, sigma_ref, mu_s, sigma_s))
            inv_sizes.append(1.0 / size)
            fad_means.append(float(np.mean(scores)))

        _slope, intercept, *_ = stats.linregress(inv_sizes, fad_means)
        return max(intercept, 0.0)
