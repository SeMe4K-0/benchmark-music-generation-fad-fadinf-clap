"""Download and prepare reference datasets for FAD calculation."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def prepare_reference_set(name: str, output_dir: Path) -> Path:
    """Ensure a reference audio set exists at *output_dir*.

    Supported names
    ---------------
    - ``musiccaps`` — downloads the MusicCaps dataset (Google) via HuggingFace
      ``datasets`` and converts audio to WAV files.
    - ``fma_pop`` — downloads the FMA small subset and selects popular tracks.
    - ``gtzan`` — downloads GTZAN and exports 10-second WAV clips.

    Returns the directory containing WAV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.rglob("*.wav"))
    if len(existing) >= 50:
        log.info("Reference set '%s' already has %d files — skipping download.", name, len(existing))
        return output_dir

    if name == "musiccaps":
        return _prepare_musiccaps(output_dir)
    if name == "fma_pop":
        return _prepare_fma_pop(output_dir)
    if name == "gtzan":
        return _prepare_gtzan(output_dir)
    raise ValueError(f"Unknown reference set: {name}")


def _prepare_musiccaps(output_dir: Path) -> Path:
    """Prepare MusicCaps audio clips.

    Strategy:
    1) Try ready-made HF datasets that already include an ``audio`` column.
    2) Fallback to downloading from YouTube using ``ytid + start_s`` metadata
       from ``google/MusicCaps`` and cutting 10s clips with ffmpeg.
    """
    target_count = 500
    count = _prepare_musiccaps_from_hf_audio_variants(output_dir, target_count=target_count)
    if count >= 50:
        log.info("MusicCaps: saved %d audio files to %s", count, output_dir)
        return output_dir

    log.warning(
        "Ready-made MusicCaps audio dataset not available (saved=%d). "
        "Falling back to YouTube download via ytid/start_s.",
        count,
    )
    count = _prepare_musiccaps_from_youtube(output_dir, target_count=target_count)
    log.info("MusicCaps: saved %d audio files to %s", count, output_dir)
    return output_dir


def _prepare_musiccaps_from_hf_audio_variants(output_dir: Path, target_count: int = 500) -> int:
    """Try several Hugging Face datasets that may include an ``audio`` column."""
    import soundfile as sf
    from datasets import load_dataset

    candidates = [
        ("amaai-lab/MusicCaps", "train"),
        ("amaai-lab/MusicCaps-10s", "train"),
        ("google/MusicCaps", "train"),
    ]

    count = 0
    for dataset_name, split in candidates:
        try:
            log.info("Trying MusicCaps dataset variant: %s[%s]", dataset_name, split)
            ds = load_dataset(dataset_name, split=split)
        except Exception as exc:
            log.warning("Failed to load %s[%s]: %s", dataset_name, split, exc)
            continue

        for row in ds:
            if "audio" not in row or row["audio"] is None:
                continue
            audio_data = row["audio"]
            sr = audio_data.get("sampling_rate")
            array = audio_data.get("array")
            if sr is None or array is None:
                continue

            ytid = str(row.get("ytid", row.get("youtube_id", "unknown")))
            start_s = int(float(row.get("start_s", row.get("start", 0))))
            out_path = output_dir / f"musiccaps_{ytid}_{start_s}.wav"
            if out_path.exists():
                count += 1
                if count >= target_count:
                    return count
                continue

            sf.write(str(out_path), array, sr)
            count += 1
            if count >= target_count:
                return count

        if count > 0:
            return count

    return count


def _prepare_musiccaps_from_youtube(output_dir: Path, target_count: int = 500) -> int:
    """Download MusicCaps clips by ytid + start_s using yt-dlp and ffmpeg."""
    import tempfile

    try:
        from datasets import load_dataset
    except Exception as exc:
        log.warning("datasets package unavailable for MusicCaps metadata: %s", exc)
        return 0

    ytdlp_cmd: list[str] | None = None
    if shutil.which("yt-dlp") is not None:
        ytdlp_cmd = ["yt-dlp"]
    else:
        try:
            import yt_dlp  # noqa: F401
            ytdlp_cmd = [sys.executable, "-m", "yt_dlp"]
        except Exception:
            ytdlp_cmd = None

    if ytdlp_cmd is None:
        log.warning(
            "yt-dlp is not installed (or not importable). "
            "Install it (`pip install --user yt-dlp`) to fetch MusicCaps audio."
        )
        return 0

    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg is not installed. Install ffmpeg to cut MusicCaps clips.")
        return 0

    log.info("Downloading MusicCaps metadata (google/MusicCaps) …")
    ds = load_dataset("google/MusicCaps", split="train")
    rows = list(ds)

    count = len(list(output_dir.glob("*.wav")))
    attempted = 0
    download_failed = 0
    ffmpeg_failed = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for idx, row in enumerate(rows):
            if count >= target_count:
                break

            ytid = str(row.get("ytid", "")).strip()
            if not ytid:
                continue
            start_s = float(row.get("start_s", 0.0))
            out_path = output_dir / f"musiccaps_{ytid}_{int(start_s)}.wav"
            if out_path.exists():
                count += 1
                continue

            source_path = tmp / f"{ytid}.%(ext)s"
            source_url = f"https://www.youtube.com/watch?v={ytid}"
            attempted += 1
            try:
                subprocess.run(
                    ytdlp_cmd + [
                        "-f",
                        "bestaudio",
                        "--no-playlist",
                        "-o",
                        str(source_path),
                        source_url,
                    ],
                    check=True,
                    timeout=120,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                download_failed += 1
                if attempted % 50 == 0:
                    log.info(
                        "MusicCaps fallback progress: attempts=%d saved=%d "
                        "download_failed=%d ffmpeg_failed=%d",
                        attempted,
                        count,
                        download_failed,
                        ffmpeg_failed,
                    )
                continue

            downloaded = sorted(tmp.glob(f"{ytid}.*"))
            if not downloaded:
                download_failed += 1
                continue
            src_file = downloaded[0]

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(start_s),
                        "-t",
                        "10",
                        "-i",
                        str(src_file),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-vn",
                        str(out_path),
                    ],
                    check=True,
                    timeout=120,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                ffmpeg_failed += 1
            finally:
                src_file.unlink(missing_ok=True)

            if out_path.exists():
                count += 1
                if count % 25 == 0:
                    log.info("MusicCaps YouTube fallback progress: %d clips", count)
            elif attempted % 50 == 0:
                log.info(
                    "MusicCaps fallback progress: attempts=%d saved=%d "
                    "download_failed=%d ffmpeg_failed=%d",
                    attempted,
                    count,
                    download_failed,
                    ffmpeg_failed,
                )

    log.info(
        "MusicCaps fallback finished: attempts=%d saved=%d "
        "download_failed=%d ffmpeg_failed=%d",
        attempted,
        count,
        download_failed,
        ffmpeg_failed,
    )

    return count


def _prepare_fma_pop(output_dir: Path) -> Path:
    """Download FMA-small and select a cross-genre subset.

    FMA-small (8 GiB) contains 8000 30-second clips across 8 balanced genres.
    We take the first ~500 tracks as a manageable reference set.
    """
    import tempfile
    import urllib.request
    import zipfile

    import librosa
    import numpy as np
    import soundfile as sf

    fma_url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    cache_zip = output_dir.parent / "fma_small.zip"

    if not cache_zip.exists():
        log.info("Downloading FMA-small (this may take a while) …")
        urllib.request.urlretrieve(fma_url, str(cache_zip))
    else:
        log.info("Using cached FMA-small zip at %s", cache_zip)

    # Validate cached archive; if corrupted, remove and re-download once.
    try:
        with zipfile.ZipFile(str(cache_zip), "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(f"Corrupted member: {bad_member}")
    except zipfile.BadZipFile:
        log.warning("Cached FMA archive is corrupted. Re-downloading: %s", cache_zip)
        cache_zip.unlink(missing_ok=True)
        urllib.request.urlretrieve(fma_url, str(cache_zip))
        with zipfile.ZipFile(str(cache_zip), "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(
                    f"Downloaded FMA archive is still corrupted: {bad_member}"
                )

    with tempfile.TemporaryDirectory() as tmpdir:
        log.info("Extracting FMA-small …")
        with zipfile.ZipFile(str(cache_zip)) as zf:
            mp3_members = [m for m in zf.namelist() if m.endswith(".mp3")][:500]
            for member in mp3_members:
                zf.extract(member, tmpdir)

        count = 0
        for mp3_path in Path(tmpdir).rglob("*.mp3"):
            try:
                y, sr = librosa.load(str(mp3_path), sr=16000, duration=10.0, mono=True)
                out_name = f"fma_{mp3_path.stem}.wav"
                sf.write(str(output_dir / out_name), y.astype(np.float32), sr)
                count += 1
            except Exception:
                continue

    log.info("FMA-Pop: saved %d audio files to %s", count, output_dir)
    return output_dir


def _prepare_gtzan(output_dir: Path) -> Path:
    """Download GTZAN via datasets and export up to 500 WAV clips."""
    import soundfile as sf
    from datasets import load_dataset

    target_count = 500
    count = 0

    candidates = [
        ("marsyas/gtzan", None, "train"),
        ("marsyas/gtzan", "default", "train"),
    ]

    ds = None
    for dataset_name, config_name, split in candidates:
        try:
            if config_name is None:
                log.info("Trying GTZAN dataset variant: %s[%s]", dataset_name, split)
                ds = load_dataset(dataset_name, split=split)
            else:
                log.info(
                    "Trying GTZAN dataset variant: %s/%s[%s]",
                    dataset_name,
                    config_name,
                    split,
                )
                ds = load_dataset(dataset_name, config_name, split=split)
            break
        except Exception as exc:
            log.warning("Failed to load %s[%s]: %s", dataset_name, split, exc)

    if ds is None:
        log.warning("GTZAN dataset unavailable. Saved 0 files.")
        return output_dir

    for idx, row in enumerate(ds):
        if count >= target_count:
            break
        audio_data = row.get("audio")
        if audio_data is None:
            continue
        sr = audio_data.get("sampling_rate")
        array = audio_data.get("array")
        if sr is None or array is None:
            continue

        genre = str(row.get("genre", "unknown"))
        out_path = output_dir / f"gtzan_{idx:04d}_{genre}.wav"
        if out_path.exists():
            count += 1
            continue
        sf.write(str(out_path), array, sr)
        count += 1

    log.info("GTZAN: saved %d audio files to %s", count, output_dir)
    return output_dir
