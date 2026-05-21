#!/usr/bin/env python3
"""Compute FAD metrics for all generated audio.

Usage
-----
    python evaluate.py
    python evaluate.py --embeddings clap-laion-music encodec --references musiccaps
    python evaluate.py --skip-per-song
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import config
from evaluation.fad_evaluator import FADEvaluator
from evaluation.metrics import compute_clap_score
from evaluation.reference_sets import prepare_reference_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


EMBEDDING_ALIASES = {
    "clap": "clap-laion-music",
    "mert": "MERT-v1-95M-layer4",
    "encodec": "encodec",
    "vggish": "vggish",
}

REFERENCE_ALIASES = {
    "fma": "fma_pop",
    "fma_pop": "fma_pop",
    "jamendo": "jamendo",
    "musiccaps": "musiccaps",
    "gtzan": "gtzan",
}


def _normalize_embeddings(names: list[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        key = name.strip()
        resolved_name = EMBEDDING_ALIASES.get(key, key)
        resolved.append(resolved_name)
    return resolved


def _normalize_references(names: list[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        key = name.strip()
        resolved_name = REFERENCE_ALIASES.get(key, key)
        resolved.append(resolved_name)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="FAD evaluation pipeline")
    parser.add_argument(
        "--embeddings", nargs="+", default=config.EMBEDDING_MODELS,
        help="Embedding model names for FAD",
    )
    parser.add_argument(
        "--references", nargs="+", default=list(config.REFERENCE_SETS.keys()),
        help="Reference set names (fma_pop, musiccaps, gtzan)",
    )
    parser.add_argument("--generated-dir", type=str, default=str(config.GENERATED_DIR))
    parser.add_argument("--results-dir", type=str, default=str(config.RESULTS_DIR))
    parser.add_argument("--skip-per-song", action="store_true", help="Skip per-song FAD (slow)")
    parser.add_argument("--skip-clap", action="store_true", help="Skip CLAP score computation")
    parser.add_argument(
        "--download-references", action="store_true",
        help="Download reference datasets if missing",
    )
    args = parser.parse_args()

    generated_root = Path(args.generated_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _normalize_embeddings(args.embeddings)
    references = _normalize_references(args.references)

    # --- Prepare reference sets -------------------------------------------
    ref_dirs: dict[str, Path] = {}
    for ref_name in references:
        ref_path = config.REFERENCE_SETS.get(ref_name)
        if ref_path is None:
            log.error("Unknown reference set: %s", ref_name)
            continue
        if args.download_references:
            prepare_reference_set(ref_name, ref_path)
        if not any(ref_path.rglob("*.wav")):
            log.warning(
                "Reference '%s' is empty at %s. "
                "Use --download-references or place WAV files there.",
                ref_name,
                ref_path,
            )
            continue
        ref_dirs[ref_name] = ref_path

    if not ref_dirs:
        log.error(
            "No valid reference sets available. Nothing to evaluate. "
            "Available references: %s",
            list(config.REFERENCE_SETS.keys()),
        )
        return

    # --- FAD evaluation ---------------------------------------------------
    evaluator = FADEvaluator(
        embedding_models=embeddings,
        reference_dirs=ref_dirs,
    )

    log.info("Computing FAD and FAD-inf …")
    fad_df = evaluator.evaluate_all(generated_root)
    fad_csv = results_dir / "fad_results.csv"
    fad_df.to_csv(fad_csv, index=False)
    log.info("FAD results saved to %s", fad_csv)
    print("\n=== FAD Results ===")
    if fad_df.empty:
        print("No FAD results (likely no generated audio or empty references).")
    else:
        print(fad_df.to_string(index=False))

    # --- Per-song FAD -----------------------------------------------------
    if not args.skip_per_song:
        log.info("Computing per-song FAD …")
        per_song_df = evaluator.evaluate_per_song(generated_root)
        per_song_csv = results_dir / "per_song_fad.csv"
        per_song_df.to_csv(per_song_csv, index=False)
        log.info("Per-song FAD saved to %s", per_song_csv)
        if per_song_df.empty:
            log.warning("Per-song FAD is empty; skipping summary.")
        else:
            summary = (
                per_song_df.groupby(["generator", "embedding", "reference"])["per_song_fad"]
                .agg(["mean", "std", "min", "max", "median"])
                .reset_index()
            )
            print("\n=== Per-song FAD Summary ===")
            print(summary.to_string(index=False))
            summary.to_csv(results_dir / "per_song_fad_summary.csv", index=False)

    # --- CLAP score -------------------------------------------------------
    if not args.skip_clap:
        log.info("Computing CLAP scores …")
        prompts = json.loads(config.PROMPTS_FILE.read_text())
        clap_df = compute_clap_score(generated_root, prompts)
        clap_csv = results_dir / "clap_scores.csv"
        clap_df.to_csv(clap_csv, index=False)
        log.info("CLAP scores saved to %s", clap_csv)
        if clap_df.empty:
            log.warning("CLAP score table is empty; skipping summary.")
        else:
            clap_summary = (
                clap_df.groupby("generator")["clap_score"]
                .agg(["mean", "std", "min", "max", "median"])
                .reset_index()
            )
            print("\n=== CLAP Score Summary ===")
            print(clap_summary.to_string(index=False))
            clap_summary.to_csv(results_dir / "clap_summary.csv", index=False)

    log.info("All evaluation complete.")


if __name__ == "__main__":
    main()
