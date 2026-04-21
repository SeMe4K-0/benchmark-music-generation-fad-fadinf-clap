#!/usr/bin/env python3
"""Batch music generation using all 5 models.

Usage
-----
    python generate.py --models all --num-prompts 250
    python generate.py --models musicgen audioldm2 --num-prompts 10
    python generate.py --models musicgen --start-index 50 --num-prompts 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

import config
from generators import GENERATORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_prompts(path: Path, num_prompts: int | None = None, start_index: int = 0) -> list[str]:
    with open(path) as f:
        prompts: list[str] = json.load(f)
    prompts = prompts[start_index:]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def run_generation(
    model_names: list[str],
    prompts: list[str],
    duration: float,
    output_root: Path,
    start_index: int = 0,
) -> dict[str, list[dict]]:
    """Generate audio for every (model, prompt) pair.

    Returns a manifest dict mapping model name -> list of result dicts.
    """
    manifest: dict[str, list[dict]] = {}

    for model_name in model_names:
        log.info("Loading model: %s", model_name)
        gen_cls = GENERATORS[model_name]
        model_cfg = config.MODEL_CONFIGS[model_name]
        generator = gen_cls(model_id=model_cfg["model_id"])
        try:
            generator.load_model()
        except Exception as exc:
            log.error("Failed to load model %s: %s — skipping", model_name, exc)
            continue

        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []

        for idx, prompt in enumerate(tqdm(prompts, desc=model_name)):
            global_idx = start_index + idx
            filename = f"{global_idx:04d}.wav"
            out_path = model_dir / filename

            if out_path.exists():
                log.info("  [%s] skip existing %s", model_name, filename)
                results.append({
                    "index": global_idx,
                    "prompt": prompt,
                    "file": str(out_path),
                    "status": "skipped",
                })
                continue

            t0 = time.time()
            try:
                generator.generate_and_save(prompt, out_path, duration=duration)
                elapsed = time.time() - t0
                results.append({
                    "index": global_idx,
                    "prompt": prompt,
                    "file": str(out_path),
                    "elapsed_s": round(elapsed, 2),
                    "status": "ok",
                })
            except Exception as exc:
                log.error("  [%s] prompt %d failed: %s", model_name, global_idx, exc)
                results.append({
                    "index": global_idx,
                    "prompt": prompt,
                    "file": str(out_path),
                    "status": "error",
                    "error": str(exc),
                })

        generator.unload_model()
        manifest[model_name] = results
        log.info("Finished %s: %d/%d ok", model_name,
                 sum(1 for r in results if r["status"] == "ok"), len(results))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch music generation")
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Model names or 'all'",
    )
    parser.add_argument("--num-prompts", type=int, default=None, help="How many prompts to use")
    parser.add_argument("--start-index", type=int, default=0, help="Prompt start index (for resuming)")
    parser.add_argument("--duration", type=float, default=config.DURATION, help="Audio duration in seconds")
    parser.add_argument("--output-dir", type=str, default=str(config.GENERATED_DIR))
    parser.add_argument("--prompts-file", type=str, default=str(config.PROMPTS_FILE))
    args = parser.parse_args()

    model_names = list(GENERATORS.keys()) if "all" in args.models else args.models
    for name in model_names:
        if name not in GENERATORS:
            parser.error(f"Unknown model: {name}. Available: {list(GENERATORS.keys())}")

    prompts = load_prompts(Path(args.prompts_file), args.num_prompts, args.start_index)
    log.info("Prompts loaded: %d (start_index=%d)", len(prompts), args.start_index)
    log.info("Models: %s", model_names)

    output_root = Path(args.output_dir)
    manifest = run_generation(model_names, prompts, args.duration, output_root, args.start_index)

    manifest_path = output_root / "generation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
