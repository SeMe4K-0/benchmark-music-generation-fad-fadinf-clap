#!/usr/bin/env python3
from __future__ import annotations

"""Visualise FAD evaluation results.

Reads CSV files produced by ``evaluate.py`` and generates publication-ready
plots saved to ``outputs/plots/``.

Usage
-----
    python analyze.py
    python analyze.py --results-dir outputs/results --plots-dir outputs/plots
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_ORDER = ["musicgen", "audioldm", "musicldm", "riffusion", "audioldm_l"]
MODEL_LABELS = {
    "musicgen": "MusicGen",
    "audioldm": "AudioLDM-M",
    "musicldm": "MusicLDM",
    "riffusion": "Riffusion",
    "audioldm_l": "AudioLDM-L",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2", n_colors=5)


# =========================================================================
# Plotting functions
# =========================================================================

def plot_fad_bar(fad_df: pd.DataFrame, plots_dir: Path) -> None:
    """Grouped bar chart: FAD per model, one subplot per embedding."""
    embeddings = fad_df["embedding"].unique()
    references = fad_df["reference"].unique()
    n_emb = len(embeddings)

    fig, axes = plt.subplots(1, n_emb, figsize=(5 * n_emb, 5), sharey=False)
    if n_emb == 1:
        axes = [axes]

    for ax, emb in zip(axes, embeddings):
        subset = fad_df[fad_df["embedding"] == emb].copy()
        subset["generator"] = pd.Categorical(
            subset["generator"], categories=MODEL_ORDER, ordered=True,
        )
        subset = subset.sort_values("generator")

        sns.barplot(
            data=subset, x="generator", y="fad", hue="reference",
            ax=ax, palette="Set2",
        )
        ax.set_title(f"FAD — {emb}")
        ax.set_xlabel("")
        ax.set_ylabel("FAD")
        ax.set_xticklabels(
            [MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
            rotation=25, ha="right",
        )
        ax.legend(title="Reference", fontsize=8)

    fig.tight_layout()
    path = plots_dir / "fad_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_fad_inf_bar(fad_df: pd.DataFrame, plots_dir: Path) -> None:
    """Same as above but for FAD-inf."""
    embeddings = fad_df["embedding"].unique()
    n_emb = len(embeddings)

    fig, axes = plt.subplots(1, n_emb, figsize=(5 * n_emb, 5), sharey=False)
    if n_emb == 1:
        axes = [axes]

    for ax, emb in zip(axes, embeddings):
        subset = fad_df[fad_df["embedding"] == emb].copy()
        subset["generator"] = pd.Categorical(
            subset["generator"], categories=MODEL_ORDER, ordered=True,
        )
        subset = subset.sort_values("generator")

        sns.barplot(
            data=subset, x="generator", y="fad_inf", hue="reference",
            ax=ax, palette="Set2",
        )
        ax.set_title(f"FAD-inf — {emb}")
        ax.set_xlabel("")
        ax.set_ylabel("FAD-inf")
        ax.set_xticklabels(
            [MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
            rotation=25, ha="right",
        )
        ax.legend(title="Reference", fontsize=8)

    fig.tight_layout()
    path = plots_dir / "fad_inf_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_fad_heatmap(fad_df: pd.DataFrame, plots_dir: Path) -> None:
    """Heatmap: model x embedding with FAD values, one per reference."""
    references = fad_df["reference"].unique()

    for ref in references:
        subset = fad_df[fad_df["reference"] == ref]
        pivot = subset.pivot_table(index="generator", columns="embedding", values="fad")

        ordered_idx = [m for m in MODEL_ORDER if m in pivot.index]
        pivot = pivot.reindex(ordered_idx)
        pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title(f"FAD Heatmap (ref={ref})")
        ax.set_ylabel("")
        fig.tight_layout()
        path = plots_dir / f"fad_heatmap_{ref}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("Saved %s", path)


def plot_fad_inf_heatmap(fad_df: pd.DataFrame, plots_dir: Path) -> None:
    """Heatmap for FAD-inf values."""
    references = fad_df["reference"].unique()

    for ref in references:
        subset = fad_df[fad_df["reference"] == ref]
        pivot = subset.pivot_table(index="generator", columns="embedding", values="fad_inf")

        ordered_idx = [m for m in MODEL_ORDER if m in pivot.index]
        pivot = pivot.reindex(ordered_idx)
        pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title(f"FAD-inf Heatmap (ref={ref})")
        ax.set_ylabel("")
        fig.tight_layout()
        path = plots_dir / f"fad_inf_heatmap_{ref}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("Saved %s", path)


def plot_per_song_boxplot(per_song_df: pd.DataFrame, plots_dir: Path) -> None:
    """Box-plots of per-song FAD per model, split by embedding."""
    embeddings = per_song_df["embedding"].unique()
    references = per_song_df["reference"].unique()

    for emb in embeddings:
        for ref in references:
            subset = per_song_df[
                (per_song_df["embedding"] == emb) & (per_song_df["reference"] == ref)
            ].copy()
            if subset.empty:
                continue

            subset["generator"] = pd.Categorical(
                subset["generator"], categories=MODEL_ORDER, ordered=True,
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=subset, x="generator", y="per_song_fad",
                palette=PALETTE, ax=ax, showfliers=True, fliersize=2,
            )
            ax.set_title(f"Per-song FAD — emb={emb}, ref={ref}")
            ax.set_xlabel("")
            ax.set_ylabel("Per-song FAD")
            ax.set_xticklabels(
                [MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                rotation=25, ha="right",
            )
            fig.tight_layout()
            path = plots_dir / f"per_song_boxplot_{emb}_{ref}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log.info("Saved %s", path)


def plot_clap_scores(clap_df: pd.DataFrame, plots_dir: Path) -> None:
    """Violin + strip plot of CLAP scores per model."""
    clap_df = clap_df.copy()
    clap_df["generator"] = pd.Categorical(
        clap_df["generator"], categories=MODEL_ORDER, ordered=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=clap_df, x="generator", y="clap_score",
        palette=PALETTE, ax=ax, inner="quartile", cut=0,
    )
    ax.set_title("CLAP Score (Prompt Adherence)")
    ax.set_xlabel("")
    ax.set_ylabel("CLAP Cosine Similarity")
    ax.set_xticklabels(
        [MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
        rotation=25, ha="right",
    )
    fig.tight_layout()
    path = plots_dir / "clap_scores_violin.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_fad_vs_sample_size(fad_df: pd.DataFrame, plots_dir: Path) -> None:
    """Demonstrate sample-size bias: FAD and FAD-inf side by side."""
    if "fad" not in fad_df.columns or "fad_inf" not in fad_df.columns:
        return

    fad_melted = fad_df.melt(
        id_vars=["generator", "embedding", "reference"],
        value_vars=["fad", "fad_inf"],
        var_name="metric",
        value_name="score",
    )
    fad_melted["generator"] = pd.Categorical(
        fad_melted["generator"], categories=MODEL_ORDER, ordered=True,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=fad_melted, x="generator", y="score", hue="metric",
        palette=["#e74c3c", "#2ecc71"], ax=ax, ci="sd",
    )
    ax.set_title("FAD vs FAD-inf (averaged across embeddings & references)")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_xticklabels(
        [MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
        rotation=25, ha="right",
    )
    ax.legend(title="Metric")
    fig.tight_layout()
    path = plots_dir / "fad_vs_fad_inf.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def print_summary_table(fad_df: pd.DataFrame) -> None:
    """Print a nicely formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: FAD and FAD-inf by Model, Embedding, Reference")
    print("=" * 80)

    fad_df_display = fad_df.copy()
    fad_df_display["generator"] = fad_df_display["generator"].map(
        lambda g: MODEL_LABELS.get(g, g)
    )
    fad_df_display = fad_df_display.rename(columns={
        "generator": "Model",
        "embedding": "Embedding",
        "reference": "Reference",
        "fad": "FAD",
        "fad_inf": "FAD-inf",
    })
    print(fad_df_display.to_string(index=False))
    print("=" * 80)

    print("\nMean FAD per model (across all embeddings & references):")
    means = (
        fad_df.groupby("generator")[["fad", "fad_inf"]]
        .mean()
        .reindex(MODEL_ORDER)
    )
    means.index = [MODEL_LABELS.get(m, m) for m in means.index]
    print(means.to_string())
    print()


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FAD results")
    parser.add_argument("--results-dir", type=str, default=str(config.RESULTS_DIR))
    parser.add_argument("--plots-dir", type=str, default=str(config.PLOTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data --------------------------------------------------------
    fad_csv = results_dir / "fad_results.csv"
    if not fad_csv.exists():
        log.error("FAD results not found at %s. Run evaluate.py first.", fad_csv)
        return

    fad_df = pd.read_csv(fad_csv)
    print_summary_table(fad_df)

    # Core plots
    plot_fad_bar(fad_df, plots_dir)
    plot_fad_inf_bar(fad_df, plots_dir)
    plot_fad_heatmap(fad_df, plots_dir)
    plot_fad_inf_heatmap(fad_df, plots_dir)
    plot_fad_vs_sample_size(fad_df, plots_dir)

    # Per-song FAD
    per_song_csv = results_dir / "per_song_fad.csv"
    if per_song_csv.exists():
        per_song_df = pd.read_csv(per_song_csv)
        plot_per_song_boxplot(per_song_df, plots_dir)
    else:
        log.info("No per-song FAD data found — skipping boxplots.")

    # CLAP scores
    clap_csv = results_dir / "clap_scores.csv"
    if clap_csv.exists():
        clap_df = pd.read_csv(clap_csv)
        plot_clap_scores(clap_df, plots_dir)
    else:
        log.info("No CLAP score data found — skipping violin plot.")

    log.info("All plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
