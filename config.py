from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
GENERATED_DIR = OUTPUTS_DIR / "generated"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PROMPTS_FILE = BASE_DIR / "prompts" / "prompts.json"

SAMPLE_RATE = 16000
DURATION = 10.0

MODEL_CONFIGS = {
    "musicgen": {
        "model_id": "facebook/musicgen-small",
        "class": "generators.musicgen_gen.MusicGenGenerator",
    },
    "audioldm": {
        "model_id": "cvssp/audioldm-m-full",
        "class": "generators.audioldm2_gen.AudioLDM2Generator",
    },
    "musicldm": {
        "model_id": "ucsd-reach/musicldm",
        "class": "generators.mustango_gen.MusicLDMGenerator",
    },
    "riffusion": {
        "model_id": "riffusion/riffusion-model-v1",
        "class": "generators.riffusion_gen.RiffusionGenerator",
    },
    "audioldm_l": {
        "model_id": "cvssp/audioldm-l-full",
        "class": "generators.ace_step_gen.ACEStepGenerator",
    },
}

EMBEDDING_MODELS = ["clap-laion-music", "encodec", "MERT-v1-95M-layer4", "vggish"]

REFERENCE_SETS = {
    "fma_pop": OUTPUTS_DIR / "reference" / "fma_pop",
    "jamendo": OUTPUTS_DIR / "reference" / "jamendo",
    "musiccaps": OUTPUTS_DIR / "reference" / "musiccaps",
    "gtzan": OUTPUTS_DIR / "reference" / "gtzan",
}

FAD_INF_SUBSAMPLE_SIZES = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
FAD_INF_NUM_RUNS = 5
