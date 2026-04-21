from .base import BaseGenerator
from .musicgen_gen import MusicGenGenerator
from .audioldm2_gen import AudioLDM2Generator
from .mustango_gen import MustangoGenerator as MusicLDMGenerator
from .riffusion_gen import RiffusionGenerator
from .ace_step_gen import ACEStepGenerator

GENERATORS = {
    "musicgen": MusicGenGenerator,
    "audioldm": AudioLDM2Generator,
    "musicldm": MusicLDMGenerator,
    "riffusion": RiffusionGenerator,
    "audioldm_l": ACEStepGenerator,
}

__all__ = [
    "BaseGenerator",
    "MusicGenGenerator",
    "AudioLDM2Generator",
    "MusicLDMGenerator",
    "RiffusionGenerator",
    "ACEStepGenerator",
    "GENERATORS",
]
