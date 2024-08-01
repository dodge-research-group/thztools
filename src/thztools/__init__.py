"""
THzTools provides data analysis software tools for terahertz time-domain
spectroscopy (THz-TDS).
"""

from thztools.__about__ import __version__
from thztools.thztools import (
    FitResult,
    GlobalOptions,
    NoiseModel,
    NoiseResult,
    apply_frf,
    fit,
    get_option,
    noisefit,
    options,
    reset_option,
    scaleshift,
    set_option,
    timebase,
    wave,
)

__all__ = [
    "FitResult",
    "GlobalOptions",
    "NoiseModel",
    "NoiseResult",
    "fit",
    "get_option",
    "noisefit",
    "options",
    "reset_option",
    "scaleshift",
    "set_option",
    "timebase",
    "apply_frf",
    "wave",
    "__version__",
]
