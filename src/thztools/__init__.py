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
    fit,
    get_option,
    noisefit,
    options,
    scaleshift,
    set_option,
    timebase,
    transfer,
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
    "scaleshift",
    "set_option",
    "timebase",
    "transfer",
    "wave",
    "__version__",
]
