"""Detector implementations and the default comparison set."""

from __future__ import annotations

from collections.abc import Callable

from .ae_recon import AEReconstruction, sliding_windows
from .base import (
    MAD_SCALE,
    RAMP_BLINDNESS_CONSTANT,
    BaseDetector,
    Detector,
    interpolate_nans,
    robust_z,
    scaled_mad,
)
from .mad_ref import MADReference
from .mad_z import MADZScore
from .matrix_profile import (
    STUMPY_AVAILABLE,
    STUMPY_MISSING_MESSAGE,
    MatrixProfileDetector,
)
from .random_ import RandomScore
from .threshold import FixedThreshold

DetectorFactory = Callable[[], BaseDetector]

__all__ = [
    "AEReconstruction",
    "BaseDetector",
    "Detector",
    "DetectorFactory",
    "FixedThreshold",
    "MADReference",
    "MADZScore",
    "MAD_SCALE",
    "MatrixProfileDetector",
    "RAMP_BLINDNESS_CONSTANT",
    "RandomScore",
    "STUMPY_AVAILABLE",
    "STUMPY_MISSING_MESSAGE",
    "default_detectors",
    "interpolate_nans",
    "robust_z",
    "scaled_mad",
    "sliding_windows",
]


def default_detectors(include_matrix_profile: bool = True) -> dict[str, DetectorFactory]:
    """Return the default comparison set as name -> zero-argument factory.

    Factories rather than instances, so that every (geometry, magnitude, seed)
    cell in the sweep gets a detector that has never seen another series.

    Args:
        include_matrix_profile: Include the stumpy-backed detector when stumpy
            is importable. Silently omitted when it is not.
    """
    factories: dict[str, DetectorFactory] = {
        "MADZScore": lambda: MADZScore(threshold=3.0),
        "MADReference": lambda: MADReference(threshold=3.0, reference_fraction=0.2),
        "AEReconstruction": lambda: AEReconstruction(seed=0),
        "FixedThreshold": lambda: FixedThreshold(level=2.0),
        "RandomScore": lambda: RandomScore(alpha=0.01, seed=0),
    }
    if include_matrix_profile and STUMPY_AVAILABLE:
        factories["MatrixProfile"] = lambda: MatrixProfileDetector(window=32)
    return factories


def available_detector_names(include_matrix_profile: bool = True) -> list[str]:
    """Return the names in the default set, in table order."""
    return list(default_detectors(include_matrix_profile).keys())
