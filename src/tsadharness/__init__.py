"""tsad-harness: a test harness for time-series anomaly detectors.

The harness generates signals with known ground truth, runs several standard
detectors against them, and reports which failure geometries each detector
cannot see. Two controls catch findings that are artifacts of a detector's own
training window rather than properties of the data.

The central result is that a self-referencing robust z-score cannot detect a
ramp at any magnitude:
:data:`~tsadharness.detectors.base.RAMP_BLINDNESS_CONSTANT`.
"""

from __future__ import annotations

from .controls import ControlResult, OnsetStability, disjoint_window_control, onset_stability
from .detectors import (
    RAMP_BLINDNESS_CONSTANT,
    AEReconstruction,
    FixedThreshold,
    MADReference,
    MADZScore,
    MatrixProfileDetector,
    RandomScore,
    default_detectors,
)
from .protocol import (
    event_wise,
    false_positive_table,
    magnitude_sweep,
    point_adjusted,
    point_wise,
    ramp_ceiling_table,
)
from .signals import GENERATORS, Signal, make_signal

__version__ = "0.1.0"

__all__ = [
    "AEReconstruction",
    "ControlResult",
    "FixedThreshold",
    "GENERATORS",
    "MADReference",
    "MADZScore",
    "MatrixProfileDetector",
    "OnsetStability",
    "RAMP_BLINDNESS_CONSTANT",
    "RandomScore",
    "Signal",
    "__version__",
    "default_detectors",
    "disjoint_window_control",
    "event_wise",
    "false_positive_table",
    "magnitude_sweep",
    "make_signal",
    "onset_stability",
    "point_adjusted",
    "point_wise",
    "ramp_ceiling_table",
]
