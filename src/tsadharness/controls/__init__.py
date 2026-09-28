"""Controls that check whether a finding is a property of the data or of the pipeline."""

from __future__ import annotations

from .disjoint_window import (
    ControlResult,
    artifact_demo,
    corroborated_demo,
    default_model_factory,
    disjoint_window_control,
)
from .window_shift import (
    DEFAULT_PERSISTENCE,
    OnsetStability,
    default_baseline_windows,
    default_windowed_factory,
    first_persistent_index,
    onset_stability,
)

__all__ = [
    "ControlResult",
    "DEFAULT_PERSISTENCE",
    "OnsetStability",
    "artifact_demo",
    "corroborated_demo",
    "default_baseline_windows",
    "default_model_factory",
    "default_windowed_factory",
    "disjoint_window_control",
    "first_persistent_index",
    "onset_stability",
]
