"""A naive fixed absolute threshold. The baseline nobody publishes.

This detector encodes a single number that an engineer picked in advance: "flag
anything above 2.0". It has no statistics, no training and no adaptivity.

It is in the harness because it is a surprisingly strong baseline whenever the
physical scale is known, and because its failure mode is the honest one: it
does nothing at all until the series crosses the number, so its magnitude sweep
is a step function at ``level``. A learned detector that cannot beat this on a
given geometry has not justified its complexity there.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, interpolate_nans


class FixedThreshold(BaseDetector):
    """Flags points whose raw value exceeds a pre-declared absolute level.

    Args:
        level: The absolute value above which a point is anomalous. Expressed
            in the same units as the series; the harness's baseline is 1.0.
    """

    name = "FixedThreshold"

    def __init__(self, level: float = 2.0) -> None:
        super().__init__(threshold=float(level))
        self.level = float(level)

    def fit(self, series: np.ndarray) -> None:
        """No-op: the level is supplied by the analyst, not learned."""
        _ = np.asarray(series, dtype=float)
        self._fitted = True

    def score(self, series: np.ndarray) -> np.ndarray:
        """Return the raw (gap-interpolated) series as the anomaly score."""
        self._require_fit()
        return interpolate_nans(series)
