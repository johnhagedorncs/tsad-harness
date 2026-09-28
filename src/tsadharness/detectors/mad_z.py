"""Self-referencing robust z-score. The detector that fails.

This is the rule as it is most commonly written: take the median and the MAD
of the series in front of you, divide, flag anything beyond three sigma. It is
correct for point anomalies and level shifts, and it is structurally incapable
of reporting a ramp of any magnitude.

The reason is that the reference statistics are computed from the same data
being scored. A ramp drags the median into its own middle and inflates the MAD
in exact proportion to its own range, so the two effects cancel and the
resulting z-score is scale free. See
:data:`~tsadharness.detectors.base.RAMP_BLINDNESS_CONSTANT` and
``paper/ramp_blindness.md``.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, interpolate_nans, robust_z, scaled_mad


class MADZScore(BaseDetector):
    """Robust z-score whose center and scale come from the scored series.

    ``fit`` is intentionally a no-op beyond recording that it was called: this
    detector has no memory of a healthy period, which is precisely the defect
    under study.

    Args:
        threshold: Firing threshold in z units. The conventional 3.0 is above
            the ramp ceiling of ~1.349 and is therefore unreachable on ramp
            geometries at any magnitude.
    """

    name = "MADZScore"

    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__(threshold=threshold)

    def fit(self, series: np.ndarray) -> None:
        """Record that fit was called. No statistics are retained by design."""
        _ = np.asarray(series, dtype=float)
        self._fitted = True

    def score(self, series: np.ndarray) -> np.ndarray:
        """Score ``series`` against its own median and scaled MAD."""
        self._require_fit()
        values = interpolate_nans(series)
        center = float(np.median(values))
        scale = scaled_mad(values, center=center)
        return robust_z(values, center, scale)
