"""Robust z-score against a declared stationary reference window. The fix.

The only change from :class:`~tsadharness.detectors.mad_z.MADZScore` is where
the median and the MAD come from: a window the analyst asserts is healthy,
fixed at fit time, rather than the data being scored. That one change removes
the cancellation that produces ramp blindness, because the scale no longer
grows with the anomaly.

The cost is honest and worth stating: the reference window is now an assumption
that can be wrong. If the declared window already contains degradation, this
detector under-reports by exactly the amount of degradation it swallowed. The
harness treats that as a modelling assumption to be checked (see
:mod:`tsadharness.controls.disjoint_window`) rather than something to hide.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, interpolate_nans, robust_z, scaled_mad


class MADReference(BaseDetector):
    """Robust z-score with center and scale frozen from a reference window.

    Args:
        threshold: Firing threshold in z units.
        reference_fraction: Leading fraction of the fitted series to treat as
            the stationary reference. Ignored when ``reference_slice`` is given.
        reference_slice: Explicit ``(start, stop)`` index pair for the
            reference window. Takes precedence over ``reference_fraction``.

    Raises:
        ValueError: If ``reference_fraction`` is outside (0, 1] or the explicit
            slice is empty.
    """

    name = "MADReference"

    def __init__(
        self,
        threshold: float = 3.0,
        reference_fraction: float = 0.2,
        reference_slice: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        if reference_slice is None and not 0.0 < reference_fraction <= 1.0:
            raise ValueError(f"reference_fraction must be in (0, 1], got {reference_fraction}")
        if reference_slice is not None and reference_slice[1] <= reference_slice[0]:
            raise ValueError(f"reference_slice must be non-empty, got {reference_slice}")
        self.reference_fraction = float(reference_fraction)
        self.reference_slice = reference_slice
        self.center_: float | None = None
        self.scale_: float | None = None
        self.window_: tuple[int, int] | None = None

    def _window(self, n: int) -> tuple[int, int]:
        if self.reference_slice is not None:
            start, stop = self.reference_slice
            return max(0, int(start)), min(n, int(stop))
        stop = max(2, int(round(n * self.reference_fraction)))
        return 0, min(n, stop)

    def fit(self, series: np.ndarray) -> None:
        """Freeze the median and scaled MAD of the declared reference window."""
        values = interpolate_nans(series)
        start, stop = self._window(values.size)
        window = values[start:stop]
        if window.size == 0:
            raise ValueError(f"{self.name}: reference window {(start, stop)} is empty")
        self.window_ = (start, stop)
        self.center_ = float(np.median(window))
        self.scale_ = scaled_mad(window, center=self.center_)
        self._fitted = True

    def score(self, series: np.ndarray) -> np.ndarray:
        """Score ``series`` against the frozen reference statistics."""
        self._require_fit()
        assert self.center_ is not None and self.scale_ is not None
        values = interpolate_nans(series)
        return robust_z(values, self.center_, self.scale_)
