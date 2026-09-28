"""Common detector protocol and shared numeric helpers.

Every detector in this harness follows the same three-step contract:

1. ``fit(series)`` sees a series and learns whatever it needs, including its
   own decision threshold.
2. ``score(series)`` returns one score per point, higher meaning more
   anomalous.
3. ``detect(series)`` binarises the scores at ``threshold``.

``fit`` deliberately receives the *whole* series rather than a pre-sliced
training set. Which part of that series a detector is entitled to treat as
"normal" is a modelling decision, and forcing each detector to declare it is
the entire point of the comparison: :class:`~tsadharness.detectors.mad_z.MADZScore`
declares nothing and recomputes its reference from the data it is scoring,
while :class:`~tsadharness.detectors.mad_ref.MADReference` declares a fixed
stationary window. That single difference is what the harness measures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

MAD_SCALE: float = 1.4826
"""Normal-consistency factor making MAD an unbiased sigma estimate for Gaussians."""

RAMP_BLINDNESS_CONSTANT: float = 0.5 / (0.25 * MAD_SCALE)
"""max|z| attainable by a self-referencing robust z-score on a uniform ramp.

Equal to ``0.5 / (0.25 * 1.4826) = 1.3489815...``. Derived in
``paper/ramp_blindness.md``: for a ramp of range R the deviations from the
median are uniform on [-R/2, +R/2], so MAD = R/4 and the scaled MAD is
0.37065*R, while the largest deviation is 0.5*R. R cancels. Any threshold
above this constant is unreachable by construction, for every R.
"""


@runtime_checkable
class Detector(Protocol):
    """Structural type implemented by every detector in the harness."""

    name: str

    def fit(self, series: np.ndarray) -> None:
        """Learn parameters and a decision threshold from ``series``."""
        ...

    def score(self, series: np.ndarray) -> np.ndarray:
        """Return per-point anomaly scores; higher means more anomalous."""
        ...


def interpolate_nans(series: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN runs so that array maths stays well defined.

    Endpoint NaNs are filled with the nearest finite value. If the series is
    entirely NaN, zeros are returned.

    This is a deliberate, declared choice rather than a silent convenience:
    interpolating a gap invents data, and any onset computed across an
    interpolated region is an index, not a time. See
    :mod:`tsadharness.controls.window_shift`.

    Args:
        series: Possibly gap-valued series.

    Returns:
        A finite float array of the same shape.
    """
    values = np.asarray(series, dtype=float).copy()
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    if finite.all():
        return values
    idx = np.arange(values.size, dtype=float)
    values[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    return values


def scaled_mad(values: np.ndarray, center: float | None = None) -> float:
    """Return the normal-consistency-scaled median absolute deviation.

    Args:
        values: Finite sample.
        center: Center to deviate from; the median of ``values`` if omitted.

    Returns:
        ``1.4826 * median(|values - center|)``. Zero input spread yields a
        floor of ``np.finfo(float).tiny`` so that callers never divide by zero.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(np.finfo(float).tiny)
    mid = float(np.median(arr)) if center is None else float(center)
    mad = float(np.median(np.abs(arr - mid)))
    scaled = MAD_SCALE * mad
    return scaled if scaled > 0.0 else float(np.finfo(float).tiny)


def robust_z(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Return ``|values - center| / scale`` with a safe scale floor."""
    safe = scale if scale > 0.0 else float(np.finfo(float).tiny)
    return np.abs(np.asarray(values, dtype=float) - center) / safe


class BaseDetector(ABC):
    """Convenience base implementing ``detect`` and fit-state bookkeeping."""

    name: str = "base"

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = float(threshold)
        self._fitted = False

    @abstractmethod
    def fit(self, series: np.ndarray) -> None:
        """Learn parameters and a decision threshold from ``series``."""

    @abstractmethod
    def score(self, series: np.ndarray) -> np.ndarray:
        """Return per-point anomaly scores; higher means more anomalous."""

    def detect(self, series: np.ndarray) -> np.ndarray:
        """Return the boolean firing mask ``score(series) > threshold``."""
        return np.asarray(self.score(series) > self.threshold, dtype=bool)

    def fit_detect(self, series: np.ndarray) -> np.ndarray:
        """Fit on ``series`` and return the firing mask for the same series."""
        self.fit(series)
        return self.detect(series)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: fit() must be called before score()")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, threshold={self.threshold})"
