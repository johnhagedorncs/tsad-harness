"""Matrix-profile discord detection, via stumpy when it is installed.

The import is guarded. stumpy is an optional dependency, and the rest of the
harness must run without it; when it is missing this detector reports
:data:`STUMPY_AVAILABLE` as False and tests that need it skip with a clear
message rather than failing.

A note on what this detector can and cannot see, because it matters for reading
the result tables and because its numbers in this repository are near-zero
everywhere. That is a real property of the method as configured, not a
misconfiguration, and it has two distinct causes.

First, ``stumpy.stump`` compares **z-normalized** subsequences. z-normalization
subtracts each window's own mean and divides by its own standard deviation, so
it discards level and scale by construction. A step change is *precisely* a
change in level, so after z-normalization the windows before and after a step
are near-identical and the profile does not rise. A smooth ramp is worse still:
every window inside it z-normalizes to the same straight line, so the profile
inside a ramp is *lower* than in stationary noise. Measured on a 1000-sample
series with m=32, the median profile value is about 5.4 on flat noise and about
0.97 inside a ramp.

Second, a discord is by definition a subsequence with no close neighbour. A
step that persists for 200 samples produces roughly ``m`` mutually similar
straddling windows, each of which is its own nearest neighbour's near-twin, so
the step does not register as a discord at all.

The threshold used here is additionally computed from the profile being scored,
which reintroduces the self-reference problem of
:class:`~tsadharness.detectors.mad_z.MADZScore` at a second level. The detector
is kept in the harness because "a well-regarded method contributes nothing on
these geometries" is a result worth printing, not one worth hiding.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, interpolate_nans, scaled_mad

try:  # pragma: no cover - exercised by whichever branch the environment takes
    import stumpy as _stumpy

    STUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _stumpy = None
    STUMPY_AVAILABLE = False

STUMPY_MISSING_MESSAGE = (
    "stumpy is not installed; MatrixProfileDetector is unavailable. "
    "Install it with `pip install 'tsad-harness[mp]'` to include this detector."
)


class MatrixProfileDetector(BaseDetector):
    """Flags discords, i.e. subsequences with an unusually distant neighbour.

    Args:
        window: Subsequence length ``m`` passed to ``stumpy.stump``.
        threshold_k: Threshold is ``median(profile) + threshold_k *
            scaled_mad(profile)``, computed on the profile being scored.

    Raises:
        ImportError: On construction, if stumpy is not importable.
    """

    name = "MatrixProfile"

    def __init__(self, window: int = 32, threshold_k: float = 3.0) -> None:
        super().__init__(threshold=np.inf)
        if not STUMPY_AVAILABLE:
            raise ImportError(STUMPY_MISSING_MESSAGE)
        self.window = int(window)
        self.threshold_k = float(threshold_k)

    def fit(self, series: np.ndarray) -> None:
        """No parameters are learned; the profile is computed at score time."""
        _ = interpolate_nans(series)
        self._fitted = True

    def score(self, series: np.ndarray) -> np.ndarray:
        """Return a per-point score from the matrix profile.

        A subsequence's profile value is attributed to every point that
        subsequence covers, and each point takes the maximum over the
        subsequences covering it. Attributing the value only to the window's
        start index would push a discord up to ``window - 1`` samples earlier
        than the event that caused it, which would show up as a spurious miss
        against the ground-truth mask.
        """
        self._require_fit()
        values = interpolate_nans(series)
        if values.size < 3 * self.window:
            return np.zeros(values.size)
        profile = np.asarray(_stumpy.stump(values, m=self.window)[:, 0], dtype=float)
        profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
        med = float(np.median(profile))
        self.threshold = med + self.threshold_k * scaled_mad(profile, center=med)

        out = np.zeros(values.size, dtype=float)
        for start, value in enumerate(profile):
            stop = start + self.window
            np.maximum(out[start:stop], value, out=out[start:stop])
        # Points past the last full subsequence inherit the final profile value.
        tail = profile.size + self.window - 1
        if tail < values.size:
            out[tail:] = profile[-1]
        return out
