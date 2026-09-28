"""Onset-stability control: is the detected onset a time, or a choice?

For a step change, the onset is a physical instant. Move the baseline window
around and the detector keeps returning the same index, because the series
crosses any sane threshold at the same place.

For a ramp, it is not. A ramp crosses a threshold wherever the threshold
happens to sit, and the threshold is set by the baseline window. Widen the
baseline, or slide it later so that it swallows part of the early degradation,
and the estimated scale grows, and the crossing moves. The "detected onset"
is therefore a function of an analyst's choice, and the difference between two
such onsets is not a lead time.

This is why "we detected it N days before the work order" is not a claim this
harness will let you make about a ramp. The control recomputes the onset under
several baseline windows and, when the answer moves, sets
``suppress_lead_time_claim``. That field is machine-readable on purpose: a
caveat in a footnote gets dropped when the number is copied into a slide, and a
boolean in the result object does not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..detectors.base import BaseDetector
from ..detectors.mad_ref import MADReference

Window = tuple[int, int]
WindowedDetectorFactory = Callable[[Window], BaseDetector]

DEFAULT_PERSISTENCE: int = 5
"""Consecutive firing samples required to call an onset.

Without a persistence requirement the onset is whichever isolated noise sample
first crosses the threshold, which for a 3-sigma rule on Gaussian noise happens
roughly every 370 samples. That is a property of the noise, not of the event.
"""


def default_windowed_factory(window: Window) -> BaseDetector:
    """Return a reference z-score detector using ``window`` as its baseline."""
    return MADReference(threshold=3.0, reference_slice=window)


@dataclass(frozen=True)
class OnsetStability:
    """Result of recomputing an onset under several baseline windows.

    Attributes:
        onsets: Baseline window -> onset index, or None where the detector
            never fired persistently after that window.
        onset_values: The onsets that were found, in window order.
        spread: ``max - min`` over the onsets that were found, in samples.
        spread_fraction: ``spread`` as a fraction of the series length.
        tolerance_fraction: Spread fraction above which the onset is judged
            unstable.
        n_windows: Number of baseline windows tried.
        n_detected: Number of windows that produced an onset.
        classification: ``"step_type"`` when the onset is stable,
            ``"ramp_type"`` when it moves with the baseline window, and
            ``"undetermined"`` when too few windows produced an onset.
        suppress_lead_time_claim: True when no lead-time claim may be made from
            these onsets. Machine-readable by design.
        reason: Short explanation of the classification.
    """

    onsets: dict[Window, int | None]
    onset_values: list[int]
    spread: int | None
    spread_fraction: float | None
    tolerance_fraction: float
    n_windows: int
    n_detected: int
    classification: str
    suppress_lead_time_claim: bool
    reason: str


def default_baseline_windows(n: int) -> list[Window]:
    """Return a spread of plausible baseline windows for a series of length ``n``.

    The set mixes windows of different lengths and different start offsets,
    including later-starting windows that may already contain early
    degradation. That inclusion is intentional: in practice nobody knows where
    the healthy period ends, and the control's job is to expose how much the
    answer depends on guessing it.
    """
    return [
        (0, max(2, int(0.10 * n))),
        (0, max(3, int(0.20 * n))),
        (int(0.10 * n), max(int(0.10 * n) + 2, int(0.30 * n))),
        (0, max(4, int(0.40 * n))),
        (int(0.20 * n), max(int(0.20 * n) + 2, int(0.50 * n))),
    ]


def first_persistent_index(
    flags: np.ndarray, start: int = 0, persistence: int = DEFAULT_PERSISTENCE
) -> int | None:
    """Return the first index at or after ``start`` beginning a persistent run.

    Args:
        flags: Boolean firing mask.
        start: Earliest index to consider. Firings inside the baseline window
            are ignored, since that window was asserted to be healthy.
        persistence: Number of consecutive firing samples required.

    Returns:
        The index at which the qualifying run begins, or None.
    """
    mask = np.asarray(flags, dtype=bool)
    if persistence <= 1:
        idx = np.flatnonzero(mask)
        idx = idx[idx >= start]
        return int(idx[0]) if idx.size else None
    if mask.size < persistence:
        return None
    runs = np.convolve(mask.astype(np.int32), np.ones(persistence, dtype=np.int32), mode="valid")
    hits = np.flatnonzero(runs == persistence)
    hits = hits[hits >= start]
    return int(hits[0]) if hits.size else None


def onset_stability(
    series: np.ndarray,
    detector: WindowedDetectorFactory = default_windowed_factory,
    baseline_windows: Sequence[Window] | None = None,
    persistence: int = DEFAULT_PERSISTENCE,
    tolerance_fraction: float = 0.02,
) -> OnsetStability:
    """Recompute the onset under several baseline windows and judge stability.

    Args:
        series: The full series.
        detector: Callable mapping a baseline window to an unfitted detector
            that uses that window as its reference.
        baseline_windows: Windows to try; :func:`default_baseline_windows` if
            None.
        persistence: Consecutive firing samples required to call an onset.
        tolerance_fraction: Onset spread, as a fraction of series length, above
            which the onset is judged unstable.

    Returns:
        An :class:`OnsetStability`. Check ``suppress_lead_time_claim`` before
        reporting any lead time.

    Raises:
        ValueError: If ``tolerance_fraction`` is negative or no windows are given.
    """
    values = np.asarray(series, dtype=float)
    n = values.size
    if tolerance_fraction < 0.0:
        raise ValueError(f"tolerance_fraction must be >= 0, got {tolerance_fraction}")
    windows = list(default_baseline_windows(n) if baseline_windows is None else baseline_windows)
    if not windows:
        raise ValueError("at least one baseline window is required")

    onsets: dict[Window, int | None] = {}
    for window in windows:
        start, stop = int(window[0]), int(window[1])
        if not 0 <= start < stop <= n:
            raise ValueError(f"baseline window {window} is invalid for length {n}")
        det = detector((start, stop))
        det.fit(values)
        onsets[(start, stop)] = first_persistent_index(
            det.detect(values), start=stop, persistence=persistence
        )

    found = [v for v in onsets.values() if v is not None]
    n_detected = len(found)

    if n_detected < 2:
        return OnsetStability(
            onsets=onsets,
            onset_values=sorted(found),
            spread=None,
            spread_fraction=None,
            tolerance_fraction=tolerance_fraction,
            n_windows=len(windows),
            n_detected=n_detected,
            classification="undetermined",
            suppress_lead_time_claim=True,
            reason=(
                f"only {n_detected} of {len(windows)} baseline windows produced an "
                "onset; stability cannot be assessed, so no lead time may be claimed"
            ),
        )

    spread = int(max(found) - min(found))
    spread_fraction = spread / float(n) if n else 0.0
    inconsistent = n_detected < len(windows)
    unstable = spread_fraction > tolerance_fraction

    if unstable:
        classification = "ramp_type"
        reason = (
            f"onset moved {spread} samples ({spread_fraction:.1%} of the series) across "
            f"{len(windows)} baseline windows, exceeding the {tolerance_fraction:.1%} "
            "tolerance; the onset is a function of the baseline window, not a time"
        )
    elif inconsistent:
        classification = "undetermined"
        reason = (
            f"onset was stable to {spread} samples but {len(windows) - n_detected} of "
            f"{len(windows)} windows found nothing; detection itself is window-dependent"
        )
    else:
        classification = "step_type"
        reason = (
            f"onset was stable to {spread} samples ({spread_fraction:.1%}) across all "
            f"{len(windows)} baseline windows; the onset is a physical instant"
        )

    return OnsetStability(
        onsets=onsets,
        onset_values=sorted(found),
        spread=spread,
        spread_fraction=spread_fraction,
        tolerance_fraction=tolerance_fraction,
        n_windows=len(windows),
        n_detected=n_detected,
        classification=classification,
        suppress_lead_time_claim=bool(unstable or inconsistent),
        reason=reason,
    )
