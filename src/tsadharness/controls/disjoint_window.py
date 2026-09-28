"""Disjoint-window control for the temporal-generalization artifact.

The failure this guards against is specific and common. A model is trained on
window [t0, t1]. It is then asked to score the period immediately after t1, and
it reports a rising error. The rise is read as the onset of a fault. Very often
it is nothing of the kind: it is the model failing to generalize past the edge
of its own training window, and the "onset" coincides with t1 because t1 is
where the model's knowledge stops, not because anything happened to the asset.

The control is a second model trained on a *non-overlapping* window, rescoring
the identical period:

* If the two models disagree about the absolute error level in that period,
  the error level is a property of the training window, hence of the pipeline.
  The finding is an artifact.
* If the two models agree, the error level is a property of the data. The
  finding survives.

This does not prove a fault is real - both models can share a blind spot - but
it is cheap and it removes the single most common false finding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..detectors.base import BaseDetector
from ..detectors.mad_ref import MADReference

Window = tuple[int, int]
ModelFactory = Callable[[], BaseDetector]


def default_model_factory() -> BaseDetector:
    """Return the default control model: a reference z-score over its whole fit slice."""
    return MADReference(threshold=3.0, reference_fraction=1.0)


@dataclass(frozen=True)
class ControlResult:
    """Outcome of a disjoint-window comparison.

    Attributes:
        event_window: The period both models were asked to score.
        window_a: Training window of the first model.
        window_b: Training window of the second model, disjoint from A.
        error_a: Mean score assigned to the event window by model A.
        error_b: Mean score assigned to the event window by model B.
        ratio: ``max(error_a, error_b) / min(error_a, error_b)``, always >= 1.
        relative_gap: ``|error_a - error_b| / max(error_a, error_b)`` in [0, 1].
        ratio_tolerance: Largest ratio still counted as agreement.
        agree: True when ``ratio <= ratio_tolerance``.
        model_a_flags: True when model A's own threshold fires in the window,
            i.e. a single-model pipeline would report a finding here.
        is_artifact: True when model A reports a finding that model B does not
            corroborate. This is the machine-readable verdict.
        verdict: Short human-readable summary.
    """

    event_window: Window
    window_a: Window
    window_b: Window
    error_a: float
    error_b: float
    ratio: float
    relative_gap: float
    ratio_tolerance: float
    agree: bool
    model_a_flags: bool
    is_artifact: bool
    verdict: str


def _validate(window: Window, n: int, label: str) -> Window:
    start, stop = int(window[0]), int(window[1])
    if not 0 <= start < stop <= n:
        raise ValueError(f"{label} window {window} is invalid for a series of length {n}")
    return start, stop


def _overlaps(a: Window, b: Window) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def disjoint_window_control(
    series: np.ndarray,
    event_window: Window,
    train_a: Window,
    train_b: Window,
    model_factory: ModelFactory = default_model_factory,
    ratio_tolerance: float = 2.0,
) -> ControlResult:
    """Rescore one period with two models trained on non-overlapping windows.

    Args:
        series: The full series.
        event_window: ``(start, stop)`` period under investigation.
        train_a: Training window for the first model.
        train_b: Training window for the second model. Must not overlap
            ``train_a``.
        model_factory: Zero-argument callable returning an unfitted detector.
            Both models come from the same factory, so the only difference
            between them is the data they saw.
        ratio_tolerance: Two error levels agree when the larger is at most this
            many times the smaller. The default of 2.0 is deliberately lenient:
            a genuine finding should survive a factor-of-two check easily.

    Returns:
        A :class:`ControlResult`.

    Raises:
        ValueError: If any window is out of range, if the training windows
            overlap, or if ``ratio_tolerance`` is below 1.
    """
    values = np.asarray(series, dtype=float)
    n = values.size
    if ratio_tolerance < 1.0:
        raise ValueError(f"ratio_tolerance must be >= 1, got {ratio_tolerance}")
    event = _validate(event_window, n, "event")
    win_a = _validate(train_a, n, "train_a")
    win_b = _validate(train_b, n, "train_b")
    if _overlaps(win_a, win_b):
        raise ValueError(
            f"train_a {win_a} and train_b {win_b} overlap; the control requires "
            "non-overlapping training windows"
        )

    model_a = model_factory()
    model_a.fit(values[win_a[0] : win_a[1]])
    scores_a = model_a.score(values[event[0] : event[1]])

    model_b = model_factory()
    model_b.fit(values[win_b[0] : win_b[1]])
    scores_b = model_b.score(values[event[0] : event[1]])

    error_a = float(np.mean(scores_a))
    error_b = float(np.mean(scores_b))
    hi = max(abs(error_a), abs(error_b))
    lo = min(abs(error_a), abs(error_b))
    tiny = float(np.finfo(float).tiny)
    ratio = hi / lo if lo > tiny else float("inf")
    relative_gap = (hi - lo) / hi if hi > tiny else 0.0
    agree = ratio <= ratio_tolerance
    model_a_flags = bool(np.any(scores_a > model_a.threshold))
    is_artifact = bool(model_a_flags and not agree)

    if is_artifact:
        verdict = (
            f"ARTIFACT: model A (trained on {win_a}) reports a finding that model B "
            f"(trained on {win_b}) does not corroborate; error levels differ by "
            f"{ratio:.1f}x. The finding is a property of the training window."
        )
    elif model_a_flags:
        verdict = (
            f"CORROBORATED: both models report a comparable error level in {event} "
            f"(ratio {ratio:.2f}x). The finding is a property of the data."
        )
    else:
        verdict = f"NO FINDING: model A does not fire in {event}; nothing to corroborate."

    return ControlResult(
        event_window=event,
        window_a=win_a,
        window_b=win_b,
        error_a=error_a,
        error_b=error_b,
        ratio=ratio,
        relative_gap=relative_gap,
        ratio_tolerance=float(ratio_tolerance),
        agree=agree,
        model_a_flags=model_a_flags,
        is_artifact=is_artifact,
        verdict=verdict,
    )


def artifact_demo(n: int = 1200, noise: float = 0.02, seed: int = 0) -> ControlResult:
    """Constructed case: no anomaly exists, yet one model confidently reports one.

    The series is a healthy two-mode asset - it alternates between two stable
    operating levels, both normal, and its ground-truth mask is all-False.
    Model A is trained on a window containing only the low mode, then asked to
    score a window containing only the high mode. It fires hard. Model B,
    trained on a disjoint window that happens to sit in the high mode, sees
    nothing unusual in the same period.

    A single-model pipeline reports a large, confident, entirely spurious
    finding here. The control catches it.

    Returns:
        A :class:`ControlResult` with ``is_artifact`` True.
    """
    from ..signals import regime_switch

    signal = regime_switch(n=n, magnitude=1.8, noise=noise, seed=seed, n_blocks=6)
    block = n // 6
    return disjoint_window_control(
        signal.values,
        event_window=(block, 2 * block),
        train_a=(0, block),
        train_b=(3 * block, 4 * block),
    )


def corroborated_demo(n: int = 1000, noise: float = 0.02, seed: int = 0) -> ControlResult:
    """Constructed case: a real step change survives the control.

    Both models are trained on disjoint windows inside the genuinely healthy
    baseline, and both agree that the final segment sits far outside it. The
    control returns ``is_artifact`` False, which is the behaviour that makes it
    useful: a control that rejected everything would be worthless.

    Returns:
        A :class:`ControlResult` with ``is_artifact`` False and ``agree`` True.
    """
    from ..signals import step_change

    signal = step_change(n=n, magnitude=3.0, noise=noise, seed=seed, baseline_frac=0.8)
    return disjoint_window_control(
        signal.values,
        event_window=(int(0.8 * n), n),
        train_a=(0, int(0.3 * n)),
        train_b=(int(0.4 * n), int(0.7 * n)),
    )
