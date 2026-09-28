"""Evaluation protocol: point-wise, event-wise and point-adjusted scoring.

Three scorings are reported side by side throughout this repository, and the
disagreement between them is itself a result.

**Point-wise** treats every timestamp independently. It is the strictest and
the least forgiving of a detector that fires slightly late.

**Event-wise** collapses each contiguous ground-truth run to a single unit: an
event counts as found if the detector fires anywhere inside it. This matches
how an operator actually consumes alerts.

**Point-adjusted** takes a point-wise prediction and, for every ground-truth
event containing at least one predicted point, marks the *entire* event as
predicted. This is the metric used by a large fraction of the published
literature and it is severely inflated. A single lucky point inside a
1000-sample event buys 1000 true positives. Under this metric a uniform random
scorer is close to unbeatable on any benchmark with long events, which is one
of the concrete mechanisms behind the argument in Wu and Keogh,
"Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating
the Illusion of Progress" (arXiv:2009.13807).

Point-adjusted F1 is computed here so that it can be printed next to the other
two and seen to be inflated. It should not be used on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .detectors import DetectorFactory, default_detectors
from .signals import GENERATORS, NO_ANOMALY_GENERATORS, Signal, make_signal

DEFAULT_MAGNITUDES: tuple[float, ...] = (
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    300.0,
    1000.0,
)
"""Magnitude grid for the sweep, spanning nearly three orders of magnitude."""

DEFAULT_REPEATS: int = 5
"""Independent seeds per (detector, geometry, magnitude) cell."""

SWEEP_GEOMETRIES: tuple[str, ...] = (
    "linear_ramp",
    "exponential_ramp",
    "seasonal_plus_ramp",
    "ramp_with_gaps",
    "step_change",
    "step_with_recovery",
    "spikes",
)
"""Geometries carrying a real anomaly, swept over magnitude."""


@dataclass(frozen=True)
class ScoreSet:
    """Precision, recall and F1 with the underlying counts retained."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float]:
        """Return the score set as a plain dictionary."""
        return dict(asdict(self))


def _prf(tp: int, fp: int, fn: int) -> ScoreSet:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ScoreSet(precision, recall, f1, int(tp), int(fp), int(fn))


def _as_mask(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=bool).ravel()


def point_wise(y_true: np.ndarray, y_pred: np.ndarray) -> ScoreSet:
    """Score every timestamp independently.

    Args:
        y_true: Boolean ground-truth mask.
        y_pred: Boolean prediction mask of the same length.

    Returns:
        Point-wise precision, recall and F1.
    """
    t, p = _as_mask(y_true), _as_mask(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    return _prf(tp, fp, fn)


def find_events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True runs as half-open ``(start, stop)`` intervals.

    Args:
        mask: Boolean mask.

    Returns:
        A list of intervals; empty when the mask is all False.
    """
    m = _as_mask(mask)
    if m.size == 0 or not m.any():
        return []
    padded = np.concatenate(([False], m, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Apply the point-adjust transform to a prediction mask.

    Every ground-truth event containing at least one predicted point is marked
    entirely predicted. Predictions outside ground-truth events are left alone.

    This function exists so that the inflation it produces can be measured.
    See the module docstring.

    Args:
        y_true: Boolean ground-truth mask.
        y_pred: Boolean prediction mask.

    Returns:
        The adjusted prediction mask.
    """
    t, p = _as_mask(y_true), _as_mask(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    adjusted = p.copy()
    for start, stop in find_events(t):
        if p[start:stop].any():
            adjusted[start:stop] = True
    return adjusted


def point_adjusted(y_true: np.ndarray, y_pred: np.ndarray) -> ScoreSet:
    """Score point-wise after applying :func:`point_adjust`. Known to inflate."""
    return point_wise(y_true, point_adjust(y_true, y_pred))


def event_wise(y_true: np.ndarray, y_pred: np.ndarray) -> ScoreSet:
    """Score at the level of contiguous events.

    A ground-truth event is a true positive if the prediction overlaps it
    anywhere. A predicted run that overlaps no ground-truth event is one false
    positive, regardless of how long it is.

    Args:
        y_true: Boolean ground-truth mask.
        y_pred: Boolean prediction mask.

    Returns:
        Event-wise precision, recall and F1.
    """
    t, p = _as_mask(y_true), _as_mask(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    true_events = find_events(t)
    pred_events = find_events(p)
    tp = sum(1 for a, b in true_events if p[a:b].any())
    fn = len(true_events) - tp
    fp = sum(1 for a, b in pred_events if not t[a:b].any())
    return _prf(tp, fp, fn)


def all_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, ScoreSet]:
    """Return all three scorings keyed by ``point``, ``event``, ``point_adjusted``."""
    return {
        "point": point_wise(y_true, y_pred),
        "event": event_wise(y_true, y_pred),
        "point_adjusted": point_adjusted(y_true, y_pred),
    }


def detected(signal: Signal, y_pred: np.ndarray) -> bool:
    """True when the prediction fires at least once inside the ground truth.

    This is the quantity plotted in the magnitude sweep. It is deliberately
    generous - one point anywhere inside the event is enough - so that a
    detection rate of zero cannot be blamed on a strict matching rule.
    """
    if not signal.has_anomaly:
        raise ValueError(f"{signal.name} has no anomaly; use fired() instead")
    return bool((_as_mask(y_pred) & signal.ground_truth).any())


def fired(y_pred: np.ndarray) -> bool:
    """True when the prediction fires anywhere at all."""
    return bool(_as_mask(y_pred).any())


def evaluate_once(
    factory: DetectorFactory,
    signal: Signal,
) -> dict[str, float]:
    """Fit a fresh detector on one signal and return its scores.

    Args:
        factory: Zero-argument callable returning an unfitted detector.
        signal: The signal to fit and score.

    Returns:
        A flat dictionary of metrics for this single trial.
    """
    detector = factory()
    detector.fit(signal.values)
    scores = detector.score(signal.values)
    pred = np.asarray(scores > detector.threshold, dtype=bool)
    row: dict[str, float] = {
        "max_score": float(np.max(scores)) if scores.size else 0.0,
        "threshold": float(detector.threshold),
        "fired": float(fired(pred)),
        "point_fire_rate": float(np.mean(pred)) if pred.size else 0.0,
        "n_pred_events": float(len(find_events(pred))),
    }
    if signal.has_anomaly:
        row["detected"] = float(detected(signal, pred))
        # The fraction of firings that land on healthy data. A detector can post
        # a detection rate of 1.0 while almost every point it flags is wrong -
        # see the MatrixProfile row on linear_ramp - so this is recorded next to
        # the detection rate rather than left to be inferred.
        n_fired = int(np.sum(pred))
        row["outside_gt_fire_fraction"] = (
            float(np.sum(pred & ~signal.ground_truth) / n_fired) if n_fired else 0.0
        )
        for key, s in all_scores(signal.ground_truth, pred).items():
            row[f"{key}_precision"] = s.precision
            row[f"{key}_recall"] = s.recall
            row[f"{key}_f1"] = s.f1
    return row


def magnitude_sweep(
    detectors: dict[str, DetectorFactory] | None = None,
    geometries: Sequence[str] = SWEEP_GEOMETRIES,
    magnitudes: Sequence[float] = DEFAULT_MAGNITUDES,
    repeats: int = DEFAULT_REPEATS,
    n: int = 1000,
    noise: float = 0.02,
) -> list[dict[str, object]]:
    """Sweep detection rate over anomaly magnitude for every detector.

    Args:
        detectors: Name -> factory mapping; the default comparison set if None.
        geometries: Generator names to sweep. Must all contain an anomaly.
        magnitudes: Magnitude grid.
        repeats: Independent seeds per cell.
        n: Series length.
        noise: Noise standard deviation.

    Returns:
        One row per (detector, geometry, magnitude) cell, with the detection
        rate and the three F1 variants averaged over repeats.
    """
    factories = default_detectors() if detectors is None else detectors
    rows: list[dict[str, object]] = []
    for det_name, factory in factories.items():
        for geom in geometries:
            if geom not in GENERATORS:
                raise KeyError(f"unknown generator {geom!r}")
            for mag in magnitudes:
                trials = [
                    evaluate_once(
                        factory,
                        make_signal(geom, n=n, magnitude=mag, noise=noise, seed=seed),
                    )
                    for seed in range(repeats)
                ]
                shape = make_signal(geom, n=n, magnitude=mag, noise=noise, seed=0)
                rows.append(
                    {
                        "detector": det_name,
                        "generator": geom,
                        "shape": shape.metadata.get("shape", "unknown"),
                        "magnitude": float(mag),
                        "repeats": int(repeats),
                        "detection_rate": float(np.mean([t["detected"] for t in trials])),
                        "outside_gt_fire_fraction": float(
                            np.mean([t["outside_gt_fire_fraction"] for t in trials])
                        ),
                        "point_precision": float(np.mean([t["point_precision"] for t in trials])),
                        "mean_max_score": float(np.mean([t["max_score"] for t in trials])),
                        "threshold": float(np.mean([t["threshold"] for t in trials])),
                        "point_f1": float(np.mean([t["point_f1"] for t in trials])),
                        "event_f1": float(np.mean([t["event_f1"] for t in trials])),
                        "point_adjusted_f1": float(
                            np.mean([t["point_adjusted_f1"] for t in trials])
                        ),
                    }
                )
    return rows


def false_positive_table(
    detectors: dict[str, DetectorFactory] | None = None,
    generators: Sequence[str] = tuple(NO_ANOMALY_GENERATORS),
    repeats: int = 20,
    n: int = 1000,
    noise: float = 0.02,
) -> list[dict[str, object]]:
    """Measure how often each detector fires when there is nothing to find.

    Args:
        detectors: Name -> factory mapping; the default comparison set if None.
        generators: Generator names whose ground truth is all-False.
        repeats: Independent seeds per cell.
        n: Series length.
        noise: Noise standard deviation.

    Returns:
        One row per (detector, generator) cell.

    Raises:
        ValueError: If a named generator actually contains an anomaly.
    """
    factories = default_detectors() if detectors is None else detectors
    rows: list[dict[str, object]] = []
    for det_name, factory in factories.items():
        for geom in generators:
            probe = make_signal(geom, n=n, noise=noise, seed=0)
            if probe.has_anomaly:
                raise ValueError(f"{geom} contains an anomaly; not a false-positive case")
            trials = [
                evaluate_once(factory, make_signal(geom, n=n, noise=noise, seed=seed))
                for seed in range(repeats)
            ]
            rows.append(
                {
                    "detector": det_name,
                    "generator": geom,
                    "repeats": int(repeats),
                    "series_false_positive_rate": float(np.mean([t["fired"] for t in trials])),
                    "point_false_positive_rate": float(
                        np.mean([t["point_fire_rate"] for t in trials])
                    ),
                    "mean_false_events": float(np.mean([t["n_pred_events"] for t in trials])),
                    "mean_max_score": float(np.mean([t["max_score"] for t in trials])),
                    "threshold": float(np.mean([t["threshold"] for t in trials])),
                }
            )
    return rows


def ramp_ceiling_table(
    magnitudes: Iterable[float] = (2.0, 5.0, 10.0, 100.0, 1000.0),
    n: int = 1001,
    baseline_frac: float = 0.0,
) -> list[dict[str, float]]:
    """Return measured ``max|z|`` for a noiseless ramp at several magnitudes.

    Used to produce the numeric verification table in ``paper/ramp_blindness.md``
    directly from a real run rather than from the algebra.

    Args:
        magnitudes: Final level as a multiple of baseline.
        n: Series length. ``n % 4 == 1`` makes the discrete MAD land exactly on
            R/4, so the measured constant is exact rather than approximate.
        baseline_frac: Leading flat fraction. The constant is invariant for any
            value at or below 0.25.

    Returns:
        One row per magnitude with the measured ceiling and its residual
        against the analytic constant.
    """
    from .detectors import RAMP_BLINDNESS_CONSTANT, MADZScore
    from .signals import linear_ramp

    rows: list[dict[str, float]] = []
    for mag in magnitudes:
        signal = linear_ramp(n=n, magnitude=mag, noise=0.0, seed=0, baseline_frac=baseline_frac)
        det = MADZScore()
        det.fit(signal.values)
        max_z = float(np.max(det.score(signal.values)))
        rows.append(
            {
                "magnitude": float(mag),
                "ramp_range": float(mag - 1.0),
                "max_abs_z": max_z,
                "analytic_constant": RAMP_BLINDNESS_CONSTANT,
                "abs_residual": abs(max_z - RAMP_BLINDNESS_CONSTANT),
                "fires_at_threshold_3": float(max_z > 3.0),
            }
        )
    return rows
