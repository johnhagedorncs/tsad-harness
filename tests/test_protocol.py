"""Tests for the scoring functions and the sweep protocol."""

from __future__ import annotations

import numpy as np
import pytest

from tsadharness.detectors import MADReference, MADZScore, RandomScore
from tsadharness.protocol import (
    detected,
    evaluate_once,
    event_wise,
    false_positive_table,
    find_events,
    fired,
    magnitude_sweep,
    point_adjust,
    point_adjusted,
    point_wise,
)
from tsadharness.signals import linear_ramp, pure_noise, step_change


def _mask(*bits: int) -> np.ndarray:
    return np.array(bits, dtype=bool)


def test_find_events_on_an_empty_mask() -> None:
    """No events in an all-False mask."""
    assert find_events(np.zeros(10, dtype=bool)) == []


def test_find_events_finds_contiguous_runs() -> None:
    """Half-open intervals, one per run."""
    mask = _mask(0, 1, 1, 0, 0, 1, 0, 1, 1, 1)
    assert find_events(mask) == [(1, 3), (5, 6), (7, 10)]


def test_find_events_handles_runs_touching_both_edges() -> None:
    """Runs at index 0 and at the final index are still found."""
    assert find_events(_mask(1, 1, 0, 1)) == [(0, 2), (3, 4)]
    assert find_events(np.ones(4, dtype=bool)) == [(0, 4)]


def test_point_wise_perfect_prediction() -> None:
    """Exact agreement scores 1.0 on all three components."""
    truth = _mask(0, 1, 1, 0)
    scores = point_wise(truth, truth)
    assert scores.precision == 1.0 and scores.recall == 1.0 and scores.f1 == 1.0
    assert (scores.tp, scores.fp, scores.fn) == (2, 0, 0)


def test_point_wise_counts_are_correct() -> None:
    """tp/fp/fn are the plain confusion-matrix counts."""
    truth = _mask(0, 1, 1, 0, 0)
    pred = _mask(0, 1, 0, 1, 0)
    scores = point_wise(truth, pred)
    assert (scores.tp, scores.fp, scores.fn) == (1, 1, 1)
    assert scores.precision == pytest.approx(0.5)
    assert scores.recall == pytest.approx(0.5)


def test_point_wise_empty_prediction_scores_zero() -> None:
    """Predicting nothing earns nothing, and does not divide by zero."""
    scores = point_wise(_mask(0, 1, 1), _mask(0, 0, 0))
    assert scores.precision == 0.0 and scores.recall == 0.0 and scores.f1 == 0.0


def test_point_wise_rejects_shape_mismatch() -> None:
    """Misaligned masks are a bug, not something to broadcast around."""
    with pytest.raises(ValueError, match="shape mismatch"):
        point_wise(np.zeros(5, dtype=bool), np.zeros(4, dtype=bool))


def test_point_adjust_expands_a_single_hit_to_the_whole_event() -> None:
    """One point inside an event marks the entire event. The inflation itself."""
    truth = _mask(0, 1, 1, 1, 1, 0)
    pred = _mask(0, 0, 0, 1, 0, 0)
    np.testing.assert_array_equal(point_adjust(truth, pred), _mask(0, 1, 1, 1, 1, 0))


def test_point_adjust_leaves_untouched_events_alone() -> None:
    """An event with no predicted point inside stays unpredicted."""
    truth = _mask(1, 1, 0, 1, 1)
    pred = _mask(1, 0, 0, 0, 0)
    np.testing.assert_array_equal(point_adjust(truth, pred), _mask(1, 1, 0, 0, 0))


def test_point_adjust_preserves_false_positives_outside_events() -> None:
    """The transform never removes a prediction; it only adds."""
    truth = _mask(0, 1, 1, 0, 0)
    pred = _mask(1, 0, 1, 0, 0)
    adjusted = point_adjust(truth, pred)
    assert adjusted[0]
    assert adjusted[1] and adjusted[2]


def test_point_adjusted_f1_is_never_below_point_f1_on_a_hit() -> None:
    """Point-adjust can only inflate recall, never deflate it."""
    truth = np.zeros(100, dtype=bool)
    truth[20:80] = True
    pred = np.zeros(100, dtype=bool)
    pred[50] = True
    assert point_adjusted(truth, pred).f1 > point_wise(truth, pred).f1


def test_point_adjusted_f1_is_near_perfect_from_one_lucky_point() -> None:
    """A single point in a 60-sample event yields F1 = 1.0. This is the flaw."""
    truth = np.zeros(100, dtype=bool)
    truth[20:80] = True
    pred = np.zeros(100, dtype=bool)
    pred[50] = True
    assert point_adjusted(truth, pred).f1 == pytest.approx(1.0)
    assert point_wise(truth, pred).f1 < 0.05


def test_random_scoring_beats_a_good_detector_under_point_adjust() -> None:
    """The concrete illusion-of-progress demonstration, asserted numerically.

    A uniform random scorer earns a near-perfect point-adjusted F1 on a long
    event while its honest point-wise F1 is near zero. Any leaderboard ranked
    on point-adjusted F1 is therefore not measuring detection ability.
    """
    signal = step_change(n=2000, magnitude=5.0, noise=0.02, seed=0, baseline_frac=0.5)
    det = RandomScore(alpha=0.01, seed=0)
    det.fit(signal.values)
    pred = det.detect(signal.values)
    adjusted = point_adjusted(signal.ground_truth, pred).f1
    honest = point_wise(signal.ground_truth, pred).f1
    assert adjusted > 0.9
    assert honest < 0.1


def test_event_wise_credits_an_overlapping_prediction() -> None:
    """Any overlap finds the event."""
    truth = _mask(0, 1, 1, 1, 0)
    pred = _mask(0, 0, 1, 0, 0)
    scores = event_wise(truth, pred)
    assert scores.tp == 1 and scores.fn == 0 and scores.fp == 0
    assert scores.recall == 1.0


def test_event_wise_counts_a_missed_event() -> None:
    """No overlap means a false negative."""
    truth = _mask(0, 1, 1, 0, 0)
    pred = _mask(0, 0, 0, 0, 1)
    scores = event_wise(truth, pred)
    assert scores.tp == 0 and scores.fn == 1 and scores.fp == 1


def test_event_wise_charges_one_false_positive_per_spurious_run() -> None:
    """A long spurious run costs one event, not one point per sample."""
    truth = np.zeros(20, dtype=bool)
    pred = np.zeros(20, dtype=bool)
    pred[5:15] = True
    assert event_wise(truth, pred).fp == 1


def test_event_wise_and_point_wise_disagree_on_partial_coverage() -> None:
    """The three metrics are genuinely different, which is why all three ship."""
    truth = np.zeros(100, dtype=bool)
    truth[20:80] = True
    pred = np.zeros(100, dtype=bool)
    pred[20:30] = True
    assert event_wise(truth, pred).recall == 1.0
    assert point_wise(truth, pred).recall == pytest.approx(10 / 60)


def test_detected_requires_overlap_with_ground_truth() -> None:
    """detected() is the sweep's y-axis; it must mean what it says."""
    signal = step_change(n=200, magnitude=5.0, noise=0.0, seed=0, baseline_frac=0.8)
    inside = np.zeros(200, dtype=bool)
    inside[180] = True
    outside = np.zeros(200, dtype=bool)
    outside[10] = True
    assert detected(signal, inside)
    assert not detected(signal, outside)


def test_detected_rejects_a_signal_without_an_anomaly() -> None:
    """Asking whether a no-anomaly series was detected is a category error."""
    with pytest.raises(ValueError, match="no anomaly"):
        detected(pure_noise(n=100, seed=0), np.ones(100, dtype=bool))


def test_fired_reports_any_firing() -> None:
    """fired() is the false-positive table's unit of account."""
    assert fired(_mask(0, 0, 1))
    assert not fired(_mask(0, 0, 0))


def test_evaluate_once_returns_the_expected_fields() -> None:
    """The per-trial record carries everything the sweep aggregates."""
    signal = step_change(n=500, magnitude=5.0, noise=0.02, seed=0)
    row = evaluate_once(lambda: MADReference(reference_fraction=0.2), signal)
    for key in ("max_score", "threshold", "detected", "point_f1", "event_f1", "point_adjusted_f1"):
        assert key in row
    assert row["detected"] == 1.0


def test_evaluate_once_omits_detection_fields_for_no_anomaly_signals() -> None:
    """No ground-truth event means no recall to report."""
    row = evaluate_once(lambda: MADReference(), pure_noise(n=300, seed=0))
    assert "detected" not in row
    assert "fired" in row


def test_magnitude_sweep_covers_every_cell() -> None:
    """One row per detector x geometry x magnitude."""
    detectors = {"z": lambda: MADZScore(), "ref": lambda: MADReference(reference_fraction=0.2)}
    rows = magnitude_sweep(
        detectors=detectors,
        geometries=["linear_ramp", "step_change"],
        magnitudes=[2.0, 10.0],
        repeats=2,
        n=400,
    )
    assert len(rows) == 2 * 2 * 2
    assert {r["detector"] for r in rows} == {"z", "ref"}


def test_magnitude_sweep_detection_rates_are_probabilities() -> None:
    """Rates stay inside [0, 1]."""
    rows = magnitude_sweep(
        detectors={"ref": lambda: MADReference(reference_fraction=0.2)},
        geometries=["linear_ramp"],
        magnitudes=[3.0],
        repeats=3,
        n=400,
    )
    assert 0.0 <= float(rows[0]["detection_rate"]) <= 1.0


def test_sweep_reproduces_the_headline_contrast() -> None:
    """The whole repository in one assertion.

    Across three orders of magnitude of ramp size, the self-referencing rule
    never fires and the reference-window rule always does.
    """
    rows = magnitude_sweep(
        detectors={
            "MADZScore": lambda: MADZScore(threshold=3.0),
            "MADReference": lambda: MADReference(threshold=3.0, reference_fraction=0.2),
        },
        geometries=["linear_ramp"],
        magnitudes=[2.0, 10.0, 100.0, 1000.0],
        repeats=3,
        n=1000,
    )
    broken = [r for r in rows if r["detector"] == "MADZScore"]
    fixed = [r for r in rows if r["detector"] == "MADReference"]
    assert all(float(r["detection_rate"]) == 0.0 for r in broken)
    assert all(float(r["detection_rate"]) == 1.0 for r in fixed)


def test_sweep_records_the_flat_ceiling() -> None:
    """mean_max_score for the broken rule barely moves across magnitudes."""
    rows = magnitude_sweep(
        detectors={"MADZScore": lambda: MADZScore()},
        geometries=["linear_ramp"],
        magnitudes=[2.0, 10.0, 100.0, 1000.0],
        repeats=2,
        n=1000,
        noise=0.0,
    )
    ceilings = [float(r["mean_max_score"]) for r in rows]
    assert max(ceilings) - min(ceilings) < 0.01


def test_sweep_rejects_an_unknown_geometry() -> None:
    """Typos fail loudly."""
    with pytest.raises(KeyError, match="unknown generator"):
        magnitude_sweep(
            detectors={"z": lambda: MADZScore()},
            geometries=["not_a_generator"],
            magnitudes=[2.0],
            repeats=1,
        )


def test_false_positive_table_covers_both_no_anomaly_geometries() -> None:
    """One row per detector per no-anomaly geometry."""
    rows = false_positive_table(
        detectors={"ref": lambda: MADReference(reference_fraction=0.2)}, repeats=5, n=400
    )
    assert {r["generator"] for r in rows} == {"pure_noise", "regime_switch"}
    for row in rows:
        assert 0.0 <= float(row["series_false_positive_rate"]) <= 1.0


def test_false_positive_table_rejects_a_geometry_with_an_anomaly() -> None:
    """Guards against silently reporting recall as a false-positive rate."""
    with pytest.raises(ValueError, match="not a false-positive case"):
        false_positive_table(
            detectors={"ref": lambda: MADReference()},
            generators=["linear_ramp"],
            repeats=2,
            n=200,
        )


def test_fixed_threshold_is_silent_on_pure_noise() -> None:
    """At least one detector posts a genuine zero, so the table has a floor."""
    from tsadharness.detectors import FixedThreshold

    rows = false_positive_table(
        detectors={"fixed": lambda: FixedThreshold(level=2.0)},
        generators=["pure_noise"],
        repeats=10,
        n=1000,
    )
    assert float(rows[0]["series_false_positive_rate"]) == 0.0


def test_random_scoring_has_a_high_false_positive_rate() -> None:
    """The other half of the honesty check: recall bought with false alarms."""
    rows = false_positive_table(
        detectors={"random": lambda: RandomScore(alpha=0.01, seed=0)},
        generators=["pure_noise"],
        repeats=10,
        n=1000,
    )
    assert float(rows[0]["series_false_positive_rate"]) > 0.9


def test_ramp_is_invisible_to_the_broken_rule_even_with_gaps() -> None:
    """Ramp blindness survives irregular sampling."""
    from tsadharness.signals import ramp_with_gaps

    signal = ramp_with_gaps(n=1000, magnitude=50.0, noise=0.02, seed=0)
    det = MADZScore(threshold=3.0)
    det.fit(signal.values)
    assert not detected(signal, det.detect(signal.values))


def test_linear_ramp_sweep_beats_nothing_at_any_magnitude() -> None:
    """Direct statement of the negative result at extreme magnitude."""
    signal = linear_ramp(n=1000, magnitude=1e6, noise=0.02, seed=0)
    det = MADZScore(threshold=3.0)
    det.fit(signal.values)
    assert float(np.max(det.score(signal.values))) < 3.0
