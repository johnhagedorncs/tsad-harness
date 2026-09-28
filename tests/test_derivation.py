"""Tests for the central analytic result: ramp blindness.

If any test in this file fails, the claim the repository is built on is wrong
and the paper needs rewriting. These are the load-bearing tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsadharness.detectors import MAD_SCALE, RAMP_BLINDNESS_CONSTANT, MADReference, MADZScore
from tsadharness.protocol import ramp_ceiling_table
from tsadharness.signals import linear_ramp

# n % 4 == 1 makes the discrete median of absolute deviations land exactly on
# R/4 rather than one grid step either side of it, so the measured ceiling
# equals the analytic constant to floating-point precision.
EXACT_N = 1001


def _max_abs_z(values: np.ndarray) -> float:
    det = MADZScore()
    det.fit(values)
    return float(np.max(det.score(values)))


def test_constant_matches_closed_form() -> None:
    """The exported constant is exactly 0.5 / (0.25 * 1.4826)."""
    assert pytest.approx(0.5 / (0.25 * MAD_SCALE), abs=1e-15) == RAMP_BLINDNESS_CONSTANT


def test_constant_numeric_value() -> None:
    """The constant is 1.3490 to four decimal places."""
    assert pytest.approx(1.3490, abs=5e-5) == RAMP_BLINDNESS_CONSTANT


@pytest.mark.parametrize("magnitude_range", [2.0, 5.0, 10.0, 100.0, 1000.0])
def test_max_z_is_independent_of_ramp_magnitude(magnitude_range: float) -> None:
    """max|z| on a pure ramp equals the constant to 1e-6, for every range R.

    This is the headline claim: a 1000x ramp is exactly as invisible as a 2x
    ramp, because R cancels between the numerator and the MAD.
    """
    values = np.linspace(0.0, magnitude_range, EXACT_N)
    assert _max_abs_z(values) == pytest.approx(RAMP_BLINDNESS_CONSTANT, abs=1e-6)


def test_max_z_identical_across_all_magnitudes() -> None:
    """The ceilings for different R are equal to each other, not merely close."""
    ceilings = [_max_abs_z(np.linspace(0.0, r, EXACT_N)) for r in (2.0, 5.0, 10.0, 100.0, 1000.0)]
    assert max(ceilings) - min(ceilings) < 1e-9


@pytest.mark.parametrize("offset", [-500.0, 0.0, 3.7, 1e6])
def test_max_z_is_independent_of_ramp_offset(offset: float) -> None:
    """Shifting the whole ramp leaves the ceiling unchanged (it is location free)."""
    values = np.linspace(0.0, 10.0, EXACT_N) + offset
    assert _max_abs_z(values) == pytest.approx(RAMP_BLINDNESS_CONSTANT, abs=1e-6)


def test_descending_ramp_has_the_same_ceiling() -> None:
    """A falling ramp is as invisible as a rising one; the score uses |z|."""
    assert _max_abs_z(np.linspace(10.0, 0.0, EXACT_N)) == pytest.approx(
        RAMP_BLINDNESS_CONSTANT, abs=1e-6
    )


@pytest.mark.parametrize("baseline_frac", [0.0, 0.05, 0.10, 0.20, 0.25])
def test_ceiling_holds_for_baseline_fractions_up_to_one_quarter(baseline_frac: float) -> None:
    """The constant survives a leading flat baseline of up to 1/4 of the series.

    A generalisation of the basic result. With a leading fraction b of flat
    baseline, the median sits at R(0.5-b)/(1-b) and the MAD at 0.25R/(1-b);
    the (1-b) factors cancel along with R, so the ceiling is unchanged for
    every b <= 1/4.
    """
    n = 200_001
    n_base = int(n * baseline_frac)
    values = np.concatenate([np.zeros(n_base), np.linspace(0.0, 10.0, n - n_base)])
    assert _max_abs_z(values) == pytest.approx(RAMP_BLINDNESS_CONSTANT, abs=1e-3)


@pytest.mark.parametrize("baseline_frac", [0.30, 0.40])
def test_ceiling_breaks_down_above_one_quarter_baseline(baseline_frac: float) -> None:
    """Above b = 1/4 the ceiling rises. The claim has a stated boundary.

    This is a negative test for my own result: the blindness is not universal,
    it holds while the ramp occupies at least three quarters of the scored
    window. Beyond that the flat baseline starts to dominate the MAD and the
    ramp becomes visible again.
    """
    n = 200_001
    n_base = int(n * baseline_frac)
    values = np.concatenate([np.zeros(n_base), np.linspace(0.0, 10.0, n - n_base)])
    assert _max_abs_z(values) > RAMP_BLINDNESS_CONSTANT * 1.05


def test_threshold_of_three_is_unreachable_on_a_ramp() -> None:
    """The conventional 3-sigma rule can never fire on a ramp, at any magnitude."""
    for magnitude in (2.0, 10.0, 100.0, 1000.0, 1e6):
        signal = linear_ramp(n=1000, magnitude=magnitude, noise=0.0, seed=0, baseline_frac=0.2)
        det = MADZScore(threshold=3.0)
        det.fit(signal.values)
        assert not det.detect(signal.values).any()


def test_any_threshold_above_the_constant_is_unreachable() -> None:
    """Not just 3.0: every threshold above the ceiling is unreachable."""
    values = np.linspace(0.0, 42.0, EXACT_N)
    for threshold in (1.35, 1.5, 2.0, 3.0, 5.0):
        det = MADZScore(threshold=threshold)
        det.fit(values)
        assert not det.detect(values).any()


def test_threshold_below_the_constant_does_fire() -> None:
    """The ceiling is a ceiling, not a floor: below it the rule still fires."""
    values = np.linspace(0.0, 42.0, EXACT_N)
    det = MADZScore(threshold=1.30)
    det.fit(values)
    assert det.detect(values).any()


def test_reference_window_fixes_the_ramp() -> None:
    """The structural fix: a declared stationary window makes the ramp visible."""
    signal = linear_ramp(n=1000, magnitude=3.0, noise=0.02, seed=0, baseline_frac=0.2)
    fixed = MADReference(threshold=3.0, reference_fraction=0.2)
    fixed.fit(signal.values)
    assert fixed.detect(signal.values)[signal.ground_truth].any()


def test_reference_window_scales_with_magnitude() -> None:
    """Unlike the self-referencing rule, the fix's max score grows with R."""
    scores = []
    for magnitude in (2.0, 10.0, 100.0):
        signal = linear_ramp(n=1000, magnitude=magnitude, noise=0.02, seed=0, baseline_frac=0.2)
        det = MADReference(threshold=3.0, reference_fraction=0.2)
        det.fit(signal.values)
        scores.append(float(np.max(det.score(signal.values))))
    assert scores[0] < scores[1] < scores[2]


def test_self_referencing_max_score_does_not_scale_with_magnitude() -> None:
    """The contrast: the broken rule's max score is flat across magnitudes."""
    scores = []
    for magnitude in (2.0, 10.0, 100.0):
        signal = linear_ramp(n=1000, magnitude=magnitude, noise=0.0, seed=0, baseline_frac=0.2)
        det = MADZScore(threshold=3.0)
        det.fit(signal.values)
        scores.append(float(np.max(det.score(signal.values))))
    assert max(scores) - min(scores) < 1e-6


def test_ramp_ceiling_table_reports_zero_residuals() -> None:
    """The table used to build the paper agrees with the algebra."""
    rows = ramp_ceiling_table(magnitudes=(2.0, 5.0, 10.0, 100.0, 1000.0), n=EXACT_N)
    assert len(rows) == 5
    for row in rows:
        assert row["abs_residual"] < 1e-6
        assert row["fires_at_threshold_3"] == 0.0


def test_step_is_visible_to_the_same_broken_rule() -> None:
    """The rule is not broken in general, only on ramps.

    A step occupying a minority of the window is caught easily, which is what
    makes ramp blindness a geometry-specific defect rather than a bad detector.
    """
    from tsadharness.signals import step_change

    signal = step_change(n=1000, magnitude=3.0, noise=0.02, seed=0, baseline_frac=0.8)
    det = MADZScore(threshold=3.0)
    det.fit(signal.values)
    assert det.detect(signal.values)[signal.ground_truth].any()


def test_step_becomes_invisible_at_exactly_half_contamination() -> None:
    """The median/MAD breakdown point: a 50/50 step is invisible too.

    A corollary worth stating precisely, because it bounds the positive
    control. For a step the breakdown fraction is 1/2; for a ramp it is 1/4.
    """
    n = 100_000
    values = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 4.0)])
    assert _max_abs_z(values) < 3.0
