"""Protocol conformance and behaviour tests for every detector."""

from __future__ import annotations

import numpy as np
import pytest

from tsadharness.detectors import (
    STUMPY_AVAILABLE,
    AEReconstruction,
    Detector,
    FixedThreshold,
    MADReference,
    MADZScore,
    RandomScore,
    default_detectors,
    interpolate_nans,
    scaled_mad,
    sliding_windows,
)
from tsadharness.signals import linear_ramp, pure_noise, ramp_with_gaps, spikes, step_change

FACTORIES = default_detectors()
FACTORY_IDS = sorted(FACTORIES)


def _factory(name: str):
    return FACTORIES[name]


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_detector_satisfies_the_protocol(name: str) -> None:
    """Every detector is a structural Detector."""
    det = _factory(name)()
    assert isinstance(det, Detector)
    assert isinstance(det.name, str) and det.name


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_score_returns_one_finite_score_per_point(name: str) -> None:
    """Scores are float, finite, and aligned to the input length."""
    signal = pure_noise(n=400, noise=0.02, seed=0)
    det = _factory(name)()
    det.fit(signal.values)
    scores = det.score(signal.values)
    assert scores.shape == signal.values.shape
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_detect_returns_a_boolean_mask(name: str) -> None:
    """detect() is a mask of the right dtype and length."""
    signal = pure_noise(n=400, noise=0.02, seed=0)
    det = _factory(name)()
    det.fit(signal.values)
    flags = det.detect(signal.values)
    assert flags.dtype == np.bool_
    assert flags.shape == signal.values.shape


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_scoring_is_deterministic(name: str) -> None:
    """The same fitted detector scoring the same series twice agrees exactly."""
    signal = linear_ramp(n=400, magnitude=3.0, noise=0.02, seed=0)
    det = _factory(name)()
    det.fit(signal.values)
    np.testing.assert_array_equal(det.score(signal.values), det.score(signal.values))


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_two_instances_from_the_factory_agree(name: str) -> None:
    """Factories produce interchangeable detectors; no hidden global state."""
    signal = linear_ramp(n=400, magnitude=3.0, noise=0.02, seed=0)
    a, b = _factory(name)(), _factory(name)()
    a.fit(signal.values)
    b.fit(signal.values)
    np.testing.assert_allclose(a.score(signal.values), b.score(signal.values))


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_handles_gaps_without_raising(name: str) -> None:
    """NaN input is handled by a declared policy rather than crashing."""
    signal = ramp_with_gaps(n=600, magnitude=5.0, noise=0.02, seed=0)
    det = _factory(name)()
    det.fit(signal.values)
    scores = det.score(signal.values)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("name", FACTORY_IDS)
def test_handles_a_constant_series_without_dividing_by_zero(name: str) -> None:
    """Zero spread must not produce inf or NaN scores."""
    values = np.full(300, 2.5)
    det = _factory(name)()
    det.fit(values)
    assert np.isfinite(det.score(values)).all()


@pytest.mark.parametrize("name", ["MADZScore", "MADReference", "AEReconstruction"])
def test_score_before_fit_raises(name: str) -> None:
    """Calling score() first is a programming error and says so."""
    det = _factory(name)()
    with pytest.raises(RuntimeError, match="fit"):
        det.score(np.zeros(100))


def test_mad_z_uses_the_scored_series_not_the_fitted_one() -> None:
    """MADZScore has no memory: fitting elsewhere changes nothing."""
    ramp = linear_ramp(n=1000, magnitude=10.0, noise=0.0, seed=0).values
    flat = pure_noise(n=1000, noise=0.02, seed=0).values
    det = MADZScore()
    det.fit(flat)
    from_flat_fit = det.score(ramp)
    det.fit(ramp)
    from_ramp_fit = det.score(ramp)
    np.testing.assert_allclose(from_flat_fit, from_ramp_fit)


def test_mad_reference_remembers_its_window() -> None:
    """MADReference does have memory: what it was fitted on matters."""
    ramp = linear_ramp(n=1000, magnitude=10.0, noise=0.02, seed=0).values
    a = MADReference(reference_fraction=0.2)
    a.fit(ramp)
    b = MADReference(reference_fraction=1.0)
    b.fit(ramp)
    assert float(np.max(a.score(ramp))) > float(np.max(b.score(ramp)))


def test_mad_reference_records_the_window_it_used() -> None:
    """The declared window is inspectable after fitting."""
    det = MADReference(reference_fraction=0.25)
    det.fit(np.arange(400, dtype=float))
    assert det.window_ == (0, 100)
    assert det.center_ is not None and det.scale_ is not None


def test_mad_reference_accepts_an_explicit_slice() -> None:
    """An explicit slice overrides the fraction."""
    det = MADReference(reference_slice=(50, 150))
    det.fit(np.arange(400, dtype=float))
    assert det.window_ == (50, 150)


def test_mad_reference_validates_its_arguments() -> None:
    """Bad windows are rejected at construction."""
    with pytest.raises(ValueError, match="reference_fraction"):
        MADReference(reference_fraction=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        MADReference(reference_slice=(100, 100))


def test_fixed_threshold_fires_exactly_at_its_level() -> None:
    """No statistics involved: the rule is the number."""
    det = FixedThreshold(level=2.0)
    values = np.array([1.0, 1.9, 2.0, 2.1, 5.0])
    det.fit(values)
    np.testing.assert_array_equal(det.detect(values), [False, False, False, True, True])


def test_fixed_threshold_never_fires_below_its_level() -> None:
    """On a quiet baseline it is silent, which is its one virtue."""
    det = FixedThreshold(level=2.0)
    signal = pure_noise(n=5000, noise=0.02, seed=0)
    det.fit(signal.values)
    assert not det.detect(signal.values).any()


def test_random_score_is_reproducible() -> None:
    """The honesty check must itself be reproducible."""
    det = RandomScore(alpha=0.01, seed=42)
    values = np.zeros(500)
    det.fit(values)
    np.testing.assert_array_equal(det.score(values), det.score(values))


def test_random_score_ignores_the_data_entirely() -> None:
    """Identical scores for wildly different series. That is the point."""
    det = RandomScore(alpha=0.01, seed=42)
    a = pure_noise(n=500, seed=0).values
    b = linear_ramp(n=500, magnitude=1000.0, seed=0).values
    det.fit(a)
    np.testing.assert_array_equal(det.score(a), det.score(b))


def test_random_score_fires_at_about_its_nominal_rate() -> None:
    """The nominal alpha is the realised per-point firing rate."""
    det = RandomScore(alpha=0.05, seed=1)
    values = np.zeros(20_000)
    det.fit(values)
    assert float(np.mean(det.detect(values))) == pytest.approx(0.05, abs=0.01)


def test_random_score_validates_alpha() -> None:
    """alpha must be a probability."""
    with pytest.raises(ValueError, match="alpha"):
        RandomScore(alpha=1.5)


def test_random_score_almost_always_finds_a_long_event() -> None:
    """1 - (1-alpha)**L approaches 1: random guessing "detects" long events.

    This is the mechanism behind inflated benchmark scores, so it is asserted
    rather than merely asserted in prose.
    """
    det = RandomScore(alpha=0.01, seed=0)
    signal = step_change(n=1000, magnitude=5.0, noise=0.02, seed=0, baseline_frac=0.8)
    det.fit(signal.values)
    assert det.detect(signal.values)[signal.ground_truth].any()


def test_autoencoder_reconstructs_its_training_window_well() -> None:
    """Training error is far below the anomaly threshold it calibrates."""
    signal = pure_noise(n=1500, noise=0.02, seed=0)
    det = AEReconstruction(seed=0, epochs=200)
    det.fit(signal.values)
    scores = det.score(signal.values)
    assert float(np.median(scores)) < det.threshold


def test_autoencoder_training_loss_decreases() -> None:
    """Gradient descent actually descends."""
    signal = pure_noise(n=1500, noise=0.02, seed=0)
    det = AEReconstruction(seed=0, epochs=300)
    det.fit(signal.values)
    assert det.train_loss_[-1] < det.train_loss_[0]


def test_autoencoder_flags_a_large_step() -> None:
    """It finds the geometry it is meant to find."""
    signal = step_change(n=1500, magnitude=10.0, noise=0.02, seed=0, baseline_frac=0.8)
    det = AEReconstruction(seed=0)
    det.fit(signal.values)
    assert det.detect(signal.values)[signal.ground_truth].any()


def test_autoencoder_is_deterministic_under_a_seed() -> None:
    """Same seed, same weights, same scores."""
    signal = linear_ramp(n=800, magnitude=4.0, noise=0.02, seed=0)
    a = AEReconstruction(seed=3, epochs=100)
    b = AEReconstruction(seed=3, epochs=100)
    a.fit(signal.values)
    b.fit(signal.values)
    np.testing.assert_allclose(a.score(signal.values), b.score(signal.values))


@pytest.mark.parametrize(
    "series_name", ["pure_noise", "linear_ramp", "step_change", "ramp_with_gaps"]
)
def test_autoencoder_weights_stay_finite(series_name: str) -> None:
    """Training never diverges on any harness geometry.

    The forward and backward passes suppress spurious floating-point status
    flags from NumPy's vectorised matmul, so this asserts the invariant that
    suppression could otherwise hide.
    """
    from tsadharness.signals import make_signal

    signal = make_signal(series_name, n=1000, magnitude=100.0, noise=0.02, seed=0)
    det = AEReconstruction(seed=0)
    det.fit(signal.values)
    for param in (det.w1_, det.b1_, det.w2_, det.b2_):
        assert np.isfinite(param).all()
    assert np.isfinite(det.train_loss_).all()
    assert np.isfinite(det.score(signal.values)).all()


def test_autoencoder_rejects_a_too_short_reference_window() -> None:
    """A window shorter than the model window is an error, not a silent pad."""
    det = AEReconstruction(window=64, reference_fraction=0.01)
    with pytest.raises(ValueError, match="too short"):
        det.fit(np.zeros(50))


def test_sliding_windows_shape_and_content() -> None:
    """The window matrix is right, since every learned detector rests on it."""
    values = np.arange(10, dtype=float)
    windows = sliding_windows(values, 4)
    assert windows.shape == (7, 4)
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[-1], [6, 7, 8, 9])


def test_sliding_windows_validates_its_input() -> None:
    """Degenerate windows are rejected."""
    with pytest.raises(ValueError, match="window must be"):
        sliding_windows(np.zeros(10), 1)
    with pytest.raises(ValueError, match="exceeds"):
        sliding_windows(np.zeros(10), 20)


def test_interpolate_nans_fills_interior_and_edges() -> None:
    """Gap policy is explicit and total."""
    values = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    filled = interpolate_nans(values)
    assert np.isfinite(filled).all()
    assert filled[2] == pytest.approx(2.0)
    assert filled[0] == pytest.approx(1.0)
    assert filled[-1] == pytest.approx(3.0)


def test_interpolate_nans_handles_an_all_nan_series() -> None:
    """The degenerate case returns zeros rather than raising."""
    filled = interpolate_nans(np.full(10, np.nan))
    assert np.isfinite(filled).all()


def test_scaled_mad_matches_sigma_for_gaussian_data() -> None:
    """The 1.4826 factor does what it claims on normal data."""
    values = np.random.default_rng(0).normal(0.0, 3.0, 200_000)
    assert scaled_mad(values) == pytest.approx(3.0, rel=0.02)


def test_scaled_mad_never_returns_zero() -> None:
    """A constant series yields a positive floor, protecting every divide."""
    assert scaled_mad(np.ones(100)) > 0.0


def test_spikes_are_found_by_the_reference_detector() -> None:
    """Sanity: the easy geometry is easy."""
    signal = spikes(n=1000, magnitude=10.0, noise=0.02, seed=0)
    det = MADReference(reference_fraction=0.2)
    det.fit(signal.values)
    assert det.detect(signal.values)[signal.ground_truth].any()


@pytest.mark.skipif(not STUMPY_AVAILABLE, reason="stumpy is not installed")
def test_matrix_profile_scores_align_with_the_series() -> None:
    """When stumpy is present the detector conforms like any other."""
    from tsadharness.detectors import MatrixProfileDetector

    signal = step_change(n=600, magnitude=5.0, noise=0.02, seed=0)
    det = MatrixProfileDetector(window=32)
    det.fit(signal.values)
    scores = det.score(signal.values)
    assert scores.shape == signal.values.shape
    assert np.isfinite(scores).all()


@pytest.mark.skipif(not STUMPY_AVAILABLE, reason="stumpy is not installed")
def test_matrix_profile_is_lower_inside_a_ramp_than_on_flat_noise() -> None:
    """z-normalisation makes a ramp look *more* self-similar, not less.

    Documents the second, independent reason this detector cannot see ramps.
    """
    from tsadharness.detectors import MatrixProfileDetector

    ramp = linear_ramp(n=1000, magnitude=10.0, noise=0.02, seed=0)
    flat = pure_noise(n=1000, noise=0.02, seed=0)
    det = MatrixProfileDetector(window=32)
    det.fit(ramp.values)
    ramp_median = float(np.median(det.score(ramp.values)))
    det.fit(flat.values)
    flat_median = float(np.median(det.score(flat.values)))
    assert ramp_median < flat_median


@pytest.mark.skipif(STUMPY_AVAILABLE, reason="stumpy is installed")
def test_matrix_profile_raises_a_clear_error_without_stumpy() -> None:
    """Without stumpy the detector explains itself instead of crashing obscurely."""
    from tsadharness.detectors import MatrixProfileDetector

    with pytest.raises(ImportError, match="stumpy"):
        MatrixProfileDetector()


def test_default_detector_set_always_includes_the_random_baseline() -> None:
    """RandomScore is not optional; it anchors every table."""
    assert "RandomScore" in default_detectors()
    assert "RandomScore" in default_detectors(include_matrix_profile=False)


def test_matrix_profile_can_be_excluded() -> None:
    """The default set degrades gracefully when stumpy is unwanted."""
    assert "MatrixProfile" not in default_detectors(include_matrix_profile=False)
