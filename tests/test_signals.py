"""Property tests for the nine signal generators."""

from __future__ import annotations

import numpy as np
import pytest

from tsadharness.signals import (
    ANOMALOUS_GENERATORS,
    GENERATORS,
    NO_ANOMALY_GENERATORS,
    Signal,
    exponential_ramp,
    linear_ramp,
    make_signal,
    pure_noise,
    ramp_with_gaps,
    regime_switch,
    seasonal_plus_ramp,
    spikes,
    step_change,
    step_with_recovery,
)

ALL_NAMES = sorted(GENERATORS)
RAMP_NAMES = ["linear_ramp", "exponential_ramp", "seasonal_plus_ramp"]


def _block_means(values: np.ndarray, n_blocks: int = 10) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return np.array([np.mean(b) for b in np.array_split(finite, n_blocks)])


def test_there_are_exactly_nine_generators() -> None:
    """The harness advertises nine geometries; keep the registry honest."""
    assert len(GENERATORS) == 9


@pytest.mark.parametrize("name", ALL_NAMES)
def test_shapes_match(name: str) -> None:
    """values and ground_truth are the same length, and it is the requested n."""
    signal = make_signal(name, n=500, magnitude=3.0, noise=0.01, seed=1)
    assert signal.values.shape == (500,)
    assert signal.ground_truth.shape == (500,)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_reproducible_under_a_fixed_seed(name: str) -> None:
    """Two calls with the same seed produce bit-identical output."""
    a = make_signal(name, n=400, magnitude=4.0, noise=0.03, seed=7)
    b = make_signal(name, n=400, magnitude=4.0, noise=0.03, seed=7)
    np.testing.assert_array_equal(
        np.nan_to_num(a.values, nan=-999), np.nan_to_num(b.values, nan=-999)
    )
    np.testing.assert_array_equal(a.ground_truth, b.ground_truth)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_different_seeds_give_different_series(name: str) -> None:
    """Seeds actually vary the realisation."""
    a = make_signal(name, n=400, magnitude=4.0, noise=0.03, seed=1)
    b = make_signal(name, n=400, magnitude=4.0, noise=0.03, seed=2)
    assert not np.allclose(np.nan_to_num(a.values, nan=-999), np.nan_to_num(b.values, nan=-999))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_generator_reports_its_own_name(name: str) -> None:
    """The name field matches the registry key, so result tables join correctly."""
    assert make_signal(name, n=200, seed=0).name == name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_metadata_records_the_seed(name: str) -> None:
    """Every generator records enough metadata to reproduce itself."""
    signal = make_signal(name, n=200, noise=0.01, seed=5)
    assert signal.metadata["seed"] == 5
    assert "shape" in signal.metadata


@pytest.mark.parametrize("name", ALL_NAMES)
def test_only_gap_generator_produces_nans(name: str) -> None:
    """NaNs are confined to the generator that advertises them."""
    signal = make_signal(name, n=300, seed=0)
    has_nan = bool(np.isnan(signal.values).any())
    assert has_nan == (name == "ramp_with_gaps")


@pytest.mark.parametrize("name", ANOMALOUS_GENERATORS)
def test_anomalous_generators_label_something(name: str) -> None:
    """Every generator outside the no-anomaly set marks at least one point."""
    signal = make_signal(name, n=500, magnitude=3.0, noise=0.01, seed=0)
    assert signal.has_anomaly
    assert signal.ground_truth.sum() > 0


@pytest.mark.parametrize("name", ANOMALOUS_GENERATORS)
def test_anomalous_generators_leave_some_normal_points(name: str) -> None:
    """A series that is entirely anomalous has no baseline to compare against."""
    signal = make_signal(name, n=500, magnitude=3.0, noise=0.01, seed=0)
    assert (~signal.ground_truth).sum() > 0


@pytest.mark.parametrize("name", NO_ANOMALY_GENERATORS)
def test_no_anomaly_generators_have_all_false_ground_truth(name: str) -> None:
    """pure_noise and regime_switch must never label anything.

    These two are what the false-positive table is built on. If either ever
    starts labelling points, every false-positive number in the repository
    silently becomes meaningless.
    """
    for seed in range(5):
        signal = make_signal(name, n=400, magnitude=3.0, noise=0.02, seed=seed)
        assert not signal.ground_truth.any()
        assert not signal.has_anomaly


@pytest.mark.parametrize("name", RAMP_NAMES)
def test_ramp_generators_actually_ramp(name: str) -> None:
    """The mean level rises monotonically across blocks of the ramp segment."""
    signal = make_signal(name, n=2000, magnitude=10.0, noise=0.0, seed=0)
    start = int(signal.metadata["anomaly_start"])
    means = _block_means(signal.values[start:], n_blocks=8)
    assert np.all(np.diff(means) > 0), f"{name} block means are not increasing: {means}"


@pytest.mark.parametrize("name", RAMP_NAMES)
def test_ramp_generators_end_higher_than_they_start(name: str) -> None:
    """End level reflects the requested magnitude."""
    signal = make_signal(name, n=2000, magnitude=10.0, noise=0.0, seed=0)
    assert np.mean(signal.values[-50:]) > np.mean(signal.values[:50]) * 3.0


def test_linear_ramp_is_linear() -> None:
    """A straight line: second differences vanish inside the ramp."""
    signal = linear_ramp(n=1000, magnitude=5.0, noise=0.0, seed=0, baseline_frac=0.2)
    start = int(signal.metadata["anomaly_start"])
    second = np.diff(signal.values[start + 1 :], n=2)
    assert np.allclose(second, 0.0, atol=1e-9)


def test_exponential_ramp_is_convex() -> None:
    """Geometric growth: second differences are strictly positive."""
    signal = exponential_ramp(n=1000, magnitude=50.0, noise=0.0, seed=0, baseline_frac=0.2)
    start = int(signal.metadata["anomaly_start"])
    second = np.diff(signal.values[start:], n=2)
    assert np.all(second > 0)


def test_ramps_share_endpoints_but_differ_in_shape() -> None:
    """Linear and exponential ramps end at the same level by different paths.

    This is what lets the harness separate "fails on this shape" from "fails at
    this scale".
    """
    lin = linear_ramp(n=1000, magnitude=8.0, noise=0.0, seed=0)
    exp = exponential_ramp(n=1000, magnitude=8.0, noise=0.0, seed=0)
    assert lin.values[-1] == pytest.approx(exp.values[-1], rel=1e-9)
    assert not np.allclose(lin.values, exp.values)


def test_step_actually_steps() -> None:
    """A single large jump, flat on both sides."""
    signal = step_change(n=1000, magnitude=4.0, noise=0.0, seed=0, baseline_frac=0.8)
    start = int(signal.metadata["anomaly_start"])
    jump = signal.values[start] - signal.values[start - 1]
    assert jump == pytest.approx(3.0, rel=1e-9)
    assert np.allclose(np.diff(signal.values[:start]), 0.0, atol=1e-9)
    assert np.allclose(np.diff(signal.values[start:]), 0.0, atol=1e-9)


def test_step_ground_truth_starts_at_the_step() -> None:
    """The label boundary coincides with the discontinuity."""
    signal = step_change(n=1000, magnitude=4.0, noise=0.0, seed=0, baseline_frac=0.8)
    start = int(signal.metadata["anomaly_start"])
    assert not signal.ground_truth[start - 1]
    assert signal.ground_truth[start]


def test_step_with_recovery_returns_to_baseline() -> None:
    """The excursion decays back; the end looks like the beginning."""
    signal = step_with_recovery(n=2000, magnitude=5.0, noise=0.0, seed=0, decay=0.02)
    assert np.mean(signal.values[-50:]) == pytest.approx(1.0, abs=0.05)
    start = int(signal.metadata["anomaly_start"])
    assert signal.values[start] > 4.0


def test_step_with_recovery_is_monotonically_decreasing_after_the_step() -> None:
    """Direction matters: the excursion only ever falls once it has risen."""
    signal = step_with_recovery(n=2000, magnitude=5.0, noise=0.0, seed=0, decay=0.02)
    start = int(signal.metadata["anomaly_start"])
    assert np.all(np.diff(signal.values[start:]) < 1e-12)


def test_spikes_are_isolated_and_correctly_placed() -> None:
    """Exactly n_spikes labelled points, each at a recorded position."""
    signal = spikes(n=1000, magnitude=6.0, noise=0.0, seed=3, n_spikes=5, width=1)
    assert signal.ground_truth.sum() == 5
    for pos in signal.metadata["positions"]:
        assert signal.ground_truth[pos]
        assert signal.values[pos] == pytest.approx(6.0, rel=1e-9)


def test_spikes_leave_the_baseline_alone() -> None:
    """Everything that is not a spike sits at baseline."""
    signal = spikes(n=1000, magnitude=6.0, noise=0.0, seed=3, n_spikes=5)
    assert np.allclose(signal.values[~signal.ground_truth], 1.0, atol=1e-9)


def test_seasonal_component_oscillates() -> None:
    """The baseline segment crosses its own mean repeatedly."""
    signal = seasonal_plus_ramp(n=1000, magnitude=3.0, noise=0.0, seed=0, period=100)
    start = int(signal.metadata["anomaly_start"])
    baseline = signal.values[:start]
    crossings = np.sum(np.diff(np.sign(baseline - np.mean(baseline))) != 0)
    assert crossings >= 2


def test_regime_switch_is_bimodal_and_unlabelled() -> None:
    """Two stable healthy levels, nothing labelled."""
    signal = regime_switch(n=1200, magnitude=1.8, noise=0.0, seed=0, n_blocks=6)
    levels = np.unique(np.round(signal.values, 6))
    assert len(levels) == 2
    assert not signal.ground_truth.any()


def test_ramp_with_gaps_has_gaps_and_keeps_the_ramp() -> None:
    """NaNs are present, the labels survive, and the underlying trend remains."""
    signal = ramp_with_gaps(n=1000, magnitude=5.0, noise=0.0, seed=0, gap_frac=0.15, n_gaps=4)
    n_missing = int(np.isnan(signal.values).sum())
    assert n_missing > 0
    assert signal.ground_truth.any()
    finite = signal.values[np.isfinite(signal.values)]
    assert finite[-1] > finite[0]


def test_ramp_with_gaps_matches_the_clean_ramp_where_data_exists() -> None:
    """The gap generator only removes points; it does not perturb the rest."""
    kwargs = {"n": 1000, "magnitude": 5.0, "noise": 0.01, "seed": 0, "baseline_frac": 0.2}
    clean = linear_ramp(**kwargs)
    gapped = ramp_with_gaps(**kwargs)
    present = np.isfinite(gapped.values)
    np.testing.assert_allclose(gapped.values[present], clean.values[present])


def test_pure_noise_is_centred_on_baseline() -> None:
    """No trend, no level shift, nothing to find."""
    signal = pure_noise(n=5000, noise=0.05, seed=0)
    assert np.mean(signal.values) == pytest.approx(1.0, abs=0.01)
    assert np.std(signal.values) == pytest.approx(0.05, rel=0.1)


def test_pure_noise_ignores_magnitude() -> None:
    """magnitude is accepted for signature compatibility and has no effect."""
    a = pure_noise(n=500, magnitude=1.0, noise=0.02, seed=0)
    b = pure_noise(n=500, magnitude=1000.0, noise=0.02, seed=0)
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.1])
def test_noise_parameter_controls_spread(noise: float) -> None:
    """Larger noise means larger residual spread about the baseline."""
    signal = pure_noise(n=20_000, noise=noise, seed=0)
    assert np.std(signal.values) == pytest.approx(noise, abs=0.005)


def test_signal_rejects_mismatched_shapes() -> None:
    """The dataclass guards its own invariant."""
    with pytest.raises(ValueError, match="same shape"):
        Signal(np.zeros(10), np.zeros(9, dtype=bool), "bad", {})


def test_make_signal_rejects_unknown_names() -> None:
    """A typo in a generator name fails loudly rather than silently."""
    with pytest.raises(KeyError, match="unknown generator"):
        make_signal("no_such_generator")


def test_baseline_fraction_is_validated() -> None:
    """An out-of-range baseline fraction is rejected."""
    with pytest.raises(ValueError, match="baseline_frac"):
        linear_ramp(n=100, baseline_frac=1.5)
