"""Tests for the two controls, including their constructed demonstration cases."""

from __future__ import annotations

import numpy as np
import pytest

from tsadharness.controls import (
    artifact_demo,
    corroborated_demo,
    default_baseline_windows,
    disjoint_window_control,
    first_persistent_index,
    onset_stability,
)
from tsadharness.detectors import MADReference
from tsadharness.signals import (
    linear_ramp,
    pure_noise,
    regime_switch,
    seasonal_plus_ramp,
    step_change,
)

# --------------------------------------------------------------------------
# disjoint_window: the temporal-generalization artifact
# --------------------------------------------------------------------------


def test_artifact_demo_flags_a_finding_that_does_not_exist() -> None:
    """The constructed case: no anomaly exists, one model reports one anyway.

    The series is a healthy two-mode asset whose ground truth is all-False. A
    single-model pipeline trained on the low mode reports a confident finding
    when the asset switches to the high mode. The control must catch it.
    """
    result = artifact_demo()
    assert result.model_a_flags, "the single-model pipeline should report a finding here"
    assert not result.agree
    assert result.is_artifact
    assert "ARTIFACT" in result.verdict


def test_artifact_demo_is_built_on_a_series_with_no_anomaly() -> None:
    """Guards the premise: there really is nothing to find."""
    signal = regime_switch(n=1200, magnitude=1.8, noise=0.02, seed=0, n_blocks=6)
    assert not signal.ground_truth.any()


def test_artifact_demo_models_disagree_by_a_wide_margin() -> None:
    """The disagreement is not marginal; it is more than an order of magnitude."""
    result = artifact_demo()
    assert result.ratio > 10.0
    assert result.relative_gap > 0.9


def test_corroborated_demo_lets_a_real_finding_through() -> None:
    """A control that rejected everything would be useless."""
    result = corroborated_demo()
    assert result.model_a_flags
    assert result.agree
    assert not result.is_artifact
    assert "CORROBORATED" in result.verdict


def test_corroborated_demo_models_agree_closely() -> None:
    """Two disjoint healthy windows produce nearly the same error level."""
    result = corroborated_demo()
    assert result.ratio == pytest.approx(1.0, abs=0.2)


def test_control_reports_no_finding_when_the_model_is_silent() -> None:
    """A genuinely quiet model yields neither a finding nor an artifact verdict.

    The threshold is raised to 8 sigma here on purpose. At the conventional 3
    sigma a model is *not* silent on pure noise - see
    :func:`test_three_sigma_is_not_silent_on_pure_noise` - so this branch needs
    a model that really does stay quiet.
    """
    signal = pure_noise(n=1200, noise=0.02, seed=0)
    result = disjoint_window_control(
        signal.values,
        event_window=(400, 600),
        train_a=(0, 300),
        train_b=(700, 1000),
        model_factory=lambda: MADReference(threshold=8.0, reference_fraction=1.0),
    )
    assert not result.model_a_flags
    assert not result.is_artifact
    assert "NO FINDING" in result.verdict


def test_three_sigma_is_not_silent_on_pure_noise() -> None:
    """A 3-sigma rule fires on stationary noise, which is why the control matters.

    Roughly 0.27% of Gaussian samples exceed 3 sigma, so over a few hundred
    points at least one crossing is likely. Recording it as a test keeps the
    false-positive numbers in the README from looking like a bug.
    """
    signal = pure_noise(n=1200, noise=0.02, seed=0)
    det = MADReference(threshold=3.0, reference_fraction=1.0)
    det.fit(signal.values[0:300])
    assert det.detect(signal.values[400:600]).any()


def test_noise_does_not_produce_an_artifact_verdict() -> None:
    """Both models agree about stationary noise, so nothing is called an artifact.

    This is the property that keeps the control from crying wolf: an
    occasional threshold crossing is not enough to trigger it, because the two
    disjoint models see the same error *level*.
    """
    signal = pure_noise(n=1200, noise=0.02, seed=0)
    result = disjoint_window_control(
        signal.values, event_window=(400, 600), train_a=(0, 300), train_b=(700, 1000)
    )
    assert result.agree
    assert not result.is_artifact


def test_control_rejects_overlapping_training_windows() -> None:
    """Overlapping windows would defeat the entire purpose of the control."""
    values = pure_noise(n=1000, seed=0).values
    with pytest.raises(ValueError, match="overlap"):
        disjoint_window_control(
            values, event_window=(800, 1000), train_a=(0, 400), train_b=(300, 700)
        )


def test_control_rejects_out_of_range_windows() -> None:
    """Window bounds are validated against the series length."""
    values = pure_noise(n=500, seed=0).values
    with pytest.raises(ValueError, match="event window"):
        disjoint_window_control(
            values, event_window=(400, 900), train_a=(0, 100), train_b=(200, 300)
        )
    with pytest.raises(ValueError, match="train_a window"):
        disjoint_window_control(
            values, event_window=(400, 500), train_a=(100, 50), train_b=(200, 300)
        )


def test_control_rejects_a_tolerance_below_one() -> None:
    """A ratio tolerance below 1 is meaningless."""
    values = pure_noise(n=500, seed=0).values
    with pytest.raises(ValueError, match="ratio_tolerance"):
        disjoint_window_control(
            values,
            event_window=(400, 500),
            train_a=(0, 100),
            train_b=(200, 300),
            ratio_tolerance=0.5,
        )


def test_control_ratio_is_symmetric_in_the_two_models() -> None:
    """Swapping A and B leaves the ratio, and hence the verdict, unchanged."""
    signal = regime_switch(n=1200, magnitude=1.8, noise=0.02, seed=0, n_blocks=6)
    forward = disjoint_window_control(
        signal.values, event_window=(200, 400), train_a=(0, 200), train_b=(600, 800)
    )
    backward = disjoint_window_control(
        signal.values, event_window=(200, 400), train_a=(600, 800), train_b=(0, 200)
    )
    assert forward.ratio == pytest.approx(backward.ratio)


def test_control_records_the_windows_it_used() -> None:
    """The result is self-describing, so a saved row can be re-read later."""
    result = artifact_demo()
    assert result.window_a == (0, 200)
    assert result.window_b == (600, 800)
    assert result.event_window == (200, 400)


def test_control_ratio_is_at_least_one() -> None:
    """The ratio is defined as larger over smaller."""
    for result in (artifact_demo(), corroborated_demo()):
        assert result.ratio >= 1.0


def test_control_accepts_a_custom_model_factory() -> None:
    """Any detector can be plugged in, not just the default."""
    signal = regime_switch(n=1200, magnitude=1.8, noise=0.02, seed=0, n_blocks=6)
    result = disjoint_window_control(
        signal.values,
        event_window=(200, 400),
        train_a=(0, 200),
        train_b=(600, 800),
        model_factory=lambda: MADReference(threshold=4.0, reference_fraction=1.0),
    )
    assert result.is_artifact


# --------------------------------------------------------------------------
# window_shift: onset stability
# --------------------------------------------------------------------------


def test_step_onset_is_stable_and_permits_a_lead_time_claim() -> None:
    """A step has a physical onset: every baseline window agrees on it."""
    signal = step_change(n=1000, magnitude=3.0, noise=0.02, seed=0, baseline_frac=0.8)
    stab = onset_stability(signal.values)
    assert stab.classification == "step_type"
    assert stab.spread == 0
    assert not stab.suppress_lead_time_claim


def test_step_onset_lands_on_the_actual_step() -> None:
    """Stable and correct, not merely stable."""
    signal = step_change(n=1000, magnitude=3.0, noise=0.02, seed=0, baseline_frac=0.8)
    stab = onset_stability(signal.values)
    assert set(stab.onset_values) == {800}


def test_ramp_onset_moves_and_suppresses_the_lead_time_claim() -> None:
    """A ramp has no physical onset, and the control says so in a field."""
    signal = linear_ramp(n=1000, magnitude=3.0, noise=0.02, seed=0, baseline_frac=0.2)
    stab = onset_stability(signal.values)
    assert stab.classification == "ramp_type"
    assert stab.spread is not None and stab.spread > 100
    assert stab.suppress_lead_time_claim


def test_seasonal_ramp_onset_is_also_unstable() -> None:
    """The instability is a property of ramps, not of one generator."""
    signal = seasonal_plus_ramp(n=1000, magnitude=3.0, noise=0.02, seed=0)
    stab = onset_stability(signal.values)
    assert stab.suppress_lead_time_claim


@pytest.mark.parametrize("magnitude", [2.0, 5.0, 20.0, 100.0])
def test_ramp_onset_is_unstable_at_every_magnitude(magnitude: float) -> None:
    """Making the fault bigger does not make its onset a time."""
    signal = linear_ramp(n=1000, magnitude=magnitude, noise=0.02, seed=0)
    assert onset_stability(signal.values).suppress_lead_time_claim


def test_suppression_flag_is_a_plain_bool() -> None:
    """Machine-readable by design: a caveat in prose gets dropped, a field does not."""
    stab = onset_stability(linear_ramp(n=1000, magnitude=3.0, noise=0.02, seed=0).values)
    assert isinstance(stab.suppress_lead_time_claim, bool)


def test_undetermined_when_nothing_is_ever_detected() -> None:
    """No onset anywhere means no claim, not a claim of zero."""
    signal = pure_noise(n=1000, noise=0.02, seed=0)
    stab = onset_stability(signal.values, persistence=50)
    assert stab.classification == "undetermined"
    assert stab.suppress_lead_time_claim
    assert stab.spread is None


def test_onset_stability_reports_one_entry_per_window() -> None:
    """Every window tried is accounted for in the result."""
    signal = linear_ramp(n=1000, magnitude=5.0, noise=0.02, seed=0)
    windows = [(0, 100), (0, 200), (100, 300)]
    stab = onset_stability(signal.values, baseline_windows=windows)
    assert stab.n_windows == 3
    assert set(stab.onsets) == {(0, 100), (0, 200), (100, 300)}


def test_onset_stability_validates_windows_and_tolerance() -> None:
    """Bad inputs fail loudly."""
    values = linear_ramp(n=500, seed=0).values
    with pytest.raises(ValueError, match="baseline window"):
        onset_stability(values, baseline_windows=[(0, 900)])
    with pytest.raises(ValueError, match="at least one baseline window"):
        onset_stability(values, baseline_windows=[])
    with pytest.raises(ValueError, match="tolerance_fraction"):
        onset_stability(values, tolerance_fraction=-0.1)


def test_default_baseline_windows_are_valid_and_varied() -> None:
    """The default window set is in range and genuinely different from each other."""
    windows = default_baseline_windows(1000)
    assert len(windows) >= 3
    for start, stop in windows:
        assert 0 <= start < stop <= 1000
    assert len({w[1] - w[0] for w in windows}) > 1


def test_persistence_ignores_isolated_noise_crossings() -> None:
    """Without persistence the onset is whichever noise sample crossed first."""
    flags = np.zeros(100, dtype=bool)
    flags[10] = True  # isolated blip
    flags[50:60] = True  # real, sustained
    assert first_persistent_index(flags, persistence=1) == 10
    assert first_persistent_index(flags, persistence=5) == 50


def test_persistence_respects_the_start_offset() -> None:
    """Firings inside the asserted-healthy baseline window are ignored."""
    flags = np.zeros(100, dtype=bool)
    flags[10:20] = True
    flags[70:80] = True
    assert first_persistent_index(flags, start=0, persistence=5) == 10
    assert first_persistent_index(flags, start=30, persistence=5) == 70


def test_persistence_returns_none_when_nothing_qualifies() -> None:
    """A short burst does not qualify as an onset."""
    flags = np.zeros(100, dtype=bool)
    flags[40:42] = True
    assert first_persistent_index(flags, persistence=5) is None


def test_persistence_handles_a_series_shorter_than_the_run() -> None:
    """Degenerate input returns None rather than raising."""
    assert first_persistent_index(np.ones(3, dtype=bool), persistence=10) is None


def test_the_two_controls_answer_different_questions() -> None:
    """A real step is corroborated by one control and stable under the other.

    Together they say: the finding is real, and its onset is a time. For a
    ramp, the second half of that sentence is not available.
    """
    signal = step_change(n=1000, magnitude=3.0, noise=0.02, seed=0, baseline_frac=0.8)
    assert not corroborated_demo().is_artifact
    assert not onset_stability(signal.values).suppress_lead_time_claim

    ramp = linear_ramp(n=1000, magnitude=3.0, noise=0.02, seed=0)
    assert onset_stability(ramp.values).suppress_lead_time_claim
