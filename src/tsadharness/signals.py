"""Synthetic time-series generators with known ground truth.

Every generator returns a :class:`Signal` whose ``ground_truth`` mask marks the
points that a detector is supposed to flag. Two generators (``regime_switch``
and ``pure_noise``) carry an all-False mask on purpose: they measure the false
positive rate when there is nothing to find.

All generators share the signature ``(n, magnitude, noise, seed, **extras)`` so
that the evaluation protocol can sweep over them uniformly. Every generator is
deterministic given ``seed``.

The baseline level is fixed at 1.0 throughout, so ``magnitude`` is always a
multiple of baseline and ``noise`` is a standard deviation in the same units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

BASELINE: float = 1.0
"""Nominal healthy level shared by every generator."""


@dataclass(eq=False)
class Signal:
    """A synthetic series together with its ground-truth anomaly mask.

    Attributes:
        values: The observed series. May contain NaN for generators that
            simulate dropouts.
        ground_truth: Boolean mask, True where an anomaly is present. Same
            shape as ``values``.
        name: Generator name, used as a key in result tables.
        metadata: Generator parameters plus any derived quantities that tests
            or reports need (e.g. ``anomaly_start``).
    """

    values: np.ndarray
    ground_truth: np.ndarray
    name: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.ground_truth = np.asarray(self.ground_truth, dtype=bool)
        if self.values.shape != self.ground_truth.shape:
            raise ValueError(
                "values and ground_truth must have the same shape, got "
                f"{self.values.shape} and {self.ground_truth.shape}"
            )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def has_anomaly(self) -> bool:
        """True when at least one point is labelled anomalous."""
        return bool(self.ground_truth.any())


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _split(n: int, baseline_frac: float) -> int:
    """Return the index at which the anomalous segment starts."""
    if not 0.0 <= baseline_frac < 1.0:
        raise ValueError(f"baseline_frac must be in [0, 1), got {baseline_frac}")
    return int(round(n * baseline_frac))


def linear_ramp(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    baseline_frac: float = 0.2,
) -> Signal:
    """Flat baseline followed by a linear rise to ``magnitude`` x baseline.

    This is the geometry that a self-referencing robust z-score cannot see at
    any magnitude, provided the baseline occupies no more than 1/4 of the
    series. See ``paper/ramp_blindness.md``.

    Args:
        n: Number of samples.
        magnitude: Final level as a multiple of baseline.
        noise: Standard deviation of additive Gaussian noise.
        seed: Seed for reproducibility.
        baseline_frac: Fraction of the series before the ramp begins.

    Returns:
        A :class:`Signal` whose ground truth covers the ramp segment.
    """
    rng = _rng(seed)
    start = _split(n, baseline_frac)
    values = np.full(n, BASELINE, dtype=float)
    ramp_len = n - start
    values[start:] = np.linspace(BASELINE, BASELINE * magnitude, ramp_len)
    values += rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    gt[start:] = True
    return Signal(
        values,
        gt,
        "linear_ramp",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "baseline_frac": baseline_frac,
            "anomaly_start": start,
            "shape": "ramp",
        },
    )


def exponential_ramp(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    baseline_frac: float = 0.2,
) -> Signal:
    """Flat baseline followed by a geometric rise to ``magnitude`` x baseline.

    Included to separate "the detector fails on this shape" from "the detector
    fails at this scale": the endpoints match :func:`linear_ramp` exactly, only
    the curvature differs.
    """
    rng = _rng(seed)
    start = _split(n, baseline_frac)
    values = np.full(n, BASELINE, dtype=float)
    ramp_len = n - start
    t = np.linspace(0.0, 1.0, ramp_len)
    values[start:] = BASELINE * np.power(max(magnitude, 1e-9), t)
    values += rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    gt[start:] = True
    return Signal(
        values,
        gt,
        "exponential_ramp",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "baseline_frac": baseline_frac,
            "anomaly_start": start,
            "shape": "ramp",
        },
    )


def step_change(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    baseline_frac: float = 0.8,
) -> Signal:
    """Abrupt level shift against a flat baseline. The positive control.

    This is the geometry the median/MAD family is actually built for, and the
    one it handles correctly at any contamination fraction other than exactly
    1/2. If a detector cannot find this, the detector is broken rather than
    merely mis-specified.
    """
    rng = _rng(seed)
    start = _split(n, baseline_frac)
    values = np.full(n, BASELINE, dtype=float)
    values[start:] = BASELINE * magnitude
    values += rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    gt[start:] = True
    return Signal(
        values,
        gt,
        "step_change",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "baseline_frac": baseline_frac,
            "anomaly_start": start,
            "shape": "step",
        },
    )


def step_with_recovery(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    baseline_frac: float = 0.5,
    decay: float = 0.02,
) -> Signal:
    """Abrupt step up followed by an exponential decay back to baseline.

    Direction matters here: the same excursion read backwards is a machine
    being repaired rather than a machine failing. A scorer that only sees
    ``|z|`` cannot tell the two apart, which is why the harness records the
    geometry name alongside the score.
    """
    rng = _rng(seed)
    start = _split(n, baseline_frac)
    values = np.full(n, BASELINE, dtype=float)
    tail = np.arange(n - start, dtype=float)
    excursion = BASELINE * (magnitude - 1.0) * np.exp(-decay * tail)
    values[start:] = BASELINE + excursion
    values += rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    # Label the excursion while it is still meaningfully above baseline.
    elevated = excursion > 0.1 * BASELINE * max(magnitude - 1.0, 1e-9)
    gt[start:] = elevated
    return Signal(
        values,
        gt,
        "step_with_recovery",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "baseline_frac": baseline_frac,
            "decay": decay,
            "anomaly_start": start,
            "shape": "step",
        },
    )


def spikes(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    n_spikes: int = 5,
    width: int = 1,
) -> Signal:
    """Isolated impulses on a flat baseline. The easy case.

    Point anomalies are the geometry that published benchmarks are dominated
    by, and the geometry on which nearly every method scores well. Its presence
    here is a calibration reference, not a challenge.
    """
    rng = _rng(seed)
    values = np.full(n, BASELINE, dtype=float) + rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    margin = max(width + 1, n // 20)
    positions = rng.choice(np.arange(margin, n - margin), size=n_spikes, replace=False)
    for pos in positions:
        values[pos : pos + width] = BASELINE * magnitude
        gt[pos : pos + width] = True
    return Signal(
        values,
        gt,
        "spikes",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "n_spikes": n_spikes,
            "width": width,
            "positions": sorted(int(p) for p in positions),
            "shape": "spike",
        },
    )


def seasonal_plus_ramp(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    baseline_frac: float = 0.2,
    period: int = 100,
    amplitude: float = 0.3,
) -> Signal:
    """Sinusoidal baseline with a linear ramp added on top.

    Tests whether a periodic component masks a trend. The seasonal term widens
    the spread of the series, which inflates a self-referencing MAD and makes
    the ramp even less visible in ``z`` units.
    """
    rng = _rng(seed)
    start = _split(n, baseline_frac)
    t = np.arange(n, dtype=float)
    values = BASELINE + amplitude * np.sin(2.0 * np.pi * t / float(period))
    ramp = np.zeros(n, dtype=float)
    ramp[start:] = np.linspace(0.0, BASELINE * (magnitude - 1.0), n - start)
    values = values + ramp + rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    gt[start:] = True
    return Signal(
        values,
        gt,
        "seasonal_plus_ramp",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "baseline_frac": baseline_frac,
            "period": period,
            "amplitude": amplitude,
            "anomaly_start": start,
            "shape": "ramp",
        },
    )


def regime_switch(
    n: int = 1000,
    magnitude: float = 1.8,
    noise: float = 0.02,
    seed: int = 0,
    n_blocks: int = 6,
) -> Signal:
    """Two stable, non-faulty operating states. The false-positive trap.

    Ground truth is all False: both levels are healthy, the asset is simply
    running in two different modes. Any firing here is a false alarm, and this
    is the failure mode that destroys operator trust fastest in practice.
    """
    rng = _rng(seed)
    values = np.full(n, BASELINE, dtype=float)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    for i in range(n_blocks):
        level = BASELINE if i % 2 == 0 else BASELINE * magnitude
        values[edges[i] : edges[i + 1]] = level
    values += rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    return Signal(
        values,
        gt,
        "regime_switch",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "n_blocks": n_blocks,
            "shape": "none",
            "note": "both levels are healthy; ground truth is all-False",
        },
    )


def ramp_with_gaps(
    n: int = 1000,
    magnitude: float = 3.0,
    noise: float = 0.02,
    seed: int = 0,
    baseline_frac: float = 0.2,
    gap_frac: float = 0.15,
    n_gaps: int = 4,
) -> Signal:
    """A linear ramp punctuated by NaN dropouts.

    Irregular sampling breaks onset logic: the first index above a threshold is
    not the first *time* above it. The harness keeps this generator to make
    sure detectors declare how they handle missing data rather than silently
    interpolating and then reporting a confident onset date.
    """
    base = linear_ramp(
        n=n, magnitude=magnitude, noise=noise, seed=seed, baseline_frac=baseline_frac
    )
    rng = _rng(seed + 977)
    values = base.values.copy()
    gap_len = max(1, int(round(n * gap_frac / max(n_gaps, 1))))
    starts = rng.choice(np.arange(0, n - gap_len), size=n_gaps, replace=False)
    for s in starts:
        values[s : s + gap_len] = np.nan
    meta = dict(base.metadata)
    meta.update({"gap_frac": gap_frac, "n_gaps": n_gaps, "gap_len": gap_len})
    return Signal(values, base.ground_truth.copy(), "ramp_with_gaps", meta)


def pure_noise(
    n: int = 1000,
    magnitude: float = 1.0,
    noise: float = 0.02,
    seed: int = 0,
) -> Signal:
    """Stationary noise with no anomaly at all. The most important generator.

    ``magnitude`` is accepted for signature compatibility and deliberately
    ignored: there is nothing here to scale. Ground truth is all False, so the
    only thing this measures is how often a detector invents an event. A method
    that cannot post a low number here has not earned any of its recall
    numbers elsewhere.
    """
    rng = _rng(seed)
    values = np.full(n, BASELINE, dtype=float) + rng.normal(0.0, noise, n)
    gt = np.zeros(n, dtype=bool)
    return Signal(
        values,
        gt,
        "pure_noise",
        {
            "magnitude": magnitude,
            "noise": noise,
            "seed": seed,
            "shape": "none",
            "note": "magnitude is ignored by construction; ground truth is all-False",
        },
    )


GeneratorFn = Callable[..., Signal]

GENERATORS: dict[str, GeneratorFn] = {
    "linear_ramp": linear_ramp,
    "exponential_ramp": exponential_ramp,
    "step_change": step_change,
    "step_with_recovery": step_with_recovery,
    "spikes": spikes,
    "seasonal_plus_ramp": seasonal_plus_ramp,
    "regime_switch": regime_switch,
    "ramp_with_gaps": ramp_with_gaps,
    "pure_noise": pure_noise,
}
"""All nine generators, keyed by name."""

ANOMALOUS_GENERATORS: list[str] = [
    name for name in GENERATORS if name not in ("regime_switch", "pure_noise")
]
"""Generators that actually contain an anomaly."""

NO_ANOMALY_GENERATORS: list[str] = ["regime_switch", "pure_noise"]
"""Generators whose ground truth is all-False."""


def make_signal(name: str, **kwargs: object) -> Signal:
    """Build a signal by generator name.

    Args:
        name: Key into :data:`GENERATORS`.
        **kwargs: Forwarded to the generator.

    Raises:
        KeyError: If ``name`` is not a known generator.
    """
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(GENERATORS)}")
    return GENERATORS[name](**kwargs)  # type: ignore[arg-type]
