"""Uniform random scores. The honesty check.

This detector contains no information about the series whatsoever. It appears
in every results table in this repository, and it is not a joke entry.

Wu and Keogh (arXiv:2009.13807) argue that much of the reported progress in
time-series anomaly detection is an artifact of how benchmarks are scored. The
sharpest way to show that is to run random guessing through the identical
pipeline and print its number next to everything else. Under a per-point firing
rate of alpha, the probability that at least one point inside an event of
length L fires is ``1 - (1 - alpha)**L``, which approaches 1 for any realistic
event length. Any metric that credits an event for a single point inside it -
point-adjusted F1, most notably - will therefore award this detector a high
score on any benchmark whose events are long.

The correct reading is not "random is good". It is that a recall number is
uninterpretable unless the false-positive number is printed beside it, which is
why :func:`tsadharness.protocol.false_positive_table` exists.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector


class RandomScore(BaseDetector):
    """Assigns each point an independent uniform score in [0, 1).

    Scoring is reproducible: the same instance scoring a series of the same
    length twice returns identical scores, because the generator is re-seeded
    on every call.

    Args:
        alpha: Nominal per-point firing rate. The threshold is set to
            ``1 - alpha``.
        seed: Seed for the score stream.
    """

    name = "RandomScore"

    def __init__(self, alpha: float = 0.01, seed: int = 0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        super().__init__(threshold=1.0 - float(alpha))
        self.alpha = float(alpha)
        self.seed = int(seed)

    def fit(self, series: np.ndarray) -> None:
        """No-op: there is nothing to learn, which is the point."""
        _ = np.asarray(series, dtype=float)
        self._fitted = True

    def score(self, series: np.ndarray) -> np.ndarray:
        """Return reproducible uniform scores, one per point."""
        self._require_fit()
        n = int(np.asarray(series).shape[0])
        return np.random.default_rng(self.seed).random(n)
