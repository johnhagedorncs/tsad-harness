"""A deliberately minimal sliding-window autoencoder, in pure NumPy.

This is a single-hidden-layer bottleneck autoencoder trained by plain gradient
descent on standardized windows, scoring each window by its reconstruction
error. There is no torch dependency: the harness is meant to be runnable from a
clean checkout with NumPy, SciPy and matplotlib, and a heavyweight framework
would buy nothing here.

It is minimal on purpose, and I want to be explicit that this is not a
competitive reconstruction model. Its role in the harness is to represent the
*class* of learned detectors that train on a declared window, so that the
temporal-generalization artifact in
:mod:`tsadharness.controls.disjoint_window` can be demonstrated on something
that actually learns. A larger model would exhibit the same artifact; it would
just take longer to show it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .base import BaseDetector, interpolate_nans, scaled_mad


@contextmanager
def _quiet_matmul() -> Iterator[None]:
    """Suppress spurious floating-point status flags raised by ``matmul``.

    NumPy's vectorised ``matmul`` can set the hardware divide-by-zero, overflow
    and invalid flags even when every input and output is finite - the SIMD
    kernels read whole registers past the end of short arrays, and the flags
    they leave behind are reported as RuntimeWarnings. The effect is
    reproducible here with an all-zero input matrix and all-zero weights, where
    a genuine division cannot occur.

    Suppressing the flags is therefore correct, but it must not be allowed to
    hide a real divergence. :meth:`AEReconstruction.fit` checks that the
    trained weights are finite and raises if they are not, so a genuinely
    unstable fit still fails loudly.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield


def sliding_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Return the ``(n - window + 1, window)`` matrix of contiguous windows.

    Args:
        values: 1-D finite series.
        window: Window length, must not exceed ``values.size``.

    Raises:
        ValueError: If ``window`` is larger than the series or below 2.
    """
    arr = np.asarray(values, dtype=float)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if window > arr.size:
        raise ValueError(f"window {window} exceeds series length {arr.size}")
    n_win = arr.size - window + 1
    idx = np.arange(n_win)[:, None] + np.arange(window)[None, :]
    return arr[idx]


class AEReconstruction(BaseDetector):
    """Bottleneck autoencoder over sliding windows, scored by recon error.

    The model is fit on a declared leading fraction of the series, exactly as
    :class:`~tsadharness.detectors.mad_ref.MADReference` is, so that the two
    share an assumption and differ only in capacity.

    Args:
        window: Sliding window length.
        hidden: Bottleneck width.
        epochs: Full-batch gradient descent steps.
        lr: Learning rate.
        reference_fraction: Leading fraction of the fitted series used for
            training and for threshold calibration.
        threshold_k: Threshold is ``median(train_err) + threshold_k *
            scaled_mad(train_err)``.
        seed: Seed for weight initialisation; training is otherwise
            deterministic (full batch, no shuffling, no dropout).
    """

    name = "AEReconstruction"

    def __init__(
        self,
        window: int = 32,
        hidden: int = 4,
        epochs: int = 400,
        lr: float = 0.5,
        reference_fraction: float = 0.2,
        threshold_k: float = 8.0,
        seed: int = 0,
    ) -> None:
        super().__init__(threshold=np.inf)
        self.window = int(window)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.reference_fraction = float(reference_fraction)
        self.threshold_k = float(threshold_k)
        self.seed = int(seed)
        self.mu_: float | None = None
        self.sd_: float | None = None
        self.w1_: np.ndarray | None = None
        self.b1_: np.ndarray | None = None
        self.w2_: np.ndarray | None = None
        self.b2_: np.ndarray | None = None
        self.train_loss_: list[float] = []

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        assert self.mu_ is not None and self.sd_ is not None
        return (values - self.mu_) / self.sd_

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.w1_ is not None and self.w2_ is not None
        assert self.b1_ is not None and self.b2_ is not None
        with _quiet_matmul():
            h = np.tanh(x @ self.w1_ + self.b1_)
            return h, h @ self.w2_ + self.b2_

    def fit(self, series: np.ndarray) -> None:
        """Train on the declared reference window and calibrate the threshold."""
        values = interpolate_nans(series)
        stop = max(self.window + 4, int(round(values.size * self.reference_fraction)))
        stop = min(stop, values.size)
        train_values = values[:stop]
        if train_values.size < self.window + 1:
            raise ValueError(
                f"{self.name}: reference window of {train_values.size} points is too "
                f"short for window={self.window}"
            )

        self.mu_ = float(np.mean(train_values))
        sd = float(np.std(train_values))
        self.sd_ = sd if sd > 1e-12 else 1.0

        x = sliding_windows(self._standardize(train_values), self.window)
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.window)
        self.w1_ = rng.normal(0.0, scale, (self.window, self.hidden))
        self.b1_ = np.zeros(self.hidden)
        self.w2_ = rng.normal(0.0, scale, (self.hidden, self.window))
        self.b2_ = np.zeros(self.window)

        n = float(x.shape[0] * self.window)
        self.train_loss_ = []
        with _quiet_matmul():
            for _ in range(self.epochs):
                h, out = self._forward(x)
                resid = out - x
                self.train_loss_.append(float(np.mean(resid**2)))
                d_out = 2.0 * resid / n
                d_w2 = h.T @ d_out
                d_b2 = d_out.sum(axis=0)
                d_h = (d_out @ self.w2_.T) * (1.0 - h**2)
                d_w1 = x.T @ d_h
                d_b1 = d_h.sum(axis=0)
                self.w1_ -= self.lr * d_w1
                self.b1_ -= self.lr * d_b1
                self.w2_ -= self.lr * d_w2
                self.b2_ -= self.lr * d_b2

        # The flag suppression above must never hide an actually broken fit.
        for label, param in (
            ("w1", self.w1_),
            ("b1", self.b1_),
            ("w2", self.w2_),
            ("b2", self.b2_),
        ):
            if not np.isfinite(param).all():
                raise RuntimeError(
                    f"{self.name}: training diverged, {label} contains non-finite values; "
                    f"lower lr (currently {self.lr})"
                )
        if self.train_loss_ and not np.isfinite(self.train_loss_).all():
            raise RuntimeError(f"{self.name}: training diverged, loss became non-finite")

        self._fitted = True
        train_err = self._window_errors(train_values)
        med = float(np.median(train_err))
        self.threshold = med + self.threshold_k * scaled_mad(train_err, center=med)

    def _window_errors(self, values: np.ndarray) -> np.ndarray:
        x = sliding_windows(self._standardize(values), self.window)
        _, out = self._forward(x)
        return np.mean((out - x) ** 2, axis=1)

    def score(self, series: np.ndarray) -> np.ndarray:
        """Return per-point reconstruction error, aligned to the window end."""
        self._require_fit()
        values = interpolate_nans(series)
        if values.size < self.window:
            return np.zeros(values.size)
        errors = self._window_errors(values)
        # A window's error is attributed to its final sample; the leading
        # window-1 samples inherit the first available score.
        out = np.empty(values.size, dtype=float)
        out[self.window - 1 :] = errors
        out[: self.window - 1] = errors[0]
        return out
