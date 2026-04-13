from __future__ import annotations

import numba
import numpy as np

from pysynth._core import Effect, Signal


@numba.njit(cache=True)
def _envelope_follow(abs_x, envelope, attack_coef, release_coef):
    env = 0.0
    for i in range(len(abs_x)):
        sample = abs_x[i]
        if sample > env:
            env = attack_coef * env + (1.0 - attack_coef) * sample
        else:
            env = release_coef * env + (1.0 - release_coef) * sample
        envelope[i] = env


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


class Gain(Effect):
    """Apply a fixed gain in decibels."""

    def __init__(self, db: float) -> None:
        self.db = db

    def __call__(self, sig: Signal) -> Signal:
        return sig * _db_to_linear(self.db)


class Limiter(Effect):
    """Hard clip the signal to ``ceiling`` (in dBFS, default 0 dBFS).

    Prevents digital clipping at the output stage.
    """

    def __init__(self, ceiling_db: float = 0.0) -> None:
        self.ceiling = _db_to_linear(ceiling_db)

    def __call__(self, sig: Signal) -> Signal:
        data = np.clip(sig.data, -self.ceiling, self.ceiling).astype(np.float32)
        return Signal(data, sig.sample_rate)


class Compressor(Effect):
    """Feed-forward RMS compressor.

    Reduces dynamic range by attenuating signals above ``threshold_db``.
    Attack and release are in seconds and control how quickly the gain
    reduction engages and recovers.
    """

    def __init__(
        self,
        threshold_db: float = -12.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1,
    ) -> None:
        self.threshold = _db_to_linear(threshold_db)
        self.ratio = ratio
        self.attack = attack
        self.release = release

    def __call__(self, sig: Signal) -> Signal:
        sr = sig.sample_rate
        x = sig.data

        attack_coef = np.exp(-1.0 / (self.attack * sr))
        release_coef = np.exp(-1.0 / (self.release * sr))

        envelope = np.empty(len(x), dtype=np.float64)
        _envelope_follow(np.abs(x).astype(np.float64), envelope, attack_coef, release_coef)

        over = np.maximum(envelope - self.threshold, 0.0)
        reduction = over * (1.0 - 1.0 / self.ratio)
        gain = 1.0 - reduction / np.maximum(envelope, 1e-10)
        gain = np.clip(gain, 0.0, 1.0).astype(np.float32)

        return Signal(x * gain, sr)
