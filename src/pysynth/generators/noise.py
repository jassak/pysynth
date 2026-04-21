from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.generators.base import Generator


class WhiteNoise(Generator):
    """Uniform white noise — equal energy at all frequencies."""

    def __init__(self, amplitude: float = 1.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.amplitude = amplitude
        self.sample_rate = sample_rate

    def render(self, dur: float, **_kwargs) -> Signal:
        n = int(dur * self.sample_rate)
        data = np.random.uniform(-1.0, 1.0, n).astype(np.float32) * self.amplitude
        return Signal(data, self.sample_rate)


class PinkNoise(Generator):
    """Pink noise (1/f spectrum) via Voss-McCartney algorithm.

    Pink noise has equal energy per octave, giving a warmer sound than white noise.
    """

    def __init__(self, amplitude: float = 1.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.amplitude = amplitude
        self.sample_rate = sample_rate

    def render(self, dur: float, **_kwargs) -> Signal:
        n = int(dur * self.sample_rate)
        # Paul Kellett's refined method: sum of filtered white noise sources
        # Each row advances at half the rate of the previous, summed together.
        n_rows = 16
        rows = np.random.uniform(-1.0, 1.0, (n_rows, n))
        # Scale each row by 1/sqrt(row_index+1) to approximate 1/f spectrum
        weights = 1.0 / np.sqrt(np.arange(1, n_rows + 1, dtype=np.float32))
        data = (rows * weights[:, np.newaxis]).sum(axis=0).astype(np.float32)
        # Normalise to [-1, 1] then scale
        peak = np.max(np.abs(data))
        if peak > 0:
            data /= peak
        data *= self.amplitude
        return Signal(data, self.sample_rate)
