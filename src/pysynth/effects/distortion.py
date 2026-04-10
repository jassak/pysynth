from __future__ import annotations

import numpy as np

from pysynth._core import Effect, Signal


class Tanh(Effect):
    """Soft saturation via hyperbolic tangent.

    ``drive`` controls how hard the signal is pushed into the nonlinearity.
    Higher drive = more harmonic content and compression.
    """

    def __init__(self, drive: float = 1.0) -> None:
        self.drive = drive

    def __call__(self, sig: Signal) -> Signal:
        data = np.tanh(sig.data * self.drive).astype(np.float32)
        return Signal(data, sig.sample_rate)


class Clip(Effect):
    """Hard clipping distortion.

    Flattens the waveform above ``threshold``, producing harsh, bright harmonics.
    ``threshold`` is a linear amplitude value (0..1).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = np.clip(threshold, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        data = np.clip(sig.data, -self.threshold, self.threshold).astype(np.float32)
        # Renormalise so output peak matches input peak
        peak = np.max(np.abs(sig.data))
        if peak > 0:
            data = data / self.threshold * peak
        return Signal(data, sig.sample_rate)


class Overdrive(Effect):
    """Asymmetric soft-clipping overdrive, inspired by tube amplifier characteristics.

    Applies piecewise soft clipping with a bias offset to introduce even-order
    harmonics (2nd, 4th, ...), which are warmer than the odd-order harmonics
    produced by symmetric clipping.
    """

    def __init__(self, gain: float = 4.0, bias: float = 0.1) -> None:
        self.gain = gain
        self.bias = bias

    def __call__(self, sig: Signal) -> Signal:
        x = sig.data * self.gain + self.bias
        # Piecewise function with three regions
        data = np.where(
            x >= 1.0 / 3.0,
            np.where(x >= 2.0 / 3.0, 1.0, (3.0 - (2.0 - x * 3.0) ** 2) / 3.0),
            2.0 * x,
        ).astype(np.float32)
        # Remove DC offset introduced by bias
        data -= data.mean()
        # Normalise
        peak = np.max(np.abs(data))
        if peak > 0:
            data /= peak
        return Signal(data, sig.sample_rate)
