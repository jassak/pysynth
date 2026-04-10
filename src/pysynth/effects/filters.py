from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from pysynth._core import Effect, Signal


class LowPassFilter(Effect):
    """Butterworth low-pass filter.

    Attenuates frequencies above ``cutoff_hz``.
    """

    def __init__(self, cutoff_hz: float, order: int = 4) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        b, a = scipy_signal.butter(self.order, self.cutoff_hz / nyquist, btype="low")
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)


class HighPassFilter(Effect):
    """Butterworth high-pass filter.

    Attenuates frequencies below ``cutoff_hz``.
    """

    def __init__(self, cutoff_hz: float, order: int = 4) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        b, a = scipy_signal.butter(self.order, self.cutoff_hz / nyquist, btype="high")
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)


class BandPassFilter(Effect):
    """Butterworth band-pass filter.

    Passes frequencies between ``low_hz`` and ``high_hz``.
    """

    def __init__(self, low_hz: float, high_hz: float, order: int = 4) -> None:
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        b, a = scipy_signal.butter(
            self.order,
            [self.low_hz / nyquist, self.high_hz / nyquist],
            btype="band",
        )
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)
