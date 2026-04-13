from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from pysynth._core import Effect, Signal, _as_array

_FILTER_CHUNK = 32  # samples per coefficient-update step for time-varying cutoff


class LowPassFilter(Effect):
    """Butterworth low-pass filter.

    Attenuates frequencies above ``cutoff_hz``.

    Parameters
    ----------
    cutoff_hz:
        Cutoff frequency in Hz. Accepts a constant ``float`` or a time-varying
        ``Signal`` for a modulated filter sweep. When a
        Signal, uses causal chunk-based filtering rather than zero-phase
        ``filtfilt``.
    """

    def __init__(self, cutoff_hz: float | Signal, order: int = 4) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        if isinstance(self.cutoff_hz, Signal):
            return Signal(
                _modulated_filter(sig.data, self.order, "low", self.cutoff_hz, nyquist),
                sig.sample_rate,
            )
        b, a = scipy_signal.butter(self.order, self.cutoff_hz / nyquist, btype="low")
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)


class HighPassFilter(Effect):
    """Butterworth high-pass filter.

    Attenuates frequencies below ``cutoff_hz``.

    Parameters
    ----------
    cutoff_hz:
        Cutoff frequency in Hz. Accepts a constant ``float`` or a time-varying
        ``Signal`` for a modulated filter sweep.
    """

    def __init__(self, cutoff_hz: float | Signal, order: int = 4) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        if isinstance(self.cutoff_hz, Signal):
            return Signal(
                _modulated_filter(sig.data, self.order, "high", self.cutoff_hz, nyquist),
                sig.sample_rate,
            )
        b, a = scipy_signal.butter(self.order, self.cutoff_hz / nyquist, btype="high")
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)


class BandPassFilter(Effect):
    """Butterworth band-pass filter.

    Passes frequencies between ``low_hz`` and ``high_hz``.

    Parameters
    ----------
    low_hz:
        Lower cutoff in Hz. Accepts a constant ``float`` or a time-varying ``Signal``.
    high_hz:
        Upper cutoff in Hz. Accepts a constant ``float`` or a time-varying ``Signal``.
    """

    def __init__(self, low_hz: float | Signal, high_hz: float | Signal, order: int = 4) -> None:
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        if isinstance(self.low_hz, Signal) or isinstance(self.high_hz, Signal):
            return Signal(
                _modulated_bandpass(sig.data, self.order, self.low_hz, self.high_hz, nyquist),
                sig.sample_rate,
            )
        b, a = scipy_signal.butter(
            self.order,
            [self.low_hz / nyquist, self.high_hz / nyquist],
            btype="band",
        )
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)


def _modulated_filter(
    data: np.ndarray,
    order: int,
    btype: str,
    cutoff: Signal,
    nyquist: float,
) -> np.ndarray:
    """Chunked causal sosfilt with time-varying single cutoff."""
    n = len(data)
    cutoff_arr = _as_array(cutoff, n)
    out = np.zeros_like(data, dtype=np.float64)
    for c in range(data.shape[1]) if data.ndim == 2 else [None]:
        x = (data[:, c] if c is not None else data).astype(np.float64)
        zi = None
        for start in range(0, n, _FILTER_CHUNK):
            end = min(start + _FILTER_CHUNK, n)
            chunk = x[start:end]
            cutoff_val = float(np.clip(np.mean(cutoff_arr[start:end]), 1.0, nyquist - 1.0))
            sos = scipy_signal.butter(order, cutoff_val / nyquist, btype=btype, output="sos")
            if zi is None:
                zi = scipy_signal.sosfilt_zi(sos) * chunk[0]
            chunk_out, zi = scipy_signal.sosfilt(sos, chunk, zi=zi)
            if c is not None:
                out[start:end, c] = chunk_out
            else:
                out[start:end] = chunk_out
    return out.astype(np.float32)


def _modulated_bandpass(
    data: np.ndarray,
    order: int,
    low: float | Signal,
    high: float | Signal,
    nyquist: float,
) -> np.ndarray:
    """Chunked causal sosfilt with time-varying band cutoffs."""
    n = len(data)
    low_arr = _as_array(low, n)
    high_arr = _as_array(high, n)
    out = np.zeros_like(data, dtype=np.float64)
    for c in range(data.shape[1]) if data.ndim == 2 else [None]:
        x = (data[:, c] if c is not None else data).astype(np.float64)
        zi = None
        for start in range(0, n, _FILTER_CHUNK):
            end = min(start + _FILTER_CHUNK, n)
            chunk = x[start:end]
            low_val = float(np.clip(np.mean(low_arr[start:end]), 1.0, nyquist - 2.0))
            high_val = float(np.clip(np.mean(high_arr[start:end]), low_val + 1.0, nyquist - 1.0))
            sos = scipy_signal.butter(
                order, [low_val / nyquist, high_val / nyquist], btype="band", output="sos"
            )
            if zi is None:
                zi = scipy_signal.sosfilt_zi(sos) * chunk[0]
            chunk_out, zi = scipy_signal.sosfilt(sos, chunk, zi=zi)
            if c is not None:
                out[start:end, c] = chunk_out
            else:
                out[start:end] = chunk_out
    return out.astype(np.float32)