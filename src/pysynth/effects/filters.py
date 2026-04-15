from __future__ import annotations

import numba
import numpy as np
from scipy import signal as scipy_signal

from pysynth._core import Effect, Signal, _as_array


# ---------------------------------------------------------------------------
# Cytomic TPT State Variable Filter (SVF) — numba-accelerated
#
# Each SVF stage is 2nd-order (12 dB/oct).  For higher orders, multiple
# stages are cascaded in a single pass.  The cutoff can change every sample
# with no coefficient redesign — just recompute g = tan(pi * fc / sr).
#
# k = 2.0 gives a Butterworth-like Q ≈ 0.707 per stage.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _svf_lowpass(x, cutoff_arr, k_arr, sr, n_stages):
    n = len(x)
    ic1 = np.zeros(n_stages)
    ic2 = np.zeros(n_stages)
    out = np.empty(n)
    pi = np.pi
    for i in range(n):
        k = k_arr[i]
        fc = cutoff_arr[i]
        if fc < 1.0:
            fc = 1.0
        g = np.tan(pi * fc / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v = x[i]
        for s in range(n_stages):
            v3 = v - ic2[s]
            v1 = a1 * ic1[s] + a2 * v3
            v2 = ic2[s] + a2 * ic1[s] + a3 * v3
            ic1[s] = 2.0 * v1 - ic1[s]
            ic2[s] = 2.0 * v2 - ic2[s]
            v = v2  # lowpass output feeds next stage
        out[i] = v
    return out


@numba.njit(cache=True)
def _svf_highpass(x, cutoff_arr, k_arr, sr, n_stages):
    n = len(x)
    ic1 = np.zeros(n_stages)
    ic2 = np.zeros(n_stages)
    out = np.empty(n)
    pi = np.pi
    for i in range(n):
        k = k_arr[i]
        fc = cutoff_arr[i]
        if fc < 1.0:
            fc = 1.0
        g = np.tan(pi * fc / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v = x[i]
        for s in range(n_stages):
            v3 = v - ic2[s]
            v1 = a1 * ic1[s] + a2 * v3
            v2 = ic2[s] + a2 * ic1[s] + a3 * v3
            ic1[s] = 2.0 * v1 - ic1[s]
            ic2[s] = 2.0 * v2 - ic2[s]
            v = v - k * v1 - v2  # highpass output feeds next stage
        out[i] = v
    return out


@numba.njit(cache=True)
def _svf_bandpass(x, low_arr, high_arr, k_arr, sr, n_stages):
    """Bandpass via cascaded SVF highpass then lowpass."""
    n = len(x)
    # Highpass state
    hp_ic1 = np.zeros(n_stages)
    hp_ic2 = np.zeros(n_stages)
    # Lowpass state
    lp_ic1 = np.zeros(n_stages)
    lp_ic2 = np.zeros(n_stages)
    out = np.empty(n)
    pi = np.pi
    for i in range(n):
        k = k_arr[i]
        # Highpass stage (removes below low_arr)
        fc_hp = low_arr[i]
        if fc_hp < 1.0:
            fc_hp = 1.0
        g = np.tan(pi * fc_hp / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v = x[i]
        for s in range(n_stages):
            v3 = v - hp_ic2[s]
            v1 = a1 * hp_ic1[s] + a2 * v3
            v2 = hp_ic2[s] + a2 * hp_ic1[s] + a3 * v3
            hp_ic1[s] = 2.0 * v1 - hp_ic1[s]
            hp_ic2[s] = 2.0 * v2 - hp_ic2[s]
            v = v - k * v1 - v2

        # Lowpass stage (removes above high_arr)
        fc_lp = high_arr[i]
        if fc_lp < 1.0:
            fc_lp = 1.0
        g = np.tan(pi * fc_lp / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        for s in range(n_stages):
            v3 = v - lp_ic2[s]
            v1 = a1 * lp_ic1[s] + a2 * v3
            v2 = lp_ic2[s] + a2 * lp_ic1[s] + a3 * v3
            lp_ic1[s] = 2.0 * v1 - lp_ic1[s]
            lp_ic2[s] = 2.0 * v2 - lp_ic2[s]
            v = v2

        out[i] = v
    return out


def _svf_filter_mono(x, order, btype, cutoff_arr, k_arr, sr):
    """Apply SVF filter to a mono signal."""
    n_stages = max(1, order // 2)
    x64 = x.astype(np.float64)
    c64 = cutoff_arr.astype(np.float64)
    if btype == "low":
        return _svf_lowpass(x64, c64, k_arr, sr, n_stages).astype(np.float32)
    else:
        return _svf_highpass(x64, c64, k_arr, sr, n_stages).astype(np.float32)


def _svf_filter(data, order, btype, cutoff, resonance, nyquist):
    """SVF replacement for _modulated_filter."""
    sr = nyquist * 2.0
    n = len(data)
    cutoff_arr = np.clip(_as_array(cutoff, n), 1.0, nyquist - 1.0)
    res_arr = np.clip(_as_array(resonance, n), 0.0, 1.0)
    k_arr = (2.0 * (1.0 - res_arr)).astype(np.float64)

    if data.ndim == 2:
        out = np.empty_like(data, dtype=np.float32)
        for c in range(data.shape[1]):
            out[:, c] = _svf_filter_mono(data[:, c], order, btype, cutoff_arr, k_arr, sr)
        return out
    return _svf_filter_mono(data, order, btype, cutoff_arr, k_arr, sr)


def _svf_bandpass_filter(data, order, low, high, resonance, nyquist):
    """SVF replacement for _modulated_bandpass."""
    sr = nyquist * 2.0
    n = len(data)
    low_arr = np.clip(_as_array(low, n), 1.0, nyquist - 2.0).astype(np.float64)
    high_arr = np.clip(_as_array(high, n), low_arr + 1.0, nyquist - 1.0).astype(np.float64)
    res_arr = np.clip(_as_array(resonance, n), 0.0, 1.0)
    k_arr = (2.0 * (1.0 - res_arr)).astype(np.float64)
    n_stages = max(1, order // 2)

    if data.ndim == 2:
        out = np.empty_like(data, dtype=np.float32)
        for c in range(data.shape[1]):
            x64 = data[:, c].astype(np.float64)
            out[:, c] = _svf_bandpass(x64, low_arr, high_arr, k_arr, sr, n_stages).astype(np.float32)
        return out
    x64 = data.astype(np.float64)
    return _svf_bandpass(x64, low_arr, high_arr, k_arr, sr, n_stages).astype(np.float32)


class LowPassFilter(Effect):
    """Butterworth low-pass filter.

    Attenuates frequencies above ``cutoff_hz``.

    Parameters
    ----------
    cutoff_hz:
        Cutoff frequency in Hz. Accepts a constant ``float`` or a time-varying
        ``Signal`` for a modulated filter sweep. When a Signal, uses a
        per-sample SVF (state variable filter) for smooth, artifact-free
        modulation.
    resonance:
        Resonance amount from 0.0 (Butterworth, no peak) to 1.0
        (self-oscillation). Accepts a constant ``float`` or a time-varying
        ``Signal``. Any non-zero value activates the SVF engine.
    """

    def __init__(self, cutoff_hz: float | Signal, order: int = 4, *, resonance: float | Signal = 0.0) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order
        self.resonance = resonance

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        use_svf = isinstance(self.cutoff_hz, Signal) or isinstance(self.resonance, Signal) or self.resonance != 0.0
        if use_svf:
            return Signal(
                _svf_filter(sig.data, self.order, "low", self.cutoff_hz, self.resonance, nyquist),
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
    resonance:
        Resonance amount from 0.0 (Butterworth, no peak) to 1.0
        (self-oscillation). Accepts a constant ``float`` or a time-varying
        ``Signal``. Any non-zero value activates the SVF engine.
    """

    def __init__(self, cutoff_hz: float | Signal, order: int = 4, *, resonance: float | Signal = 0.0) -> None:
        self.cutoff_hz = cutoff_hz
        self.order = order
        self.resonance = resonance

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        use_svf = isinstance(self.cutoff_hz, Signal) or isinstance(self.resonance, Signal) or self.resonance != 0.0
        if use_svf:
            return Signal(
                _svf_filter(sig.data, self.order, "high", self.cutoff_hz, self.resonance, nyquist),
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
    resonance:
        Resonance amount from 0.0 (Butterworth, no peak) to 1.0
        (self-oscillation). Accepts a constant ``float`` or a time-varying
        ``Signal``. Any non-zero value activates the SVF engine.
    """

    def __init__(self, low_hz: float | Signal, high_hz: float | Signal, order: int = 4, *, resonance: float | Signal = 0.0) -> None:
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order
        self.resonance = resonance

    def __call__(self, sig: Signal) -> Signal:
        nyquist = sig.sample_rate / 2.0
        use_svf = isinstance(self.low_hz, Signal) or isinstance(self.high_hz, Signal) or isinstance(self.resonance, Signal) or self.resonance != 0.0
        if use_svf:
            return Signal(
                _svf_bandpass_filter(sig.data, self.order, self.low_hz, self.high_hz, self.resonance, nyquist),
                sig.sample_rate,
            )
        b, a = scipy_signal.butter(
            self.order,
            [self.low_hz / nyquist, self.high_hz / nyquist],
            btype="band",
        )
        data = scipy_signal.filtfilt(b, a, sig.data).astype(np.float32)
        return Signal(data, sig.sample_rate)
