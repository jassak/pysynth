from __future__ import annotations

import numba
import numpy as np

from pysynth._core import Effect, Signal, _as_array


# ---------------------------------------------------------------------------
# Cytomic TPT SVF-based equalizer filters — numba-accelerated
#
# Peak (bell), low-shelf, and high-shelf filters derived from the same SVF
# state update used in filters.py.  The difference is in the output mix:
#
#   Peak:       out = x + k*(A²-1)*bp          (A = 10^(dB/40), k = 1/(Q*A))
#   Low shelf:  out = x + k*(A-1)*bp + (A²-1)*lp   (g scaled by 1/√A)
#   High shelf: out = A²*hp + k*A*bp + lp           (g scaled by √A)
#
# All kernels operate in float64 and return float64.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _svf_peak(x, center_arr, gain_db_arr, q_arr, sr):
    n = len(x)
    ic1 = 0.0
    ic2 = 0.0
    out = np.empty(n)
    pi = np.pi
    for i in range(n):
        fc = center_arr[i]
        if fc < 1.0:
            fc = 1.0
        db = gain_db_arr[i]
        A = 10.0 ** (db / 40.0)
        Q = q_arr[i]
        if Q < 0.1:
            Q = 0.1
        k = 1.0 / (Q * A)
        g = np.tan(pi * fc / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1

        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + g * a2 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2

        out[i] = x[i] + k * (A * A - 1.0) * v1
    return out


@numba.njit(cache=True)
def _svf_low_shelf(x, freq_arr, gain_db_arr, sr):
    n = len(x)
    ic1 = 0.0
    ic2 = 0.0
    out = np.empty(n)
    pi = np.pi
    k = 2.0  # Butterworth damping
    for i in range(n):
        fc = freq_arr[i]
        if fc < 1.0:
            fc = 1.0
        db = gain_db_arr[i]
        A = 10.0 ** (db / 40.0)
        g = np.tan(pi * fc / sr) / np.sqrt(A)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1

        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + g * a2 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2

        # bp = v1, lp = v2
        out[i] = x[i] + k * (A - 1.0) * v1 + (A * A - 1.0) * v2
    return out


@numba.njit(cache=True)
def _svf_high_shelf(x, freq_arr, gain_db_arr, sr):
    n = len(x)
    ic1 = 0.0
    ic2 = 0.0
    out = np.empty(n)
    pi = np.pi
    k = 2.0  # Butterworth damping
    for i in range(n):
        fc = freq_arr[i]
        if fc < 1.0:
            fc = 1.0
        db = gain_db_arr[i]
        A = 10.0 ** (db / 40.0)
        g = np.tan(pi * fc / sr) * np.sqrt(A)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1

        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + g * a2 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2

        # hp = x[i] - k*v1 - v2, bp = v1, lp = v2
        hp = x[i] - k * v1 - v2
        out[i] = A * A * hp + k * A * v1 + v2
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_svf_eq(data, sr, kernel, *param_arrays):
    """Run an SVF EQ kernel on mono or stereo data."""
    if data.ndim == 2:
        out = np.empty_like(data, dtype=np.float32)
        for c in range(data.shape[1]):
            out[:, c] = kernel(data[:, c].astype(np.float64), *param_arrays, sr).astype(np.float32)
        return out
    return kernel(data.astype(np.float64), *param_arrays, sr).astype(np.float32)


# ---------------------------------------------------------------------------
# Effect classes
# ---------------------------------------------------------------------------


class PeakFilter(Effect):
    """Parametric bell (peaking) EQ band.

    Boosts or cuts a region around ``center_hz`` by ``gain_db`` decibels
    with bandwidth controlled by ``q``.

    Parameters
    ----------
    center_hz:
        Center frequency in Hz. Accepts ``float`` or time-varying ``Signal``.
    gain_db:
        Boost (positive) or cut (negative) in dB. Accepts ``float`` or
        time-varying ``Signal``. At 0 dB the filter is a passthrough.
    q:
        Quality factor controlling bandwidth. Higher values give a narrower
        band. Accepts ``float`` or time-varying ``Signal``. Default is 1.0.
    """

    def __init__(self, center_hz: float | Signal, gain_db: float | Signal, *, q: float | Signal = 1.0) -> None:
        self.center_hz = center_hz
        self.gain_db = gain_db
        self.q = q

    def __call__(self, sig: Signal) -> Signal:
        n = len(sig)
        sr = float(sig.sample_rate)
        nyq = sr / 2.0
        center_arr = np.clip(_as_array(self.center_hz, n), 1.0, nyq - 1.0).astype(np.float64)
        gain_arr = _as_array(self.gain_db, n).astype(np.float64)
        q_arr = np.clip(_as_array(self.q, n), 0.1, 100.0).astype(np.float64)
        data = _apply_svf_eq(sig.data, sr, _svf_peak, center_arr, gain_arr, q_arr)
        return Signal(data, sig.sample_rate)


class LowShelf(Effect):
    """Low-shelf EQ filter.

    Boosts or cuts frequencies below ``freq_hz`` by ``gain_db`` decibels.

    Parameters
    ----------
    freq_hz:
        Shelf corner frequency in Hz. Accepts ``float`` or time-varying
        ``Signal``.
    gain_db:
        Boost (positive) or cut (negative) in dB. Accepts ``float`` or
        time-varying ``Signal``. At 0 dB the filter is a passthrough.
    """

    def __init__(self, freq_hz: float | Signal, gain_db: float | Signal) -> None:
        self.freq_hz = freq_hz
        self.gain_db = gain_db

    def __call__(self, sig: Signal) -> Signal:
        n = len(sig)
        sr = float(sig.sample_rate)
        nyq = sr / 2.0
        freq_arr = np.clip(_as_array(self.freq_hz, n), 1.0, nyq - 1.0).astype(np.float64)
        gain_arr = _as_array(self.gain_db, n).astype(np.float64)
        data = _apply_svf_eq(sig.data, sr, _svf_low_shelf, freq_arr, gain_arr)
        return Signal(data, sig.sample_rate)


class HighShelf(Effect):
    """High-shelf EQ filter.

    Boosts or cuts frequencies above ``freq_hz`` by ``gain_db`` decibels.

    Parameters
    ----------
    freq_hz:
        Shelf corner frequency in Hz. Accepts ``float`` or time-varying
        ``Signal``.
    gain_db:
        Boost (positive) or cut (negative) in dB. Accepts ``float`` or
        time-varying ``Signal``. At 0 dB the filter is a passthrough.
    """

    def __init__(self, freq_hz: float | Signal, gain_db: float | Signal) -> None:
        self.freq_hz = freq_hz
        self.gain_db = gain_db

    def __call__(self, sig: Signal) -> Signal:
        n = len(sig)
        sr = float(sig.sample_rate)
        nyq = sr / 2.0
        freq_arr = np.clip(_as_array(self.freq_hz, n), 1.0, nyq - 1.0).astype(np.float64)
        gain_arr = _as_array(self.gain_db, n).astype(np.float64)
        data = _apply_svf_eq(sig.data, sr, _svf_high_shelf, freq_arr, gain_arr)
        return Signal(data, sig.sample_rate)
