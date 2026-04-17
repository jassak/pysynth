from __future__ import annotations

import numba
import numpy as np

from pysynth._core import Effect, Signal, _as_array


@numba.njit(cache=True)
def _fixed_delay_feedback(out, feedback, delay_samples, start, end):
    for i in range(start, end):
        out[i] += feedback * out[i - delay_samples]


@numba.njit(cache=True)
def _modulated_delay_loop(out, delay_arr, sr, feedback):
    n = len(out)
    for i in range(n):
        d = delay_arr[i] * sr
        if d < 1.0:
            d = 1.0
        if d > float(i):
            d = float(i)
        d_int = int(d)
        frac = d - d_int
        j_lo = i - d_int
        j_hi = j_lo - 1
        s_lo = out[j_lo] if j_lo >= 0 else 0.0
        s_hi = out[j_hi] if j_hi >= 0 else 0.0
        delayed = (1.0 - frac) * s_lo + frac * s_hi
        out[i] += feedback * delayed


class Delay(Effect):
    """Single-tap delay with feedback.

    Parameters
    ----------
    delay_time:
        Delay time in seconds.
    feedback:
        Fraction of the delayed signal fed back into the delay line (0..1).
    wet:
        Mix of delayed signal (0 = dry only, 1 = wet only).
    """

    def __init__(
        self,
        delay_time: float | Signal,
        feedback: float = 0.4,
        wet: float = 0.5,
    ) -> None:
        self.delay_time = delay_time
        self.feedback = np.clip(feedback, 0.0, 0.99)
        self.wet = np.clip(wet, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        if isinstance(self.delay_time, Signal):
            return Signal(
                _modulated_delay(sig.data, self.delay_time, self.feedback, self.wet, sig.sample_rate),
                sig.sample_rate,
            )
        delay_samples = int(self.delay_time * sig.sample_rate)
        x = sig.data
        if x.ndim == 2:
            channels = [
                _fixed_delay_mono(x[:, c], self.feedback, delay_samples, self.wet)
                for c in range(x.shape[1])
            ]
            return Signal(np.column_stack(channels).astype(np.float32), sig.sample_rate)
        return Signal(_fixed_delay_mono(x, self.feedback, delay_samples, self.wet), sig.sample_rate)


def _fixed_delay_mono(x: np.ndarray, feedback: float, delay_samples: int, wet: float) -> np.ndarray:
    n = len(x)
    buf = np.zeros(n + delay_samples, dtype=np.float32)
    buf[:n] = x
    out = buf.copy()
    _fixed_delay_feedback(out, feedback, delay_samples, delay_samples, n + delay_samples)
    out = out[:n]
    return ((1.0 - wet) * x + wet * out).astype(np.float32)


def _modulated_delay(
    x: np.ndarray,
    delay_sig: Signal,
    feedback: float,
    wet: float,
    sr: int,
) -> np.ndarray:
    """Per-sample feedback delay with linearly interpolated time-varying delay time.

    Because each output sample feeds back into subsequent reads, this loop is
    inherently sequential and cannot be vectorised.
    """
    n = len(x)
    delay_arr = _as_array(delay_sig, n)
    if x.ndim == 2:
        channels = []
        for c in range(x.shape[1]):
            x64 = x[:, c].astype(np.float64)
            out = x64.copy()
            _modulated_delay_loop(out, delay_arr, sr, feedback)
            channels.append(((1.0 - wet) * x64 + wet * out).astype(np.float32))
        return np.column_stack(channels)
    x64 = x.astype(np.float64)
    out = x64.copy()
    _modulated_delay_loop(out, delay_arr, sr, feedback)
    mixed = (1.0 - wet) * x64 + wet * out
    return mixed.astype(np.float32)


def _echo_mono(x: np.ndarray, n: int, extra: int, delay_samples: int, repeats: int, decay: float, wet: float) -> np.ndarray:
    out = np.zeros(n + extra, dtype=np.float32)
    out[:n] = x
    amp = decay
    for tap in range(1, repeats + 1):
        offset = delay_samples * tap
        out[offset : offset + n] += x * amp
        amp *= decay
    out = out[:n]
    return ((1.0 - wet) * x + wet * out).astype(np.float32)


class Echo(Effect):
    """Multi-tap echo: a fixed number of evenly-spaced repeats.

    Unlike Delay, Echo does not feed back; each tap decays by ``decay``
    relative to the previous.
    """

    def __init__(
        self,
        delay_time: float,
        repeats: int = 4,
        decay: float = 0.5,
        wet: float = 0.6,
    ) -> None:
        self.delay_time = delay_time
        self.repeats = repeats
        self.decay = decay
        self.wet = np.clip(wet, 0.0, 1.0)

    def __call__(self, sig: Signal) -> Signal:
        delay_samples = int(self.delay_time * sig.sample_rate)
        x = sig.data
        n = len(x)
        extra = delay_samples * self.repeats
        if x.ndim == 2:
            channels = [
                _echo_mono(x[:, c], n, extra, delay_samples, self.repeats, self.decay, self.wet)
                for c in range(x.shape[1])
            ]
            return Signal(np.column_stack(channels).astype(np.float32), sig.sample_rate)
        return Signal(_echo_mono(x, n, extra, delay_samples, self.repeats, self.decay, self.wet), sig.sample_rate)
