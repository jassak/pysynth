"""Memoryless nonlinear waveshapers.

A waveshaper applies a static transfer function ``y = f(x)`` sample-by-sample.
The shape of ``f`` determines the harmonic content added to the signal.

The Chebyshev trick
-------------------
Feeding a pure sine ``x = cos(θ)`` into the k-th Chebyshev polynomial of the
first kind gives ``T_k(cos θ) = cos(k θ)`` — exactly the k-th harmonic. So
``Chebyshev([0, 0, 0, 1])`` applied to a unit-amplitude sine at ``f`` produces
a pure sine at ``3f``. Linear combinations give any desired harmonic mix:

    >>> from pysynth import Oscillator, Chebyshev
    >>> # Pure 3rd harmonic of 440 Hz → 1320 Hz sine
    >>> sig = Oscillator("sine").render(1.0, 440.0) | Chebyshev([0, 0, 0, 1])

Aliasing
--------
None of these shapers oversample. Feeding a high-frequency signal through a
strong nonlinearity will alias. For clean results, keep input below
``sample_rate / (2 * highest_harmonic)`` or apply your own oversampling.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from pysynth._core import Effect, Signal, _as_array


class Tanh(Effect):
    """Soft saturation via hyperbolic tangent.

    ``drive`` controls how hard the signal is pushed into the nonlinearity.
    Higher drive = more harmonic content and compression.
    """

    def __init__(self, drive: float | Signal = 1.0) -> None:
        self.drive = drive

    def __call__(self, sig: Signal) -> Signal:
        drive = _as_array(self.drive, len(sig.data)) if isinstance(self.drive, Signal) else self.drive
        data = np.tanh(sig.data * drive).astype(np.float32)
        return Signal(data, sig.sample_rate)


class Clip(Effect):
    """Hard clipping distortion.

    Flattens the waveform above ``threshold``, producing harsh, bright harmonics.
    ``threshold`` is a linear amplitude value (0..1).
    """

    def __init__(self, threshold: float | Signal = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, sig: Signal) -> Signal:
        if isinstance(self.threshold, Signal):
            threshold = np.clip(_as_array(self.threshold, len(sig.data)), 0.0, 1.0)
        else:
            threshold = np.clip(self.threshold, 0.0, 1.0)
        data = np.clip(sig.data, -threshold, threshold).astype(np.float32)
        # Renormalise so output peak matches input peak
        peak = np.max(np.abs(sig.data))
        if peak > 0:
            data = (data / threshold * peak).astype(np.float32)
        return Signal(data, sig.sample_rate)


class SoftClip(Effect):
    """Rational soft clipper: ``y = x*drive / (1 + |x*drive|)``.

    Odd-symmetric — produces only odd harmonics, no DC. A cheaper alternative
    to :class:`Tanh` with a slightly different curve (approaches ±1 more slowly).
    """

    def __init__(self, drive: float | Signal = 1.0) -> None:
        self.drive = drive

    def __call__(self, sig: Signal) -> Signal:
        drive = _as_array(self.drive, len(sig.data)) if isinstance(self.drive, Signal) else self.drive
        x = sig.data * drive
        data = (x / (1.0 + np.abs(x))).astype(np.float32)
        return Signal(data, sig.sample_rate)


class Fold(Effect):
    """Triangular wavefolder.

    When the input exceeds ``threshold``, it reflects back instead of clipping,
    producing a rich odd-harmonic spectrum that grows with drive. A staple of
    West-Coast synthesis.
    """

    def __init__(self, threshold: float | Signal = 1.0) -> None:
        self.threshold = threshold

    def __call__(self, sig: Signal) -> Signal:
        n = len(sig.data)
        if isinstance(self.threshold, Signal):
            t = np.maximum(_as_array(self.threshold, n), 1e-6)
        else:
            t = max(float(self.threshold), 1e-6)
        x = sig.data
        # Triangular fold: identity for |x|<=t, reflects off ±t thereafter.
        data = (np.abs(np.mod(x - t, 4.0 * t) - 2.0 * t) - t).astype(np.float32)
        return Signal(data, sig.sample_rate)


class Rectifier(Effect):
    """Half- or full-wave rectification.

    ``mode="full"`` outputs ``|x|``; ``mode="half"`` outputs ``max(x, 0)``.
    Both introduce DC, which is removed before peak-normalizing back to the
    input's peak amplitude.
    """

    def __init__(self, mode: str = "full") -> None:
        if mode not in ("full", "half"):
            raise ValueError(f"mode must be 'full' or 'half', got {mode!r}")
        self.mode = mode

    def __call__(self, sig: Signal) -> Signal:
        x = sig.data
        if self.mode == "full":
            y = np.abs(x)
        else:
            y = np.maximum(x, 0.0)
        y = y - y.mean()
        in_peak = float(np.max(np.abs(x)))
        out_peak = float(np.max(np.abs(y)))
        if out_peak > 0 and in_peak > 0:
            y = y * (in_peak / out_peak)
        return Signal(y.astype(np.float32), sig.sample_rate)


class Chebyshev(Effect):
    """Chebyshev polynomial waveshaper: ``y = Σ a_k · T_k(x)``.

    Coefficients use mathematical indexing — ``coeffs[0]`` multiplies ``T_0``
    (constant 1, i.e. DC), ``coeffs[1]`` multiplies ``T_1(x) = x``, etc. For a
    pure sine input at amplitude 1, each ``T_k`` maps to exactly the k-th
    harmonic, so ``coeffs = [0, 0, 0, 1]`` yields the 3rd harmonic alone.

    Any coefficient may be a :class:`Signal` for time-varying harmonic
    morphing. Input is expected in ``[-1, 1]``; this is not checked. Output
    has its mean removed and is peak-normalized back to the input peak.
    """

    def __init__(self, coeffs: Sequence[float | Signal]) -> None:
        if len(coeffs) == 0:
            raise ValueError("coeffs must be non-empty")
        self.coeffs = list(coeffs)

    def __call__(self, sig: Signal) -> Signal:
        n = len(sig.data)
        x = sig.data.astype(np.float64)
        any_signal = any(isinstance(c, Signal) for c in self.coeffs)

        if not any_signal:
            y = np.polynomial.chebyshev.chebval(x, [float(c) for c in self.coeffs])
        else:
            # Per-sample Clenshaw recurrence with broadcast coefficients.
            # T_0 = 1, T_1 = x, T_k = 2x·T_{k-1} − T_{k-2}
            y = np.zeros(n, dtype=np.float64)
            t_prev = np.ones(n, dtype=np.float64)  # T_0
            a0 = _as_array(self.coeffs[0], n) if isinstance(self.coeffs[0], Signal) else np.full(n, float(self.coeffs[0]))
            y += a0 * t_prev
            if len(self.coeffs) > 1:
                t_curr = x.copy()  # T_1
                a1 = _as_array(self.coeffs[1], n) if isinstance(self.coeffs[1], Signal) else np.full(n, float(self.coeffs[1]))
                y += a1 * t_curr
                for c in self.coeffs[2:]:
                    t_next = 2.0 * x * t_curr - t_prev
                    ak = _as_array(c, n) if isinstance(c, Signal) else np.full(n, float(c))
                    y += ak * t_next
                    t_prev, t_curr = t_curr, t_next

        y = y - y.mean()
        in_peak = float(np.max(np.abs(sig.data)))
        out_peak = float(np.max(np.abs(y)))
        if out_peak > 0 and in_peak > 0:
            y = y * (in_peak / out_peak)
        return Signal(y.astype(np.float32), sig.sample_rate)


class Shaper(Effect):
    """Generic waveshaper — apply an arbitrary function sample-by-sample.

    ``fn`` receives the raw sample array and returns a transformed array of the
    same length. No normalization or DC removal is performed; the caller owns
    the transfer function entirely.

    Example — odd-symmetric squared-magnitude shaping::

        Shaper(lambda x: np.sign(x) * x**2)
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.fn = fn

    def __call__(self, sig: Signal) -> Signal:
        out = np.asarray(self.fn(sig.data), dtype=np.float32)
        return Signal(out, sig.sample_rate)


class Overdrive(Effect):
    """Asymmetric soft-clipping overdrive, inspired by tube amplifier characteristics.

    Applies piecewise soft clipping with a bias offset to introduce even-order
    harmonics (2nd, 4th, ...), which are warmer than the odd-order harmonics
    produced by symmetric clipping.
    """

    def __init__(self, gain: float | Signal = 4.0, bias: float | Signal = 0.1) -> None:
        self.gain = gain
        self.bias = bias

    def __call__(self, sig: Signal) -> Signal:
        n = len(sig.data)
        gain = _as_array(self.gain, n) if isinstance(self.gain, Signal) else self.gain
        bias = _as_array(self.bias, n) if isinstance(self.bias, Signal) else self.bias
        x = sig.data * gain + bias
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
