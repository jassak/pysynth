from __future__ import annotations

from typing import Literal

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal


Waveform = Literal["sine", "square", "saw", "triangle", "pulse"]


def _shape(waveform: Waveform, phase: np.ndarray) -> np.ndarray:
    if waveform == "sine":
        return np.sin(phase)
    if waveform == "square":
        return np.sign(np.sin(phase))
    if waveform == "saw":
        return 2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0
    if waveform == "triangle":
        return 2.0 * np.abs(2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0) - 1.0
    if waveform == "pulse":
        return np.where((phase / (2.0 * np.pi) % 1.0) < 0.5, 1.0, -1.0)
    raise ValueError(f"Unknown waveform: {waveform!r}")


def _render_component(
    waveform: Waveform,
    freq: float | Signal,
    amplitude: float,
    phase: float,
    n: int,
    sr: int,
) -> np.ndarray:
    """Render one oscillator component to a raw float64 array of length n."""
    if isinstance(freq, Signal):
        freq_data = freq.data.astype(np.float64)
        if len(freq_data) < n:
            freq_data = np.pad(freq_data, (0, n - len(freq_data)))
        else:
            freq_data = freq_data[:n]
        # φ(t) = 2π ∫f(τ)dτ  ≈  2π cumsum(f) / sample_rate
        phase_arr = 2.0 * np.pi * np.cumsum(freq_data) / sr + phase
    else:
        t = np.arange(n, dtype=np.float64) / sr
        phase_arr = 2.0 * np.pi * float(freq) * t + phase

    return _shape(waveform, phase_arr) * amplitude


class Oscillator:
    """A pitch-free, algebraically composable waveform template.

    An Oscillator defines *how* to produce a sound (waveform shape and harmonic
    structure) without committing to an absolute frequency. Frequency is
    supplied at render time via ``render(hz, dur)``.

    The second constructor argument, ``ratio``, is a **relative frequency
    multiplier**. ``Oscillator("sine", 2)`` renders at twice the fundamental.

    Oscillators form an algebra under ``+`` and ``*``:

        ``osc * scalar``   — scale the output amplitude (returns new Oscillator)
        ``scalar * osc``   — same (commutative)
        ``osc1 + osc2``    — sum the two waveforms (returns new Oscillator)

    These operations compose oscillator *definitions*, not rendered Signals.
    Audio is only produced when ``render`` is called.

    Examples::

        # Additive synthesis: build a Hammond-like timbre as an oscillator
        hammond = (Oscillator("sine")
                 + Oscillator("sine", 2) * 0.5
                 + Oscillator("sine", 3) * 0.25
                 + Oscillator("sine", 4) * 0.125)
        sig = hammond.render(hz=220, dur=2.0)

        # FM synthesis: pass a modulator Signal as hz
        mod = Oscillator("sine").render(hz=110, dur=2.0) * 60
        carrier = Oscillator("sine").render(hz=220 + mod, dur=2.0)

        # Used in a Sequencer — no freq mutation needed
        Sequencer(notes, bpm=120).render(Oscillator("sine"), envelope=env)
    """

    # Each component: (waveform, ratio, amplitude, phase).
    # A freshly constructed Oscillator has exactly one component.
    _components: list[tuple[Waveform, float, float, float]]

    def __init__(
        self,
        waveform: Waveform = "sine",
        ratio: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        self._components = [(waveform, float(ratio), 1.0, phase)]

    # ------------------------------------------------------------------ #
    # Algebra — all operators return new Oscillator instances              #
    # ------------------------------------------------------------------ #

    def __mul__(self, scalar: float) -> Oscillator:
        result = object.__new__(Oscillator)
        result._components = [(w, r, a * scalar, p) for w, r, a, p in self._components]
        return result

    def __rmul__(self, scalar: float) -> Oscillator:
        return self.__mul__(scalar)

    def __add__(self, other: Oscillator) -> Oscillator:
        result = object.__new__(Oscillator)
        result._components = self._components + other._components
        return result

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def render(
        self,
        hz: float | Signal,
        dur: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> Signal:
        """Render to a Signal at the given fundamental frequency.

        Parameters
        ----------
        hz:
            Fundamental frequency in Hz. Each component renders at ``hz * ratio``.
            Accepts a constant float or a time-varying ``Signal`` (from an LFO,
            another oscillator, etc.) for FM/vibrato/portamento effects.
        dur:
            Duration in seconds.
        sample_rate:
            Output sample rate. Defaults to 44100 Hz.
        """
        n = int(dur * sample_rate)
        buf = np.zeros(n, dtype=np.float64)

        for waveform, ratio, amplitude, phase in self._components:
            effective_freq = hz * ratio   # float*float  or  Signal*float
            buf += _render_component(waveform, effective_freq, amplitude, phase, n, sample_rate)

        return Signal(buf.astype(np.float32), sample_rate)

    def __repr__(self) -> str:
        if len(self._components) == 1:
            w, r, a, _ = self._components[0]
            return f"Oscillator({w!r}, ratio={r}, amplitude={a})"
        return f"Oscillator(components={len(self._components)})"
