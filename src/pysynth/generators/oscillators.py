from __future__ import annotations

from typing import Literal

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.generators.base import Generator


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


class Oscillator(Generator):
    """A pitch-free waveform template — the most common Generator.

    An Oscillator defines *how* to produce a sound (waveform shape and harmonic
    structure) without committing to an absolute frequency. Frequency is
    supplied at render time via ``.render(dur, hz)``.

    The second constructor argument, ``ratio``, is a **relative frequency
    multiplier**. ``Oscillator("sine", 2)`` renders at twice the fundamental.

    Inherits all algebraic operations from :class:`Generator` (``+``, ``*``,
    ``-``).  These compose generator *definitions*, not rendered Signals.
    Audio is only produced after calling ``.render(dur, hz)``.

    Examples::

        # Additive synthesis: build a Hammond-like timbre, then render it
        hammond = (Oscillator("sine")
                 + Oscillator("sine", 2) * 0.5
                 + Oscillator("sine", 3) * 0.25
                 + Oscillator("sine", 4) * 0.125)
        sig = hammond.render(2.0, hz=220)

        # FM synthesis: pass a modulator Signal as hz
        mod = Oscillator("sine").render(2.0, hz=110) * 60
        sig = Oscillator("sine").render(2.0, hz=220 + mod)

        # CV/gate sequencing
        pitch, gate = Sequencer(notes, bpm=120).cv()
        audio = Oscillator("saw").render(pitch.duration, hz=pitch)
        output = audio * adsr(0.01, 0.1, 0.7, 0.1).trigger(gate)
    """

    def __init__(
        self,
        waveform: Waveform = "sine",
        ratio: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        self._waveform = waveform
        self._ratio = float(ratio)
        self._phase = phase

    def render(self, dur: float, hz: float | Signal = 440.0, sr: int = SAMPLE_RATE, **_kwargs) -> Signal:
        """Render the oscillator at the given frequency.

        Parameters
        ----------
        dur:
            Duration in seconds.
        hz:
            Fundamental frequency in Hz, rendered at ``hz * ratio``.
            Accepts a constant float or a time-varying ``Signal`` (e.g. a pitch
            CV from a Sequencer, or a modulation signal for vibrato/FM).
        sr:
            Sample rate.
        """
        n = int(dur * sr)
        freq = hz * self._ratio

        if isinstance(freq, Signal):
            freq_data = freq.data.astype(np.float64)
            if len(freq_data) < n:
                freq_data = np.pad(freq_data, (0, n - len(freq_data)))
            else:
                freq_data = freq_data[:n]
            phase_arr = 2.0 * np.pi * np.cumsum(freq_data) / sr + self._phase
        else:
            t = np.arange(n, dtype=np.float64) / sr
            phase_arr = 2.0 * np.pi * float(freq) * t + self._phase

        data = _shape(self._waveform, phase_arr)
        return Signal(data.astype(np.float32), sr)

    def __repr__(self) -> str:
        return f"Oscillator({self._waveform!r}, ratio={self._ratio})"
