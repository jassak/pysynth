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


class _Voice:
    """A pitched sound awaiting a duration. Private — only produced by ``Oscillator.at(hz)``.

    Holds the oscillator's component list and the bound frequency.
    ``render(dur)`` performs the actual synthesis loop directly.

    The generator protocol is duck-typed: anything with
    ``.render(dur, sample_rate) -> Signal`` is a valid voice. Custom generators
    (FM, etc.) can define their own class without depending on this one.
    """

    def __init__(
        self,
        components: list[tuple[Waveform, float, float, float]],
        hz: float | Signal,
    ) -> None:
        self._components = components
        self._hz = hz

    def render(self, dur: float, sample_rate: int = SAMPLE_RATE) -> Signal:
        n = int(dur * sample_rate)
        buf = np.zeros(n, dtype=np.float64)
        for waveform, ratio, amplitude, phase in self._components:
            buf += _render_component(waveform, self._hz * ratio, amplitude, phase, n, sample_rate)
        return Signal(buf.astype(np.float32), sample_rate)


class _EnvelopedVoice:
    """Voice produced by an oscillator with a duck-typed envelope applied. Private."""

    def __init__(self, voice: _Voice, envelope) -> None:
        self._voice = voice
        self._envelope = envelope

    def render(self, dur: float, sample_rate: int = SAMPLE_RATE) -> Signal:
        sig = self._voice.render(dur, sample_rate)
        return self._envelope.apply(sig)


class _EnvelopedOscillator:
    """Generator produced by ``Oscillator * envelope``.

    Satisfies the generator protocol: ``.at(hz) -> voice`` where
    ``voice.render(dur, sr) -> Signal``.

    The envelope is re-rendered per note at the correct duration via its
    ``.apply(signal) -> Signal`` method — no pre-rendering required.
    Any object with ``.apply(signal) -> Signal`` is a valid envelope here.
    """

    def __init__(self, osc: "Oscillator", envelope) -> None:
        self._osc = osc
        self._envelope = envelope

    def at(self, hz: float | Signal) -> _EnvelopedVoice:
        return _EnvelopedVoice(self._osc.at(hz), self._envelope)

    def __repr__(self) -> str:
        return f"_EnvelopedOscillator({self._osc!r}, {self._envelope!r})"


class _ProductVoice:
    """Voice produced by two oscillators multiplied together. Private."""

    def __init__(self, a: _Voice, b: _Voice) -> None:
        self._a = a
        self._b = b

    def render(self, dur: float, sample_rate: int = SAMPLE_RATE) -> Signal:
        return self._a.render(dur, sample_rate) * self._b.render(dur, sample_rate)


class _ProductOscillator:
    """Oscillator-protocol object representing the pointwise product of two oscillators.

    Satisfies the generator protocol: ``.at(hz) -> voice`` where
    ``voice.render(dur, sr) -> Signal``.
    """

    def __init__(self, a: Oscillator, b: Oscillator) -> None:
        self._a = a
        self._b = b

    def at(self, hz: float | Signal) -> _ProductVoice:
        return _ProductVoice(self._a.at(hz), self._b.at(hz))

    def __repr__(self) -> str:
        return f"_ProductOscillator({self._a!r}, {self._b!r})"


class Oscillator:
    """A pitch-free, algebraically composable waveform template.

    An Oscillator defines *how* to produce a sound (waveform shape and harmonic
    structure) without committing to an absolute frequency. Frequency is
    supplied via ``.at(hz)``, which returns a voice ready to render.

    The second constructor argument, ``ratio``, is a **relative frequency
    multiplier**. ``Oscillator("sine", 2)`` renders at twice the fundamental.

    Oscillators form an algebra under ``+`` and ``*``:

        ``osc * scalar``        — scale the output amplitude (returns new Oscillator)
        ``scalar * osc``        — same (commutative)
        ``osc1 + osc2``         — sum the two waveforms (returns new Oscillator)
        ``osc1 * osc2``         — ring modulation; defers to Signal multiplication
                                  at render time (returns _ProductOscillator)
        ``osc * envelope``      — bake an envelope into the generator; envelope must
                                  implement ``.apply(signal) -> Signal``; re-rendered
                                  per note at the correct duration (returns _EnvelopedOscillator)

    These operations compose oscillator *definitions*, not rendered Signals.
    Audio is only produced after calling ``.at(hz).render(dur)``.

    Examples::

        # Additive synthesis: build a Hammond-like timbre, then render it
        hammond = (Oscillator("sine")
                 + Oscillator("sine", 2) * 0.5
                 + Oscillator("sine", 3) * 0.25
                 + Oscillator("sine", 4) * 0.125)
        sig = hammond.at(220).render(2.0)

        # FM synthesis: pass a modulator Signal as hz
        mod = Oscillator("sine").at(110).render(2.0) * 60
        sig = Oscillator("sine").at(220 + mod).render(2.0)

        # Vibrato via Signal arithmetic
        vibrato = Oscillator("sine").at(5).render(2.0) * 15 + 440
        sig = Oscillator("sine").at(vibrato).render(2.0)

        # Used in a Sequencer — Sequencer calls generator.at(hz).render(dur)
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

    def __mul__(self, other: "Oscillator | float") -> "Oscillator | _ProductOscillator | _EnvelopedOscillator":
        if isinstance(other, Oscillator):
            return _ProductOscillator(self, other)
        if hasattr(other, "apply"):  # duck-typed envelope: anything with .apply(signal) -> Signal
            return _EnvelopedOscillator(self, other)
        result = object.__new__(Oscillator)
        result._components = [(w, r, a * other, p) for w, r, a, p in self._components]
        return result

    def __rmul__(self, other: Oscillator | float) -> Oscillator | _ProductOscillator:
        if isinstance(other, Oscillator):
            return _ProductOscillator(other, self)
        return self.__mul__(other)

    def __add__(self, other: Oscillator) -> Oscillator:
        result = object.__new__(Oscillator)
        result._components = self._components + other._components
        return result

    # ------------------------------------------------------------------ #
    # Pitch application                                                    #
    # ------------------------------------------------------------------ #

    def at(self, hz: float | Signal) -> _Voice:
        """Fix the frequency, returning a voice ready to render.

        Parameters
        ----------
        hz:
            Fundamental frequency in Hz. Each component renders at ``hz * ratio``.
            Accepts a constant float or a time-varying ``Signal`` (e.g. from an
            oscillator rendered at a low rate) for vibrato, FM carrier offset,
            and portamento.
        """
        return _Voice(self._components, hz)

    def __repr__(self) -> str:
        if len(self._components) == 1:
            w, r, a, _ = self._components[0]
            return f"Oscillator({w!r}, ratio={r}, amplitude={a})"
        return f"Oscillator(components={len(self._components)})"
