"""Pad presets: slow-attack, evolving textures."""

from __future__ import annotations

from pysynth import Oscillator
from pysynth.envelopes import adsr

from presets.sounds.patch import Patch


def ambient_pad(
    detune: float = 0.003,
    attack: float = 0.5,
    decay: float = 0.3,
    sustain: float = 4.0,
    sustain_level: float = 0.75,
    release: float = 0.7,
) -> Patch:
    """Slow detuned-saw pad.

    *detune* is the relative frequency offset between the two oscillators.
    """
    def synth(hz, dur, _d=detune):
        a = Oscillator("saw").at(hz * (1 + _d)).render(dur)
        b = Oscillator("saw").at(hz * (1 - _d)).render(dur)
        return (a + b) * 0.3
    return Patch(
        synth=synth,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="ambient_pad",
    )


def string_pad(
    n_voices: int = 4,
    detune: float = 0.006,
    attack: float = 0.4,
    decay: float = 0.2,
    sustain: float = 3.0,
    sustain_level: float = 0.8,
    release: float = 0.6,
) -> Patch:
    """Ensemble string pad: multiple detuned saw waves.

    *n_voices* controls the number of detuned layers (2–8).
    *detune* is the maximum relative spread.
    """
    import numpy as np
    offsets = np.linspace(-detune, detune, n_voices)

    def synth(hz, dur, _offsets=offsets, _n=n_voices):
        sig = sum(
            Oscillator("saw").at(hz * (1 + o)).render(dur)
            for o in _offsets
        )
        return sig * (0.3 / _n)
    return Patch(
        synth=synth,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="string_pad",
    )
