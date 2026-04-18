"""Pad presets: slow-attack, evolving textures."""

from __future__ import annotations

from pysynth import Oscillator
from pysynth.envelopes import adsr

from presets.sounds.patch import Patch


def ambient_pad(
    detune: float = 0.003,
    attack: float = 0.5,
    decay: float = 0.3,
    sustain: float = 0.75,
    release: float = 0.7,
) -> Patch:
    """Slow detuned-saw pad.

    *detune* is the relative frequency offset between the two oscillators.
    """
    env = adsr(attack, decay, sustain, release)

    def synth(hz, gate, _d=detune, _env=env):
        a = Oscillator("saw").render(gate.duration, hz * (1 + _d))
        b = Oscillator("saw").render(gate.duration, hz * (1 - _d))
        return (a + b) * 0.3 * _env.trigger(gate)

    return Patch(synth=synth, name="ambient_pad")


def string_pad(
    n_layers: int = 4,
    detune: float = 0.006,
    randomness: float = 0.0,
    attack: float = 0.4,
    decay: float = 0.2,
    sustain: float = 0.8,
    release: float = 0.6,
) -> Patch:
    """Ensemble string pad: multiple detuned saw waves.

    *n_layers* controls the number of detuned layers (2–8).
    *detune* is the maximum relative spread.
    """
    import numpy as np
    offsets = np.linspace(-detune, detune, n_layers)
    # randomize offsets
    offsets = [o + (np.random.random() - 0.5) * detune * randomness for o in offsets]

    env = adsr(attack, decay, sustain, release)

    def synth(hz, gate, _offsets=offsets, _n=n_layers, _env=env):
        sig = sum(
            Oscillator("saw").render(gate.duration, hz * (1 + o))
            for o in _offsets
        )
        return sig * (0.3 / _n) * _env.trigger(gate)

    return Patch(synth=synth, name="string_pad")
