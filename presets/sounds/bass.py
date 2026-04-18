"""Bass instrument presets."""

from __future__ import annotations

from pysynth import Oscillator
from pysynth.envelopes import adsr

from presets.sounds.patch import Patch


def acid_bass(
    attack: float = 0.005,
    decay: float = 0.1,
    sustain: float = 0.8,
    release: float = 0.05,
) -> Patch:
    """TB-303 style acid bass. Raw saw wave — apply your own filter
    and distortion to taste."""
    env = adsr(attack, decay, sustain, release)

    def synth(hz, gate, _env=env):
        return Oscillator("saw").render(gate.duration, hz) * 0.5 * _env.trigger(gate)

    return Patch(synth=synth, name="acid_bass")


def sub_bass(
    attack: float = 0.005,
    decay: float = 0.1,
    sustain: float = 0.6,
    release: float = 0.15,
) -> Patch:
    """Pure sub-bass sine."""
    env = adsr(attack, decay, sustain, release)

    def synth(hz, gate, _env=env):
        return Oscillator("sine").render(gate.duration, hz) * 0.6 * _env.trigger(gate)

    return Patch(synth=synth, name="sub_bass")


def detuned_saw(
    detune: float = 0.004,
    attack: float = 0.005,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.1,
) -> Patch:
    """Two slightly detuned saw waves — fat unison sound.

    *detune* is the relative frequency offset between the two oscillators
    (0.004 = ±0.2%).
    """
    env = adsr(attack, decay, sustain, release)

    def synth(hz, gate, _d=detune, _env=env):
        a = Oscillator("saw").render(gate.duration, hz * (1 + _d))
        b = Oscillator("saw").render(gate.duration, hz * (1 - _d))
        return (a + b) * 0.35 * _env.trigger(gate)

    return Patch(synth=synth, name="detuned_saw")
