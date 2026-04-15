"""Bass instrument presets."""

from __future__ import annotations

from pysynth import Oscillator
from pysynth.envelopes import adsr

from presets.sounds.patch import Patch


def acid_bass(
    attack: float = 0.005,
    decay: float = 0.1,
    sustain: float = 0.3,
    sustain_level: float = 0.8,
    release: float = 0.05,
) -> Patch:
    """TB-303 style acid bass. Raw saw wave — apply your own filter
    and distortion to taste."""
    return Patch(
        synth=lambda hz, dur: Oscillator("saw").at(hz).render(dur) * 0.5,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="acid_bass",
    )


def sub_bass(
    attack: float = 0.005,
    decay: float = 0.1,
    sustain: float = 0.4,
    sustain_level: float = 0.6,
    release: float = 0.15,
) -> Patch:
    """Pure sub-bass sine."""
    return Patch(
        synth=lambda hz, dur: Oscillator("sine").at(hz).render(dur) * 0.6,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="sub_bass",
    )


def detuned_saw(
    detune: float = 0.004,
    attack: float = 0.005,
    decay: float = 0.1,
    sustain: float = 0.3,
    sustain_level: float = 0.7,
    release: float = 0.1,
) -> Patch:
    """Two slightly detuned saw waves — fat unison sound.

    *detune* is the relative frequency offset between the two oscillators
    (0.004 = ±0.2%).
    """
    def synth(hz, dur, _d=detune):
        a = Oscillator("saw").at(hz * (1 + _d)).render(dur)
        b = Oscillator("saw").at(hz * (1 - _d)).render(dur)
        return (a + b) * 0.35
    return Patch(
        synth=synth,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="detuned_saw",
    )
