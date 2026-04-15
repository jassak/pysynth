"""Keyboard instrument presets: organ, electric piano."""

from __future__ import annotations

from pysynth import Oscillator
from pysynth.envelopes import adsr

from presets.sounds.patch import Patch


def organ(
    harmonics: tuple[float, ...] = (0.80, 0.60, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05),
    attack: float = 0.01,
    decay: float = 0.05,
    sustain: float = 0.35,
    sustain_level: float = 0.7,
    release: float = 0.08,
) -> Patch:
    """Hammond-style drawbar organ.

    Each entry in *harmonics* controls the amplitude of the corresponding
    harmonic partial (1st, 2nd, 3rd, ...).
    """
    osc = Oscillator("sine", ratio=1) * harmonics[0]
    for i, level in enumerate(harmonics[1:], start=2):
        osc = osc + Oscillator("sine", ratio=i) * level
    return Patch(
        synth=lambda hz, dur, _osc=osc: _osc.at(hz).render(dur) * 0.25,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="organ",
    )


def electric_piano(
    brightness: float = 0.5,
    attack: float = 0.005,
    decay: float = 0.3,
    sustain: float = 0.4,
    sustain_level: float = 0.5,
    release: float = 0.2,
) -> Patch:
    """Rhodes-style electric piano.

    *brightness* controls the 2nd harmonic level (0.0 = pure sine,
    1.0 = strong overtone).
    """
    osc = Oscillator("sine") + Oscillator("sine", ratio=2) * brightness
    return Patch(
        synth=lambda hz, dur, _osc=osc: _osc.at(hz).render(dur) * 0.4,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="electric_piano",
    )
