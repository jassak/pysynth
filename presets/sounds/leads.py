"""Lead and pluck presets."""

from __future__ import annotations

from pysynth import Oscillator
from pysynth.envelopes import adsr

from presets.sounds.patch import Patch


def pluck(
    waveform: str = "sine",
    attack: float = 0.005,
    decay: float = 0.2,
    sustain: float = 0.0,
    sustain_level: float = 0.0,
    release: float = 0.01,
) -> Patch:
    """Short plucked tone. One-shot envelope, no sustain."""
    osc = Oscillator(waveform)
    return Patch(
        synth=lambda hz, dur, _osc=osc: _osc.at(hz).render(dur) * 0.5,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="pluck",
    )


def bell(
    mod_ratio: float = 2.0,
    mod_index: float = 3.5,
    attack: float = 0.01,
    decay: float = 0.4,
    sustain: float = 0.0,
    sustain_level: float = 0.0,
    release: float = 0.01,
) -> Patch:
    """FM bell: carrier + modulator at *mod_ratio* times the fundamental.

    *mod_index* controls brightness (higher = more metallic).
    """
    def synth(hz, dur, _mr=mod_ratio, _mi=mod_index):
        mod = Oscillator("sine").at(hz * _mr).render(dur) * (hz * _mi)
        return Oscillator("sine").at(hz + mod).render(dur)
    return Patch(
        synth=synth,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="bell",
    )


def fm_bass(
    mod_ratio: float = 0.5,
    mod_index: float = 2.0,
    attack: float = 0.005,
    decay: float = 0.15,
    sustain: float = 0.3,
    sustain_level: float = 0.6,
    release: float = 0.1,
) -> Patch:
    """Wobbly FM sub-bass."""
    def synth(hz, dur, _mr=mod_ratio, _mi=mod_index):
        mod = Oscillator("sine").at(hz * _mr).render(dur) * (hz * _mi)
        return Oscillator("sine").at(hz + mod).render(dur)
    return Patch(
        synth=synth,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="fm_bass",
    )


def fm_metal(
    mod_ratio: float = 3.14,
    mod_index: float = 8.0,
    attack: float = 0.005,
    decay: float = 0.3,
    sustain: float = 0.0,
    sustain_level: float = 0.0,
    release: float = 0.01,
) -> Patch:
    """Inharmonic FM: irrational ratio makes it metallic."""
    def synth(hz, dur, _mr=mod_ratio, _mi=mod_index):
        mod = Oscillator("sine").at(hz * _mr).render(dur) * (hz * _mi)
        return Oscillator("sine").at(hz + mod).render(dur)
    return Patch(
        synth=synth,
        envelope=adsr(attack, decay, sustain, sustain_level, release),
        name="fm_metal",
    )
