"""Percussion presets: TR-808 and TR-909 drum sounds.

All functions return a ``Signal`` (not a ``Patch``) because percussion
has no pitch dimension.  Sequence them with ``Percussion(sample).trigger(gate)``.
"""

from __future__ import annotations
from argparse import Namespace

import numpy as np

from pysynth import SAMPLE_RATE, Signal, Oscillator, WhiteNoise
from pysynth.effects import (
    HighPassFilter, BandPassFilter, SimpleReverb, Clip,
)
from pysynth.envelopes import Segment, Envelope

__all__ = ["TR808", "TR909"]


# ---------------------------------------------------------------------------
# TR-808
# ---------------------------------------------------------------------------

def tr808_kick(decay: float = 0.7) -> Signal:
    """808 kick: sine with fast pitch sweep plus click transient."""
    dur = 0.05 + decay
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch_sig = Signal(np.float32(45.0 + 115.0 * np.exp(-t * 35.0)))
    body = Oscillator("sine").render(dur, pitch_sig)
    amp = Envelope([
        Segment(0.004, 0.0, 1.0, curve=-4),
        Segment(decay, 1.0, 0.0, curve=4),
    ])
    click_env = Envelope([Segment(0.003, 1.0, 0.0, curve=-6)])
    click = click_env.apply(Oscillator("sine").render(dur, 160)) * 0.6
    return amp.apply(body) + click


def tr808_snare() -> Signal:
    """808 snare: pitched sine body + bandpass-filtered noise rattle."""
    dur = 0.25
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch1 = Signal(np.float32(180.0 + 40.0 * np.exp(-t * 50.0)))
    pitch2 = Signal(np.float32(330.0 + 40.0 * np.exp(-t * 50.0)))
    body_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur - 0.001, 1.0, 0.0, curve=3),
    ])
    body = body_env.apply(
        Oscillator("sine").render(dur, pitch1) * 0.5
        + Oscillator("sine").render(dur, pitch2) * 0.3
    )
    noise_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur - 0.001, 1.0, 0.0, curve=2),
    ])
    noise = noise_env.apply(
        BandPassFilter(2000, 9000)(WhiteNoise().render(dur))
    )
    return body * 0.6 + noise * 0.5


def tr808_hihat(open: bool = False) -> Signal:
    """808 hi-hat: six square oscillators at metallic ratios."""
    dur = 0.3 if open else 0.05
    freqs = [204.5, 298.5, 366.5, 522.7, 540.5, 800.6]
    metal = sum(
        Oscillator("square").render(dur + 0.01, f) * (1.0 / len(freqs))
        for f in freqs
    )
    filtered = (HighPassFilter(7000) | BandPassFilter(7500, 12000))(metal)
    if open:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-4),
            Segment(dur - 0.001, 1.0, 0.0, curve=2),
        ])
    else:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-4),
            Segment(dur - 0.001, 1.0, 0.0, curve=-3),
        ])
    return env.apply(filtered) * 0.5


def tr808_clap() -> Signal:
    """808 clap: multiple noise bursts spread ~5 ms apart, then a tail."""
    burst_dur = 0.012
    gap = 0.005
    tail_dur = 0.18
    total = burst_dur * 4 + gap * 3 + tail_dur
    result = Signal.silence(total)
    burst_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(burst_dur - 0.001, 1.0, 0.0, curve=-2),
    ])
    for i in range(4):
        offset = int((burst_dur + gap) * i * SAMPLE_RATE)
        burst = burst_env.apply(
            BandPassFilter(1200, 3500)(WhiteNoise().render(burst_dur))
        )
        end = min(len(result.data), offset + len(burst.data))
        result.data[offset:end] += burst.data[:end - offset] * 0.7
    tail_start = int((burst_dur + gap) * 3 * SAMPLE_RATE)
    tail_env = Envelope([
        Segment(0.005, 0.8, 1.0, curve=-2),
        Segment(tail_dur - 0.005, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(800, 3500)(WhiteNoise().render(tail_dur))
    )
    end = min(len(result.data), tail_start + len(tail.data))
    result.data[tail_start:end] += tail.data[:end - tail_start] * 0.5
    return result


def tr808_cowbell() -> Signal:
    """808 cowbell: two square oscillators at 540 Hz and 800 Hz."""
    dur = 0.12
    sig = (
        Oscillator("square").render(dur, 540) * 0.5
        + Oscillator("square").render(dur, 800) * 0.5
    )
    filtered = BandPassFilter(500, 3000)(sig)
    env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(0.03, 1.0, 0.6, curve=1),
        Segment(dur - 0.031, 0.6, 0.0, curve=2),
    ])
    return env.apply(filtered) * 0.4


def tr808_tom(pitch: float = 100, decay: float = 0.3) -> Signal:
    """808 tom: sine with pitch sweep, variable tuning."""
    dur = decay + 0.01
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch_sig = Signal(np.float32(pitch + pitch * 0.5 * np.exp(-t * 30.0)))
    env = Envelope([
        Segment(0.002, 0.0, 1.0, curve=-4),
        Segment(decay, 1.0, 0.0, curve=3),
    ])
    return env.apply(Oscillator("sine").render(dur, pitch_sig))


def tr808_rimshot() -> Signal:
    """808 rimshot: short pitched triangle + noise click."""
    dur = 0.03
    tone_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-6),
        Segment(dur - 0.001, 1.0, 0.0, curve=-3),
    ])
    tone = tone_env.apply(Oscillator("triangle").render(dur, 500))
    noise = tone_env.apply(
        HighPassFilter(2000)(WhiteNoise().render(dur))
    )
    return tone * 0.6 + noise * 0.4


TR808 = Namespace(
    kick=tr808_kick,
    snare=tr808_snare,
    hihat=tr808_hihat,
    clap=tr808_clap,
    cowbell=tr808_cowbell,
    tom=tr808_tom,
    rimshot=tr808_rimshot,
)


# ---------------------------------------------------------------------------
# TR-909
# ---------------------------------------------------------------------------

def tr909_kick() -> Signal:
    """909 kick: pitch sweep via Envelope + click transient."""
    dur = 0.35
    pitch_sig = Envelope([Segment(dur, 1.0, 0.0, curve=-15)]).render(dur) * 200.0 + 50.0
    body = Oscillator("sine").render(dur, pitch_sig)
    amp = Envelope([
        Segment(0.002, 0.0, 1.0, curve=-6),
        Segment(0.30, 1.0, 0.0, curve=3),
    ])
    click_env = Envelope([Segment(0.002, 1.0, 0.0, curve=-8)])
    click = click_env.apply(WhiteNoise().render(dur)) * 0.3
    return Clip(threshold=0.9)(amp.apply(body) + click)


def tr909_snare() -> Signal:
    """909 snare: pitch drop body + high-passed noise."""
    dur = 0.2
    pitch_sig = Envelope([Segment(dur, 1.0, 0.0, curve=-10)]).render(dur) * 60.0 + 200.0
    body_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-6),
        Segment(dur - 0.001, 1.0, 0.0, curve=2),
    ])
    body = body_env.apply(Oscillator("sine").render(dur, pitch_sig))
    noise_env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur * 0.7, 1.0, 0.15, curve=2),
        Segment(dur * 0.3 - 0.001, 0.15, 0.0, curve=1),
    ])
    noise = noise_env.apply(
        HighPassFilter(3000)(WhiteNoise().render(dur))
    )
    return body * 0.5 + noise * 0.6


def tr909_hihat(open: bool = False) -> Signal:
    """909 hi-hat: six square oscillators at metallic ratios — brighter than 808."""
    dur = 0.4 if open else 0.04
    freqs = [205.3, 304.4, 369.6, 522.7, 540.5, 811.2]
    metal = sum(
        Oscillator("square").render(dur + 0.01, f) * (1.0 / len(freqs))
        for f in freqs
    )
    filtered = (HighPassFilter(9000) | BandPassFilter(9000, 14000))(metal)
    if open:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-4),
            Segment(dur * 0.6, 1.0, 0.3, curve=1),
            Segment(dur * 0.4 - 0.001, 0.3, 0.0, curve=2),
        ])
    else:
        env = Envelope([
            Segment(0.001, 0.0, 1.0, curve=-6),
            Segment(dur - 0.001, 1.0, 0.0, curve=-4),
        ])
    return env.apply(filtered) * 0.5


def tr909_clap() -> Signal:
    """909 clap with burst layers and reverb tail."""
    burst_dur = 0.008
    gap = 0.003
    tail_dur = 0.14
    step = burst_dur + gap
    burst_env = Envelope([
        Segment(0.0005, 0.0, 1.0, curve=-6),
        Segment(burst_dur - 0.0005, 1.0, 0.0, curve=-3),
    ])
    burst = burst_env.apply(
        BandPassFilter(1000, 4000)(WhiteNoise().render(burst_dur))
    ) * 0.8
    tail_env = Envelope([
        Segment(0.003, 0.7, 1.0, curve=-2),
        Segment(tail_dur - 0.003, 1.0, 0.0, curve=3),
    ])
    tail = tail_env.apply(
        BandPassFilter(1000, 5000)(WhiteNoise().render(tail_dur))
    ) * 0.5
    return SimpleReverb(room_size=0.2, wet=0.3)(
        sum(burst.shift(i * step) for i in range(4)) + tail.shift(3 * step)
    )


def tr909_ride(dur: float = 0.6) -> Signal:
    """909 ride cymbal: metallic oscillators with longer decay."""
    freqs = [205.3, 304.4, 369.6, 522.7, 540.5, 811.2, 1043.0]
    metal = sum(
        Oscillator("square").render(dur + 0.01, f) * (1.0 / len(freqs))
        for f in freqs
    )
    filtered = (HighPassFilter(5000) | BandPassFilter(6000, 11000))(metal)
    env = Envelope([
        Segment(0.001, 0.0, 1.0, curve=-4),
        Segment(dur * 0.3, 1.0, 0.4, curve=1),
        Segment(dur * 0.7 - 0.001, 0.4, 0.0, curve=2),
    ])
    return env.apply(filtered) * 0.35


TR909 = Namespace(
    kick=tr909_kick,
    snare=tr909_snare,
    hihat=tr909_hihat,
    clap=tr909_clap,
    ride=tr909_ride,
)
