from __future__ import annotations

from typing import Literal

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.generators.oscillators import Oscillator, Waveform


class LFO:
    """Low-frequency oscillator for modulating other parameters.

    Produces a Signal in the range [offset - depth, offset + depth].

    Typical uses::

        # Vibrato: LFO centred on the carrier frequency
        vibrato = LFO(rate=5.0, depth=15.0, offset=440.0).render(2.0)
        sig = Oscillator("sine").render(hz=vibrato, dur=2.0)

        # Tremolo: LFO as amplitude modulator (centred on 0, offset shifts range)
        tremolo = LFO(rate=4.0, depth=0.3, offset=0.7).render(2.0)
        # sig * tremolo requires element-wise Signal*Signal, not yet supported;
        # use ADSR.apply or a custom envelope for now.
    """

    def __init__(
        self,
        waveform: Waveform = "sine",
        rate: float = 1.0,
        depth: float = 1.0,
        offset: float = 0.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.waveform = waveform
        self.rate = rate
        self.depth = depth
        self.offset = offset
        self.sample_rate = sample_rate

    def render(self, duration: float) -> Signal:
        """Return a modulation Signal.

        Values are in [offset - depth, offset + depth].
        """
        raw = Oscillator(self.waveform).at(self.rate).render(duration, self.sample_rate)
        return raw * self.depth + self.offset
