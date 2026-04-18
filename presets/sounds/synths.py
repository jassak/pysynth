"""Synth protocol and synthesizer classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pysynth import Oscillator, Signal
from pysynth.envelopes import Envelope


@runtime_checkable
class Synth(Protocol):
    """A synthesizer callable: ``(hz, gate) -> Signal``.

    Any object whose ``__call__`` matches this signature satisfies the
    protocol and can be passed as ``Patch(synth=..., ...)``.
    """

    def __call__(self, hz: float | Signal, gate: Signal) -> Signal: ...


@dataclass(frozen=True)
class _Modulator:
    waveform: str
    ratio: float
    index: float
    envelope: Envelope | None


class FMSynth:
    """FM synthesizer with unbounded operator chaining.

    The constructor sets the carrier waveform and an optional amplitude
    envelope.  Each ``.chain()`` call adds a modulator that frequency-
    modulates the previous operator, with an optional modulation-depth
    envelope.

    Instances follow the :class:`Synth` protocol and can be used
    directly as ``Patch(synth=fm, ...)``.

    Examples::

        # Simple two-operator FM (constant modulation, no amp envelope)
        fm = FMSynth("sine").chain("sine", ratio=2, index=3)

        # Three-operator chain with envelopes:
        # saw modulates triangle, which modulates sine (carrier)
        fm = (FMSynth("sine", envelope=adsr(0.01, 0.1, 0.7, 0.1))
              .chain("triangle", ratio=2, index=3,
                     envelope=adsr(0.1, 0.5, 0.0, 0.01))
              .chain("saw", ratio=3, index=1.2,
                     envelope=adsr(0.2, 0.3, 0.0, 0.01)))

        # Use with Patch
        patch = Patch(synth=fm, name="my_fm")
    """

    def __init__(
        self,
        waveform: str = "sine",
        *,
        envelope: Envelope | None = None,
    ) -> None:
        self._carrier: str = waveform
        self._envelope: Envelope | None = envelope
        self._modulators: tuple[_Modulator, ...] = ()

    def chain(
        self,
        waveform: str = "sine",
        *,
        ratio: float = 1.0,
        index: float = 1.0,
        envelope: Envelope | None = None,
    ) -> FMSynth:
        """Add a modulator that modulates the previous operator.

        Parameters
        ----------
        waveform:
            Waveform of this modulator operator.
        ratio:
            Frequency ratio relative to the fundamental.
        index:
            Modulation index — peak frequency deviation is ``index * hz``.
        envelope:
            Optional envelope for the modulation depth, triggered by the
            gate.  If ``None``, modulation depth is constant.
        """
        new = FMSynth.__new__(FMSynth)
        new._carrier = self._carrier
        new._envelope = self._envelope
        new._modulators = self._modulators + (
            _Modulator(waveform, ratio, index, envelope),
        )
        return new

    def __call__(self, hz: float | Signal, gate: Signal) -> Signal:
        dur = gate.duration
        mod_signal = None

        for mod in reversed(self._modulators):
            freq = hz * mod.ratio
            if mod_signal is not None:
                freq = freq + mod_signal

            raw = Oscillator(mod.waveform).render(dur, freq)
            deviation = hz * mod.index

            if mod.envelope is not None:
                env = mod.envelope.trigger(gate)
                mod_signal = raw * env * deviation
            else:
                mod_signal = raw * deviation

        carrier_freq = hz + mod_signal if mod_signal is not None else hz
        audio = Oscillator(self._carrier).render(dur, carrier_freq)

        if self._envelope is not None:
            audio = audio * self._envelope.trigger(gate)

        return audio

    def __repr__(self) -> str:
        ops = [self._carrier] + [m.waveform for m in self._modulators]
        return f"FMSynth({' \u2192 '.join(ops)})"
