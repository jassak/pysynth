"""The Patch class: a synth callable paired with an amplitude envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pysynth import Signal
from pysynth.envelopes import Envelope


@dataclass(frozen=True)
class Patch:
    """A synthesizer voice: a sound generator + an amplitude envelope.

    The ``synth`` callable takes ``(pitch, dur) -> Signal``. It can wrap
    anything — a simple oscillator, an FM network, a wavetable morph, etc.

    Quick use::

        patch.preview()                  # hear at 220 Hz, 3 s
        patch.render(440, dur=2.0)       # fixed pitch

    CV/gate (from a Sequencer)::

        pitch, gate = Sequencer(notes, bpm=120).cv()
        patch.render(pitch, gate)

    Parameters
    ----------
    synth:
        ``(pitch, dur) -> Signal``.  *pitch* is a float (Hz) or a
        time-varying ``Signal`` (pitch CV).  *dur* is seconds.
    envelope:
        Amplitude envelope applied after the synth.  In CV/gate mode
        the envelope is ``.trigger(gate)``-ed; in simple mode it is
        ``.apply()``-ed.
    name:
        Human-readable label (for ``repr``).
    """

    synth: Callable[[float | Signal, float], Signal]
    envelope: Envelope | None = None
    name: str = ""

    def render(
        self,
        pitch: float | Signal,
        gate: Signal | None = None,
        dur: float | None = None,
    ) -> Signal:
        """Render audio.

        Two calling conventions:

        ``render(440, dur=2.0)``
            Fixed pitch, explicit duration.  Envelope (if any) is applied
            with ``.apply()``.

        ``render(pitch_cv, gate_cv)``
            CV/gate from a ``Sequencer``.  Duration is taken from the
            signals.  Envelope is triggered by the gate.
        """
        if gate is not None:
            d = pitch.duration if isinstance(pitch, Signal) else gate.duration
            audio = self.synth(pitch, d)
            if self.envelope is not None:
                audio = audio * self.envelope.trigger(gate)
            return audio

        if dur is None:
            raise ValueError("dur is required when gate is not provided")
        audio = self.synth(pitch, dur)
        if self.envelope is not None:
            audio = self.envelope.apply(audio)
        return audio

    def preview(self, hz: float = 220.0, dur: float = 3.0) -> None:
        """Play at a fixed pitch through the default audio device."""
        self.render(hz, dur=dur).play()

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        env = "yes" if self.envelope is not None else "no"
        return f"Patch({label!r}, envelope={env})"
