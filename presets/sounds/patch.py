"""The Patch class: a named synth with convenience methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pysynth import SAMPLE_RATE, Signal


@dataclass(frozen=True)
class Patch:
    """A synthesizer voice ready to play.

    The ``synth`` callable takes ``(pitch, gate) -> Signal``.  It owns
    its envelopes internally — the gate triggers them.

    Quick use::

        patch.preview()                  # hear at 220 Hz, 3 s
        patch.render(440, dur=2.0)       # fixed pitch, held for 2 s

    CV/gate (from a Sequencer)::

        pitch, gate = Sequencer(notes, bpm=120).cv()
        patch.render(pitch, gate)

    Parameters
    ----------
    synth:
        ``(pitch, gate) -> Signal``.  *pitch* is a float (Hz) or a
        time-varying ``Signal`` (pitch CV).  *gate* is a ``Signal``
        controlling note on/off and velocity.
    name:
        Human-readable label (for ``repr``).
    """

    synth: Callable[[float | Signal, Signal], Signal]
    name: str = ""

    def render(
        self,
        pitch: float | Signal,
        gate: Signal | None = None,
        dur: float | None = None,
    ) -> Signal:
        """Render audio.

        Two calling conventions:

        ``render(pitch_cv, gate_cv)``
            CV/gate from a ``Sequencer``.  The gate triggers the synth's
            internal envelopes.

        ``render(440, dur=2.0)``
            Fixed pitch.  An all-ones gate is created for *dur* seconds
            (equivalent to holding a key).
        """
        if gate is not None:
            return self.synth(pitch, gate)

        if dur is None:
            raise ValueError("dur is required when gate is not provided")
        n = int(dur * SAMPLE_RATE)
        gate = Signal(np.ones(n, dtype=np.float32))
        return self.synth(pitch, gate)

    def preview(self, hz: float = 220.0, dur: float = 3.0, tail: float = 2.0) -> None:
        """Play at a fixed pitch through the default audio device.

        The gate is held high for *dur* seconds, then drops to zero for
        *tail* seconds so the release phase of the envelope plays out.
        """
        n_high = int(dur * SAMPLE_RATE)
        n_tail = int(tail * SAMPLE_RATE)
        gate_data = np.zeros(n_high + n_tail, dtype=np.float32)
        gate_data[:n_high] = 1.0
        sig = self.synth(hz, Signal(gate_data))
        # Trim trailing silence so we don't wait past the release.
        last = np.max(np.nonzero(sig.data)[0], initial=0) + 1
        sig = Signal(sig.data[:last], sig.sample_rate)
        sig.normalize().play()

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        return f"Patch({label!r})"
