from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.pitch import Note


class Sequencer:
    """Convert a sequence of Notes into pitch and gate control signals.

    Each Note carries a pitch (Hz) and duration (beats). The Sequencer
    converts beat durations to seconds via ``bpm`` and produces two Signals
    covering the entire sequence:

    - **pitch**: frequency in Hz at each sample (0.0 during rests)
    - **gate**: note velocity (0.0–1.0) while a note is on, 0.0 otherwise

    These Signals drive oscillators, envelopes, and effects via the
    existing Signal algebra — no need to bake everything into a generator.

    Parameters
    ----------
    notes:
        Sequence of Notes to play.
    bpm:
        Tempo in beats per minute.
    retrigger_gap:
        Duration in seconds of the zero-gate gap inserted between
        consecutive non-rest notes so that ADSR envelopes re-trigger.
        Set to ``0`` to disable.

    Usage::

        from pysynth.music import Scale, Note
        from pysynth.instruments import Sequencer
        from pysynth.generators import Oscillator
        from pysynth.envelopes import adsr
        from pysynth.effects import LowPassFilter

        scale = Scale(220, [1, 5/4, 3/2, 2])
        notes = [Note(scale[i], 0.5) for i in [0, 1, 2, 3, 2, 1, 0]]

        pitch, gate = Sequencer(notes, bpm=120).cv()
        audio  = Oscillator("saw").at(pitch).render(pitch.duration)
        amp    = adsr(0.01, 0.1, 0.7, 0.1).trigger(gate)
        cutoff = adsr(0.005, 0.2, 0.0, 0.05).trigger(gate) * 3000 + 400
        output = LowPassFilter(cutoff)(audio) * amp
    """

    def __init__(
        self,
        notes: list[Note],
        bpm: float = 120.0,
        retrigger_gap: float = 0.002,
    ) -> None:
        self.notes = notes
        self.bpm = bpm
        self.retrigger_gap = retrigger_gap

    def cv(
        self,
        *,
        repeats: int = 1,
        sample_rate: int = SAMPLE_RATE,
    ) -> tuple[Signal, Signal]:
        """Return ``(pitch, gate)`` control signals for the entire sequence.

        Parameters
        ----------
        repeats:
            Number of times to repeat the note sequence.
        sample_rate:
            Sample rate for the output Signals.

        Returns
        -------
        pitch:
            Signal with frequency in Hz per sample (0.0 during rests).
        gate:
            Signal with note velocity (0.0–1.0) while a note sounds,
            0.0 during rests.
        """
        beat_duration = 60.0 / self.bpm
        notes = self.notes * repeats

        total_seconds = sum(n.duration * beat_duration for n in notes)
        total_samples = int(total_seconds * sample_rate)

        pitch_buf = np.zeros(total_samples, dtype=np.float32)
        gate_buf = np.zeros(total_samples, dtype=np.float32)

        gap_samples = int(self.retrigger_gap * sample_rate)

        offset = 0
        prev_was_note = False
        last_hz = 0.0
        for note in notes:
            dur_samples = int(note.duration * beat_duration * sample_rate)
            end = min(offset + dur_samples, total_samples)
            if not note.is_rest:
                pitch_buf[offset:end] = note.pitch.hz
                gate_buf[offset:end] = note.velocity
                # Zero out the start of this note's gate if the previous
                # note was also active, creating a falling edge so the
                # envelope re-triggers.
                if prev_was_note and gap_samples > 0:
                    gap_end = min(offset + gap_samples, end)
                    gate_buf[offset:gap_end] = 0.0
                last_hz = note.pitch.hz
                prev_was_note = True
            else:
                # Sample-and-hold: keep the last pitch through rests
                # so the oscillator continues producing audio for the
                # envelope's release phase.
                if last_hz > 0.0:
                    pitch_buf[offset:end] = last_hz
                prev_was_note = False
            offset = end

        return Signal(pitch_buf, sample_rate), Signal(gate_buf, sample_rate)
