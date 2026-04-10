from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.pitch import Note


class Sequencer:
    """Render a sequence of Notes to a Signal.

    Each Note carries a pitch (Hz) and duration (beats). The Sequencer places
    them end-to-end in time, converting beat durations to seconds via ``bpm``.

    Parameters
    ----------
    notes:
        Ordered list of Notes to play. Use ``Note.rest(duration)`` for silence.
    bpm:
        Tempo in beats per minute. Controls the mapping from beat durations to
        wall-clock seconds. Set to 60 to treat note durations as seconds.

    Usage::

        from pysynth.music import Scale, Note, Sequencer
        from pysynth.generators import Oscillator
        from pysynth.envelopes import ADSR

        scale = Scale(220, [1, 5/4, 3/2, 2])
        notes = [Note(scale[i], 0.5) for i in [0, 1, 2, 3, 2, 1, 0]]
        env = ADSR(0.01, 0.05, 0.35, 0.7, 0.08)

        sig = Sequencer(notes, bpm=120).render(Oscillator("sine"), envelope=env)
        sig.play()
    """

    def __init__(self, notes: list[Note], bpm: float = 120.0) -> None:
        self.notes = notes
        self.bpm = bpm

    def render(
        self,
        generator,
        *,
        envelope=None,
        repeats: int = 1,
        sample_rate: int = SAMPLE_RATE,
    ) -> Signal:
        beat_duration = 60.0 / self.bpm  # seconds per beat
        notes = self.notes * repeats

        total_seconds = sum(n.duration * beat_duration for n in notes)
        total_samples = int(total_seconds * sample_rate)
        buf = np.zeros(total_samples, dtype=np.float32)

        offset = 0
        for note in notes:
            dur_seconds = note.duration * beat_duration
            dur_samples = int(dur_seconds * sample_rate)

            if not note.is_rest:
                note_sig = generator.render(hz=note.pitch.hz, dur=dur_seconds)

                if envelope is not None:
                    note_sig = envelope.apply(note_sig)

                # Scale by velocity
                data = note_sig.data * note.velocity
                n = min(len(data), total_samples - offset)
                buf[offset : offset + n] += data[:n]

            offset += dur_samples

        return Signal(buf, sample_rate)
