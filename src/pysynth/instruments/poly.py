from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.pitch import Note, Pitch


class PolySequencer:
    """Polyphonic sequencer with part allocation.

    Takes piano-roll events — ``(onset_beat, Note)`` tuples — and allocates
    them across *n_parts* parts, producing per-part CV/gate control signals.

    Parameters
    ----------
    events:
        Each entry is ``(onset_beat, note)``.  Events may overlap — the
        allocator assigns each to a free part.  If a ``Pitch`` is given
        instead of a ``Note``, it is wrapped as ``Note(pitch, 1.0)``.
    n_parts:
        Maximum number of simultaneous parts.  If more notes overlap than
        parts are available, the part whose current note ends soonest is
        stolen.
    bpm:
        Tempo in beats per minute (converts beat durations to seconds).

    Usage::

        from pysynth import PolySequencer, Oscillator, adsr, Pitch, Note

        events = [
            (0.0, Note(Pitch(262), 2.0)),
            (0.0, Note(Pitch(330), 2.0)),
            (0.0, Note(Pitch(392), 2.0)),
        ]
        parts = PolySequencer(events, n_parts=4, bpm=120).cv()
        audio = sum(
            Oscillator("saw").render(p.duration, p)
            * adsr(0.01, 0.1, 0.7, 0.1).trigger(g)
            for p, g in parts
        )
    """

    def __init__(
        self,
        events: list[tuple[float, Note | Pitch]],
        n_parts: int = 4,
        bpm: float = 120.0,
        retrigger_gap: float = 0.002,
    ) -> None:
        self.n_parts = n_parts
        self.bpm = bpm
        self.retrigger_gap = retrigger_gap

        # Normalise to (onset, Note) sorted by onset
        normalised: list[tuple[float, Note]] = []
        for onset, n in events:
            if isinstance(n, Pitch):
                normalised.append((onset, Note(n, 1.0)))
            else:
                normalised.append((onset, n))
        self.events = sorted(normalised, key=lambda e: e[0])

    @classmethod
    def from_chords(
        cls,
        chords: list[tuple[list[Pitch | Note], float]],
        **kwargs,
    ) -> PolySequencer:
        """Build events from a chord progression.

        Each entry is ``(pitches_or_notes, duration_in_beats)``.  Chords are
        laid out sequentially — each chord starts where the previous ended.

        Parameters
        ----------
        chords:
            List of ``(notes, duration)`` pairs.  Each *notes* element can be
            a ``Pitch`` (wrapped as ``Note(pitch, duration)``) or a ``Note``
            (whose duration is used as-is).
        **kwargs:
            Forwarded to ``PolySequencer.__init__`` (e.g. ``n_parts``, ``bpm``).
        """
        events: list[tuple[float, Note | Pitch]] = []
        beat_cursor = 0.0
        for pitches, duration in chords:
            for p in pitches:
                if isinstance(p, Pitch):
                    events.append((beat_cursor, Note(p, duration)))
                else:
                    events.append((beat_cursor, p))
            beat_cursor += duration
        return cls(events, **kwargs)

    def _allocate(self) -> list[list[tuple[float, Note]]]:
        """Assign events to parts via first-free-part allocation.

        Returns a list of length ``n_parts``, each element a list of
        ``(onset_beat, Note)`` assigned to that part.
        """
        parts: list[list[tuple[float, Note]]] = [[] for _ in range(self.n_parts)]
        # Track when each part becomes free (beat at which its last note ends)
        part_free_at = [0.0] * self.n_parts

        for onset, note in self.events:
            end = onset + note.duration

            # Find a free part: prefer the most recently used one that is
            # free (highest free_at <= onset).  This keeps sequential notes
            # on the same part rather than spreading them unnecessarily.
            best_part = -1
            best_free_at = -1.0
            for i in range(self.n_parts):
                if part_free_at[i] <= onset and part_free_at[i] > best_free_at:
                    best_part = i
                    best_free_at = part_free_at[i]

            if best_part == -1:
                # No free part — steal the one that frees up soonest
                best_part = 0
                best_free_at = part_free_at[0]
                for i in range(1, self.n_parts):
                    if part_free_at[i] < best_free_at:
                        best_part = i
                        best_free_at = part_free_at[i]

            parts[best_part].append((onset, note))
            part_free_at[best_part] = end

        return parts

    def cv(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[tuple[Signal, Signal]]:
        """Return per-part ``(pitch, gate)`` control signals.

        All returned Signals have the same duration (the span from beat 0
        to the end of the last note).  Silent portions have
        ``pitch=0.0, gate=0.0``.
        """
        if not self.events:
            empty = Signal(np.zeros(0, dtype=np.float32), sample_rate)
            return [(empty, empty) for _ in range(self.n_parts)]

        beat_duration = 60.0 / self.bpm

        # Total duration covers from beat 0 to end of latest note
        last_beat = max(onset + note.duration for onset, note in self.events)
        total_samples = int(last_beat * beat_duration * sample_rate)

        allocated = self._allocate()
        result: list[tuple[Signal, Signal]] = []

        gap_samples = int(self.retrigger_gap * sample_rate)

        for part_events in allocated:
            pitch_buf = np.zeros(total_samples, dtype=np.float32)
            gate_buf = np.zeros(total_samples, dtype=np.float32)

            prev_end_beat = -1.0
            last_hz = 0.0
            for onset, note in part_events:
                start = int(onset * beat_duration * sample_rate)
                dur_samples = int(note.duration * beat_duration * sample_rate)
                end = min(start + dur_samples, total_samples)
                if not note.is_rest:
                    pitch_buf[start:end] = note.pitch.hz
                    gate_buf[start:end] = note.velocity
                    # Insert retrigger gap when this note starts exactly
                    # where the previous note on this voice ended.
                    if prev_end_beat >= onset and gap_samples > 0:
                        gap_end = min(start + gap_samples, end)
                        gate_buf[start:gap_end] = 0.0
                    last_hz = note.pitch.hz
                    prev_end_beat = onset + note.duration

            # Sample-and-hold: forward-fill pitch so the oscillator
            # continues producing audio through gaps for the envelope's
            # release phase.
            if last_hz > 0.0:
                mask = pitch_buf != 0.0
                if mask.any():
                    idx = np.where(mask, np.arange(total_samples), 0)
                    np.maximum.accumulate(idx, out=idx)
                    pitch_buf = pitch_buf[idx]

            result.append((Signal(pitch_buf, sample_rate), Signal(gate_buf, sample_rate)))

        return result
