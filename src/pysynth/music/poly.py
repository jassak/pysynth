from __future__ import annotations

import numpy as np

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.pitch import Note, Pitch


class PolySequencer:
    """Polyphonic sequencer with voice allocation.

    Takes piano-roll events — ``(onset_beat, Note)`` tuples — and allocates
    them across *n_voices* voices, producing per-voice CV/gate control signals.

    Parameters
    ----------
    events:
        Each entry is ``(onset_beat, note)``.  Events may overlap — the
        allocator assigns each to a free voice.  If a ``Pitch`` is given
        instead of a ``Note``, it is wrapped as ``Note(pitch, 1.0)``.
    n_voices:
        Maximum number of simultaneous voices.  If more notes overlap than
        voices are available, the voice whose current note ends soonest is
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
        voices = PolySequencer(events, n_voices=4, bpm=120).cv()
        audio = sum(
            Oscillator("saw").at(p).render(p.duration)
            * adsr(0.01, 0.1, 0.3, 0.7, 0.1).trigger(g)
            for p, g in voices
        )
    """

    def __init__(
        self,
        events: list[tuple[float, Note | Pitch]],
        n_voices: int = 4,
        bpm: float = 120.0,
        retrigger_gap: float = 0.002,
    ) -> None:
        self.n_voices = n_voices
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
            Forwarded to ``PolySequencer.__init__`` (e.g. ``n_voices``, ``bpm``).
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
        """Assign events to voices via first-free-voice allocation.

        Returns a list of length ``n_voices``, each element a list of
        ``(onset_beat, Note)`` assigned to that voice.
        """
        voices: list[list[tuple[float, Note]]] = [[] for _ in range(self.n_voices)]
        # Track when each voice becomes free (beat at which its last note ends)
        voice_free_at = [0.0] * self.n_voices

        for onset, note in self.events:
            end = onset + note.duration

            # Find a free voice: prefer the most recently used one that is
            # free (highest free_at <= onset).  This keeps sequential notes
            # on the same voice rather than spreading them unnecessarily.
            best_voice = -1
            best_free_at = -1.0
            for i in range(self.n_voices):
                if voice_free_at[i] <= onset and voice_free_at[i] > best_free_at:
                    best_voice = i
                    best_free_at = voice_free_at[i]

            if best_voice == -1:
                # No free voice — steal the one that frees up soonest
                best_voice = 0
                best_free_at = voice_free_at[0]
                for i in range(1, self.n_voices):
                    if voice_free_at[i] < best_free_at:
                        best_voice = i
                        best_free_at = voice_free_at[i]

            voices[best_voice].append((onset, note))
            voice_free_at[best_voice] = end

        return voices

    def cv(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[tuple[Signal, Signal]]:
        """Return per-voice ``(pitch, gate)`` control signals.

        All returned Signals have the same duration (the span from beat 0
        to the end of the last note).  Silent portions have
        ``pitch=0.0, gate=0.0``.
        """
        if not self.events:
            empty = Signal(np.zeros(0, dtype=np.float32), sample_rate)
            return [(empty, empty) for _ in range(self.n_voices)]

        beat_duration = 60.0 / self.bpm

        # Total duration covers from beat 0 to end of latest note
        last_beat = max(onset + note.duration for onset, note in self.events)
        total_samples = int(last_beat * beat_duration * sample_rate)

        allocated = self._allocate()
        result: list[tuple[Signal, Signal]] = []

        gap_samples = int(self.retrigger_gap * sample_rate)

        for voice_events in allocated:
            pitch_buf = np.zeros(total_samples, dtype=np.float32)
            gate_buf = np.zeros(total_samples, dtype=np.float32)

            prev_end_beat = -1.0
            for onset, note in voice_events:
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
                    prev_end_beat = onset + note.duration

            result.append((Signal(pitch_buf, sample_rate), Signal(gate_buf, sample_rate)))

        return result
