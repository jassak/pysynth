from __future__ import annotations

import random
from typing import Literal

from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.pitch import Note, Pitch
from pysynth.music.sequencer import Sequencer


Pattern = Literal["up", "down", "up_down", "down_up", "random"]


class Arpeggiator:
    """Generate an arpeggiated sequence from a chord (list of Notes or Pitches).

    The Arpeggiator takes a set of pitches and a pattern, builds a Note
    sequence according to that pattern, then delegates rendering to a
    Sequencer. Each arpeggiated note has the same duration (``note_duration``
    in beats).

    Parameters
    ----------
    notes:
        The chord to arpeggiate, given as Notes or Pitches. If Notes are
        supplied their velocity is preserved; if Pitches are supplied velocity
        defaults to 1.0.
    pattern:
        Traversal order:
        - ``"up"``       — low to high, then repeat
        - ``"down"``     — high to low, then repeat
        - ``"up_down"``  — low to high then back (endpoint not duplicated)
        - ``"down_up"``  — high to low then back
        - ``"random"``   — new random order each bar
    note_duration:
        Duration of each arpeggiated note in beats.
    bpm:
        Tempo in beats per minute.
    octaves:
        Number of octaves to span. The pattern is repeated across octave
        transpositions (each octave multiplies pitch by 2).

    Usage::

        scale = Scale(220, [1, 5/4, 3/2, 2])
        chord = [Note(scale[i], 0.25) for i in [0, 1, 2, 3]]
        sig = Arpeggiator(chord, pattern="up", bpm=140).render(
            Oscillator("triangle"),
            envelope=ADSR(0.005, 0.05, 0.1, 0.5, 0.05),
            bars=4,
        )
    """

    def __init__(
        self,
        notes: list[Note | Pitch],
        pattern: Pattern = "up",
        note_duration: float = 0.25,
        bpm: float = 120.0,
        octaves: int = 1,
    ) -> None:
        self.pattern = pattern
        self.note_duration = note_duration
        self.bpm = bpm
        self.octaves = octaves

        # Normalise input to Note objects sorted by pitch (low to high)
        raw: list[Note] = []
        for n in notes:
            if isinstance(n, Pitch):
                raw.append(Note(n, note_duration))
            else:
                raw.append(Note(n.pitch, note_duration, n.velocity))
        self._base_notes = sorted(raw, key=lambda n: n.pitch.hz)

    def _build_sequence(self, bars: int) -> list[Note]:
        """Expand the base notes across octaves and apply the pattern."""
        notes: list[Note] = []
        for octave in range(self.octaves):
            factor = 2.0 ** octave
            notes.extend(Note(n.pitch * factor, self.note_duration, n.velocity) for n in self._base_notes)

        if self.pattern == "up":
            one_cycle = notes
        elif self.pattern == "down":
            one_cycle = list(reversed(notes))
        elif self.pattern == "up_down":
            one_cycle = notes + list(reversed(notes[1:-1]))
        elif self.pattern == "down_up":
            rev = list(reversed(notes))
            one_cycle = rev + notes[1:-1]
        elif self.pattern == "random":
            one_cycle = notes  # shuffled each bar below

        # Calculate how many notes fit in the requested bars
        beats_per_bar = 4.0  # 4/4 assumed; user can adjust via note_duration
        total_beats = beats_per_bar * bars
        notes_needed = int(total_beats / self.note_duration)

        sequence: list[Note] = []
        while len(sequence) < notes_needed:
            if self.pattern == "random":
                cycle = list(one_cycle)
                random.shuffle(cycle)
            else:
                cycle = one_cycle
            sequence.extend(cycle)

        return sequence[:notes_needed]

    def render(
        self,
        generator,
        *,
        envelope=None,
        bars: int = 1,
        sample_rate: int = SAMPLE_RATE,
    ) -> Signal:
        sequence = self._build_sequence(bars)
        return Sequencer(sequence, bpm=self.bpm).render(
            generator,
            envelope=envelope,
            sample_rate=sample_rate,
        )
