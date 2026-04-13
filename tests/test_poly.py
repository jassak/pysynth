from pysynth._core import SAMPLE_RATE, Signal
from pysynth.music.pitch import Pitch, Note
from pysynth.music.poly import PolySequencer


# Use a low sample rate for fast, readable tests
SR = 1000


def _samples(beats: float, bpm: float = 120.0) -> int:
    return int(beats * 60.0 / bpm * SR)


class TestVoiceAllocation:
    def test_three_simultaneous_notes_use_three_voices(self):
        events = [
            (0.0, Note(Pitch(440), 2.0)),
            (0.0, Note(Pitch(550), 2.0)),
            (0.0, Note(Pitch(660), 2.0)),
        ]
        ps = PolySequencer(events, n_voices=4, bpm=120)
        allocated = ps._allocate()

        # Three voices should have exactly one event each, fourth empty
        used = [v for v in allocated if v]
        assert len(used) == 3
        assert len(allocated[3]) == 0

    def test_sequential_notes_reuse_voice(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),
            (1.0, Note(Pitch(550), 1.0)),
            (2.0, Note(Pitch(660), 1.0)),
        ]
        ps = PolySequencer(events, n_voices=4, bpm=120)
        allocated = ps._allocate()

        # All notes fit in a single voice
        used = [v for v in allocated if v]
        assert len(used) == 1
        assert len(allocated[0]) == 3

    def test_voice_stealing_when_full(self):
        # 3 notes overlap, but only 2 voices available
        events = [
            (0.0, Note(Pitch(440), 4.0)),  # ends at beat 4
            (0.0, Note(Pitch(550), 2.0)),  # ends at beat 2
            (1.0, Note(Pitch(660), 1.0)),  # starts at beat 1, no free voice
        ]
        ps = PolySequencer(events, n_voices=2, bpm=120)
        allocated = ps._allocate()

        # All events should be allocated (third steals from voice ending soonest)
        total_events = sum(len(v) for v in allocated)
        assert total_events == 3


class TestCvOutput:
    def test_chord_pitch_gate_values(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),
            (0.0, Note(Pitch(550), 1.0)),
        ]
        voices = PolySequencer(events, n_voices=2, bpm=120).cv(sample_rate=SR)

        assert len(voices) == 2

        # Check pitch values at the midpoint of the note
        mid = _samples(0.5)
        pitches = sorted(voices[i][0].data[mid] for i in range(2))
        assert abs(pitches[0] - 440.0) < 0.01
        assert abs(pitches[1] - 550.0) < 0.01

        # Gate should be 1.0 during the note
        for i in range(2):
            assert voices[i][1].data[mid] == 1.0

    def test_all_voices_same_duration(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),
            (0.0, Note(Pitch(550), 3.0)),  # longer note
        ]
        voices = PolySequencer(events, n_voices=4, bpm=120).cv(sample_rate=SR)

        durations = [p.duration for p, g in voices]
        assert all(d == durations[0] for d in durations)

        # Duration should match the longest event
        expected = 3.0 * 60.0 / 120.0  # 3 beats at 120 bpm = 1.5s
        assert abs(durations[0] - expected) < 0.01

    def test_gap_produces_zero_gate(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),  # beats 0-1
            (2.0, Note(Pitch(440), 1.0)),  # beats 2-3 (gap at beat 1-2)
        ]
        voices = PolySequencer(events, n_voices=1, bpm=120).cv(sample_rate=SR)
        pitch, gate = voices[0]

        # In the gap (beat 1.5), gate should be 0
        gap_sample = _samples(1.5)
        assert gate.data[gap_sample] == 0.0
        assert pitch.data[gap_sample] == 0.0

    def test_velocity_preserved(self):
        events = [
            (0.0, Note(Pitch(440), 1.0, velocity=0.7)),
        ]
        voices = PolySequencer(events, n_voices=1, bpm=120).cv(sample_rate=SR)
        _, gate = voices[0]

        mid = _samples(0.5)
        assert abs(gate.data[mid] - 0.7) < 0.001

    def test_single_note_monophonic(self):
        events = [(0.0, Note(Pitch(440), 2.0))]
        voices = PolySequencer(events, n_voices=1, bpm=120).cv(sample_rate=SR)

        assert len(voices) == 1
        pitch, gate = voices[0]
        mid = _samples(1.0)
        assert abs(pitch.data[mid] - 440.0) < 0.01
        assert gate.data[mid] == 1.0

    def test_empty_events(self):
        voices = PolySequencer([], n_voices=2, bpm=120).cv(sample_rate=SR)
        assert len(voices) == 2
        for p, g in voices:
            assert len(p.data) == 0


class TestRetriggerGap:
    def test_sequential_notes_on_same_voice_have_gap(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),
            (1.0, Note(Pitch(550), 1.0)),
        ]
        voices = PolySequencer(events, n_voices=1, bpm=120).cv(sample_rate=SR)
        pitch, gate = voices[0]
        boundary = _samples(1.0)
        # Gate should be zero at the start of the second note
        assert gate.data[boundary] == 0.0
        # But high after the gap
        gap_end = boundary + int(0.002 * SR)
        assert gate.data[gap_end] > 0.0

    def test_non_adjacent_notes_no_gap(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),  # ends at beat 1
            (2.0, Note(Pitch(550), 1.0)),  # starts at beat 2 (gap between)
        ]
        voices = PolySequencer(events, n_voices=1, bpm=120).cv(sample_rate=SR)
        _, gate = voices[0]
        note2_start = _samples(2.0)
        # No retrigger gap needed — there's already silence between the notes
        assert gate.data[note2_start] > 0.0

    def test_retrigger_gap_zero_disables(self):
        events = [
            (0.0, Note(Pitch(440), 1.0)),
            (1.0, Note(Pitch(550), 1.0)),
        ]
        voices = PolySequencer(events, n_voices=1, bpm=120, retrigger_gap=0).cv(sample_rate=SR)
        _, gate = voices[0]
        boundary = _samples(1.0)
        assert gate.data[boundary - 1] > 0.0
        assert gate.data[boundary] > 0.0


class TestFromChords:
    def test_sequential_chords(self):
        chords = [
            ([Pitch(262), Pitch(330), Pitch(392)], 2.0),  # C major, 2 beats
            ([Pitch(349), Pitch(440), Pitch(523)], 2.0),  # F major, 2 beats
        ]
        ps = PolySequencer.from_chords(chords, n_voices=4, bpm=120)

        # Should produce 6 events total
        assert len(ps.events) == 6

        # First chord at beat 0, second at beat 2
        onsets = [onset for onset, _ in ps.events]
        assert onsets.count(0.0) == 3
        assert onsets.count(2.0) == 3

    def test_from_chords_cv_output(self):
        chords = [
            ([Pitch(440), Pitch(550)], 1.0),
        ]
        voices = PolySequencer.from_chords(chords, n_voices=2, bpm=120).cv(sample_rate=SR)

        mid = _samples(0.5)
        pitches = sorted(voices[i][0].data[mid] for i in range(2))
        assert abs(pitches[0] - 440.0) < 0.01
        assert abs(pitches[1] - 550.0) < 0.01

    def test_from_chords_with_notes(self):
        """Notes in from_chords preserve their own duration."""
        chords = [
            ([Note(Pitch(440), 0.5, velocity=0.8)], 2.0),
        ]
        ps = PolySequencer.from_chords(chords, bpm=120)

        # The Note's duration (0.5) should be preserved, not overridden to 2.0
        assert ps.events[0][1].duration == 0.5
        assert ps.events[0][1].velocity == 0.8

    def test_pitch_gets_chord_duration(self):
        """Pitches in from_chords get the chord's duration."""
        chords = [
            ([Pitch(440)], 3.0),
        ]
        ps = PolySequencer.from_chords(chords, bpm=120)
        assert ps.events[0][1].duration == 3.0
