import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.music.pitch import Pitch, Note
from pysynth.music.sequencer import Sequencer


SR = 1000


def _samples(beats, bpm=120.0):
    return int(beats * 60.0 / bpm * SR)


class TestSequencerCv:
    def test_returns_two_signals(self):
        notes = [Note(Pitch(440), 1.0)]
        pitch, gate = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        assert isinstance(pitch, Signal)
        assert isinstance(gate, Signal)

    def test_duration(self):
        notes = [Note(Pitch(440), 1.0), Note(Pitch(880), 1.0)]
        pitch, gate = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        expected = 2.0 * 60.0 / 120.0  # 2 beats at 120bpm = 1s
        assert abs(pitch.duration - expected) < 0.01
        assert abs(gate.duration - expected) < 0.01

    def test_pitch_values(self):
        notes = [Note(Pitch(440), 1.0), Note(Pitch(880), 1.0)]
        pitch, _ = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        mid_first = _samples(0.5)
        mid_second = _samples(1.5)
        assert pitch.data[mid_first] == pytest.approx(440.0)
        assert pitch.data[mid_second] == pytest.approx(880.0)

    def test_gate_values(self):
        notes = [Note(Pitch(440), 1.0, velocity=0.7)]
        _, gate = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        mid = _samples(0.5)
        assert gate.data[mid] == pytest.approx(0.7)

    def test_rest_produces_zero(self):
        notes = [Note(Pitch(440), 1.0), Note.rest(1.0), Note(Pitch(880), 1.0)]
        pitch, gate = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        rest_mid = _samples(1.5)
        assert pitch.data[rest_mid] == 0.0
        assert gate.data[rest_mid] == 0.0

    def test_repeats(self):
        notes = [Note(Pitch(440), 1.0)]
        pitch, gate = Sequencer(notes, bpm=120).cv(repeats=3, sample_rate=SR)
        expected = 3.0 * 60.0 / 120.0
        assert abs(pitch.duration - expected) < 0.01

    def test_repeats_pitch_values(self):
        notes = [Note(Pitch(440), 1.0), Note(Pitch(880), 1.0)]
        pitch, _ = Sequencer(notes, bpm=120).cv(repeats=2, sample_rate=SR)
        # Third beat (start of repeat) should be 440 again
        mid_third = _samples(2.5)
        assert pitch.data[mid_third] == pytest.approx(440.0)

    def test_different_bpm(self):
        notes = [Note(Pitch(440), 1.0)]
        pitch60, _ = Sequencer(notes, bpm=60).cv(sample_rate=SR)
        pitch120, _ = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        # bpm=60: 1 beat = 1s, bpm=120: 1 beat = 0.5s
        assert abs(pitch60.duration - 1.0) < 0.01
        assert abs(pitch120.duration - 0.5) < 0.01

    def test_sample_rate_respected(self):
        notes = [Note(Pitch(440), 1.0)]
        pitch, _ = Sequencer(notes, bpm=120).cv(sample_rate=2000)
        assert pitch.sample_rate == 2000

    def test_single_note_all_samples_filled(self):
        notes = [Note(Pitch(440), 2.0)]
        pitch, gate = Sequencer(notes, bpm=120).cv(sample_rate=SR)
        assert np.all(pitch.data > 0)
        assert np.all(gate.data > 0)

    def test_empty_sequence(self):
        pitch, gate = Sequencer([], bpm=120).cv(sample_rate=SR)
        assert len(pitch.data) == 0
        assert len(gate.data) == 0
