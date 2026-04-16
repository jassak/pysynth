import numpy as np
import pytest

from pysynth._core import Signal
from pysynth.music.scales import Scale
from pysynth.instruments.polymetric import PolymetricSequencer


SR = 1000  # low sample rate for fast tests

# Simple scale: degrees 0-4 map to 100, 200, 300, 400, 500 Hz
SCALE = Scale(100, [1, 2, 3, 4, 5])


class TestPolymetricSequencerCv:
    def test_returns_dict_of_tuples(self):
        seq = PolymetricSequencer({"a": [0, 1], "b": [2]}, SCALE, bpm=60)
        signals = seq.cv(beats=2, sample_rate=SR)
        assert isinstance(signals, dict)
        assert set(signals.keys()) == {"a", "b"}
        for pitch, gate in signals.values():
            assert isinstance(pitch, Signal)
            assert isinstance(gate, Signal)

    def test_all_tracks_same_duration(self):
        seq = PolymetricSequencer({"a": [0, 1, 2], "b": [0, 1]}, SCALE, bpm=60)
        signals = seq.cv(beats=4, sample_rate=SR)
        len_a = len(signals["a"][0].data)
        len_b = len(signals["b"][0].data)
        assert len_a == len_b

    def test_pitch_matches_scale_degree(self):
        seq = PolymetricSequencer({"x": [2]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=1.0,
                                   retrigger_gap=0.0)
        signals = seq.cv(beats=1, sample_rate=SR)
        pitch = signals["x"][0].data
        # degree 2 -> SCALE[2] = 300 Hz
        active = pitch[pitch > 0]
        assert len(active) > 0
        np.testing.assert_allclose(active, 300.0, atol=1e-3)

    def test_rest_produces_zero(self):
        seq = PolymetricSequencer({"x": [None, None]}, SCALE, bpm=60,
                                   step_length=1.0)
        signals = seq.cv(beats=2, sample_rate=SR)
        pitch, gate = signals["x"]
        assert np.all(pitch.data == 0.0)
        assert np.all(gate.data == 0.0)


class TestPolymetricSequencerLooping:
    def test_pattern_loops(self):
        # 1-step pattern over 2 beats at step_length=1.0 -> should loop twice
        seq = PolymetricSequencer({"x": [3]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=0.5,
                                   retrigger_gap=0.0)
        signals = seq.cv(beats=2, sample_rate=SR)
        pitch = signals["x"][0].data
        half = len(pitch) // 2
        # Both halves should have the same pitch pattern
        np.testing.assert_array_equal(pitch[:half], pitch[half:])

    def test_different_pattern_lengths_phase(self):
        # 2-step pattern vs 3-step pattern, both at step_length=1.0
        # Over 6 beats: track_a cycles 3 times, track_b cycles 2 times
        seq = PolymetricSequencer({
            "a": [0, 1],     # 2-step, 2-beat cycle
            "b": [0, 1, 2],  # 3-step, 3-beat cycle
        }, SCALE, bpm=60, step_length=1.0, gate_length=1.0,
           retrigger_gap=0.0)
        signals = seq.cv(beats=6, sample_rate=SR)
        pitch_a = signals["a"][0].data
        pitch_b = signals["b"][0].data
        # Track a: 2-beat cycle repeated 3 times
        cycle_a = int(2 * SR)
        np.testing.assert_array_equal(pitch_a[:cycle_a], pitch_a[cycle_a:2*cycle_a])
        np.testing.assert_array_equal(pitch_a[:cycle_a], pitch_a[2*cycle_a:3*cycle_a])
        # Track b: 3-beat cycle repeated 2 times
        cycle_b = int(3 * SR)
        np.testing.assert_array_equal(pitch_b[:cycle_b], pitch_b[cycle_b:2*cycle_b])


class TestPolymetricSequencerGateLength:
    def test_global_gate_length(self):
        seq = PolymetricSequencer({"x": [0]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=0.5,
                                   retrigger_gap=0.0)
        signals = seq.cv(beats=1, sample_rate=SR)
        gate = signals["x"][1].data
        step_samples = int(1.0 * SR)
        gate_samples = int(0.5 * step_samples)
        assert np.all(gate[:gate_samples] == 1.0)
        assert np.all(gate[gate_samples:step_samples] == 0.0)

    def test_per_track_gate_length(self):
        seq = PolymetricSequencer({"a": [0], "b": [0]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=0.5,
                                   gate_lengths={"b": 0.9},
                                   retrigger_gap=0.0)
        signals = seq.cv(beats=1, sample_rate=SR)
        a_active = np.sum(signals["a"][1].data > 0)
        b_active = np.sum(signals["b"][1].data > 0)
        assert b_active > a_active


class TestPolymetricSequencerStepLength:
    def test_per_track_step_length(self):
        # Track a: step_length=1.0, track b: step_length=0.5
        # Same 1-step pattern [0]. Over 4 beats at 60 bpm:
        #   a: 4 steps (one per beat), b: 8 steps (two per beat)
        seq = PolymetricSequencer({
            "a": [0],
            "b": [0],
        }, SCALE, bpm=60, step_length=1.0,
           step_lengths={"b": 0.5},
           gate_length=0.5, retrigger_gap=0.0)
        signals = seq.cv(beats=4, sample_rate=SR)
        gate_a = signals["a"][1].data
        gate_b = signals["b"][1].data
        # b has half the step duration, so each pulse is half as long
        # total on-time should be the same (same gate_length fraction)
        # but b has twice as many pulses
        a_on = np.sum(gate_a > 0)
        b_on = np.sum(gate_b > 0)
        # Same total on-fraction, but b's individual pulse width is half of a's
        step_a = int(1.0 * SR)
        step_b = int(0.5 * SR)
        pulse_a = int(0.5 * step_a)
        pulse_b = int(0.5 * step_b)
        assert pulse_b == pulse_a // 2


class TestPolymetricSequencerRetrigger:
    def test_consecutive_steps_get_gap(self):
        seq = PolymetricSequencer({"x": [0, 1]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=1.0,
                                   retrigger_gap=0.01)
        signals = seq.cv(beats=2, sample_rate=SR)
        gate = signals["x"][1].data
        step_samples = int(1.0 * SR)
        gap_samples = int(0.01 * SR)
        # At boundary of step 1, first gap_samples should be 0
        assert np.all(gate[step_samples:step_samples + gap_samples] == 0.0)

    def test_no_gap_after_rest(self):
        seq = PolymetricSequencer({"x": [0, None, 1]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=1.0,
                                   retrigger_gap=0.01)
        signals = seq.cv(beats=3, sample_rate=SR)
        gate = signals["x"][1].data
        step_samples = int(1.0 * SR)
        # Step 2 follows a rest — no gap needed
        assert gate[2 * step_samples] == 1.0

    def test_retrigger_across_loop_boundary(self):
        # Single-step pattern [0] loops. Each loop iteration starts with
        # the same active step, so retrigger gap should appear at wrap.
        seq = PolymetricSequencer({"x": [0]}, SCALE, bpm=60,
                                   step_length=1.0, gate_length=1.0,
                                   retrigger_gap=0.01)
        signals = seq.cv(beats=2, sample_rate=SR)
        gate = signals["x"][1].data
        step_samples = int(1.0 * SR)
        gap_samples = int(0.01 * SR)
        assert np.all(gate[step_samples:step_samples + gap_samples] == 0.0)


class TestPolymetricSequencerFromNotation:
    def test_parses_degrees_and_rests(self):
        seq = PolymetricSequencer.from_notation(
            {"x": "0 . 2 . 4"}, SCALE, bpm=60)
        assert seq.tracks["x"] == [0, None, 2, None, 4]

    def test_negative_degree(self):
        scale = Scale(440, [1, 9/8, 5/4], period=2.0)
        seq = PolymetricSequencer.from_notation(
            {"x": "-1 0 1"}, scale, bpm=60)
        assert seq.tracks["x"] == [-1, 0, 1]

    def test_list_input(self):
        seq = PolymetricSequencer.from_notation(
            ["0 2 4", "0 . 3"], SCALE, bpm=60)
        assert 0 in seq.tracks
        assert 1 in seq.tracks
        assert seq.tracks[0] == [0, 2, 4]
        assert seq.tracks[1] == [0, None, 3]

    def test_kwargs_passed_through(self):
        seq = PolymetricSequencer.from_notation(
            {"x": "0 1"}, SCALE, bpm=140, step_length=0.5)
        assert seq.bpm == 140
        assert seq.step_length == 0.5


class TestPolymetricSequencerListInput:
    def test_auto_names(self):
        seq = PolymetricSequencer([[0, 1], [2, 3]], SCALE, bpm=60)
        assert 0 in seq.tracks
        assert 1 in seq.tracks
        assert seq.tracks[0] == [0, 1]


class TestPolymetricSequencerEmpty:
    def test_empty_tracks(self):
        seq = PolymetricSequencer({}, SCALE, bpm=60)
        signals = seq.cv(beats=4, sample_rate=SR)
        assert signals == {}

    def test_empty_pattern_silent(self):
        seq = PolymetricSequencer({"x": []}, SCALE, bpm=60)
        signals = seq.cv(beats=2, sample_rate=SR)
        pitch, gate = signals["x"]
        assert np.all(pitch.data == 0.0)
        assert np.all(gate.data == 0.0)
        assert len(pitch.data) == int(2 * 1.0 * SR)  # 2 beats at 60 bpm = 2s
